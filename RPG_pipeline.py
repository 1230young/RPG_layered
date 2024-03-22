import sys
sys.path.append('/pyy/openseg_blob/yuyang/code/RPG-DiffusionMaster/TypoClipSDXL/typoclip_sdxl')
sys.path.append('/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/repositories/generative-models/sgm/modules/')

import logging
import os
import os.path as osp
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
from modules import timer, errors
import torch
startup_timer = timer.Timer()
from modules import extensions
import modules.scripts
from mllm import GPT4,local_llm
import argparse
# from template.demo import demo_list
import time
from mmcv import Config
import json
from inference_data import load_inference_data, draw_bbox, load_inference_data_with_glyph, load_intention_output_data, load_gpt_output
from p2layout import intension2


def resize_bbox(bbox, height, width, source_height=1457, source_width=1457, scale=1):
    """
    Resize bounding box from source image to target image
    Args:
        bbox (list): [top, left, bottom, right]
        height (int): target image height
        width (int): target image width
        source_height (int): source image height
        source_width (int): source image width
    Returns:
        list: [top, left, bottom, right]
    """
    bbox = [bbox[0] * height / source_height, bbox[1] * width / source_width,
            bbox[2] * height / source_height, bbox[3] * width / source_width]
    bbox[0]=int(min(max(0,bbox[0]),height)/scale)
    bbox[1]=int(min(max(0,bbox[1]),width)/scale)
    bbox[2]=int(min(max(0,bbox[2]),height)/scale)
    bbox[3]=int(min(max(0,bbox[3]),width)/scale)
    if bbox[0]>=bbox[2]:
        if bbox[0]>0:
            bbox[0]-=1
        else:   
            bbox[2]+=1
    if bbox[1]>=bbox[3]:
        if bbox[1]>0:
            bbox[1]-=1
        else:
            bbox[3]+=1   
    return bbox
def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config
def initialize(model_name=None, config_dir=None, ckpt_dir=None, vae_dir=None, reload=False):
    
    from modules import shared
    from modules.shared import cmd_opts
    
    from modules import options, shared_options
    shared.options_templates = shared_options.options_templates #{}
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    if os.path.exists(shared.config_filename):
        shared.opts.load(shared.config_filename)
    extensions.list_extensions()
    startup_timer.record("list extensions")
    
    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

    shared.device = devices.device
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"
    from modules import shared_state
    shared.state = shared_state.State()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)

    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()
    import modules.sd_models
    if not reload:
        
        modules.sd_models.setup_model() # load models
        modules.sd_models.list_models()
        startup_timer.record("list SD models")
    
        modules.scripts.load_scripts()
        modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
        
        startup_timer.record("load scripts")
        print('txt2img_scripts',modules.scripts.scripts_txt2img.scripts)
    
    try:
        modules.sd_models.load_model(model_name=model_name, config_dir=config_dir, ckpt_dir=ckpt_dir, vae_dir=vae_dir)
        #load lora
        temp=0
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)
    

    startup_timer.record("load SD checkpoint")
 
 
def RPG(user_prompt,diffusion_model,split_ratio=None,activate=True,use_base=False,base_ratio=0,base_prompt=None,batch_size=1,seed=1234, use_personalized=False,cfg=5,steps=20,height=1024,width=1024,use_layer=False,bboxes=None,sampler_index=0):
     # set model
    import modules.txt2img
    #TODO: add personalized regional split and regional prompt 

    regional_prompt=user_prompt
    layer_num=len(bboxes)
    split_ratio=''
    for i in range(layer_num):
        if i<layer_num-1:
            split_ratio+=f'1,1; '
        else:
            split_ratio+='1,1'
    bboxes=[resize_bbox(bbox,height,width) for bbox in bboxes]
    textprompt=None

    if use_base:
        if base_prompt is None:
            regional_prompt= user_prompt+' BREAK\n'+regional_prompt
        else:
            regional_prompt= base_prompt+' BREAK\n'+regional_prompt
        
    # Regional settings:
    regional_settings = {
    'activate':activate, # To activate regional diffusion, set True, if don't use this, set False
    'split_ratio':split_ratio, # Split ratio for regional diffusion, default is 1,1, which means vertically split the image into two regions with same height and width,
    'base_ratio':base_ratio, # The weight of base prompt
    'use_base':use_base, # Whether to use base prompt
    'use_common':False, # Whether to use common prompt
    'use_layer':True, # Whether to use layered data
    'bboxes':bboxes, # The bounding box of layered data
    }
    
    image, _, _, _ = modules.txt2img.txt2img(
        id_task="task",
        prompt=regional_prompt,
        negative_prompt="",
        prompt_styles=[],
        steps=steps,
        sampler_index=sampler_index,
        restore_faces=False,
        tiling=False,
        n_iter=1,
        batch_size=batch_size,
        cfg_scale=cfg,
        seed=seed, # -1 means random, choose a number larger than 0 to get a deterministic result
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        height=height,
        width=width,
        enable_hr=False,
        denoising_strength=0.7,
        hr_scale=0,
        hr_upscaler="Latent",
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        override_settings_texts=[],
        **regional_settings,
    )
    return image, regional_prompt, split_ratio, textprompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_config_file", type=str, default='test/glyph_mixed_model_sdxl-lora-128_noise-offset_train-byt5-mapper_frominit_mix_sdxl_canva_w1-1_8ep_4x16_premiumx2_no_glyph.py')
    parser.add_argument('--user_prompt', type=str,help='input user prompt', default="")
    parser.add_argument('--intention', type=str,help='user intention', default="")
    parser.add_argument('--model_name', type=str,default='albedobaseXL_v20.safetensors',help='the name of the ckpt, in the folder of models/Stable-diffusion')
    parser.add_argument('--activate',default=True,type=bool,help='whether to activate regional diffusion')
    parser.add_argument('--use_base',action='store_true',help='whether to use base prompt')
    parser.add_argument('--base_ratio',default=0.3,type=float,help='the weight of base prompt')
    parser.add_argument('--base_prompt',default=None,type=str,help='the base prompt')
    parser.add_argument('--batch_size',default=1,type=int,help='the batch size of txt2img')
    parser.add_argument('--seed',default=1234,type=int,help='the seed of txt2img')
    parser.add_argument('--cfg',default=5,type=float,help='context-free guidance scale')
    parser.add_argument('--steps',default=20,type=int,help='the steps of txt2img')
    parser.add_argument('--height',default=1024,type=int,help='the height of the generated image')
    parser.add_argument('--width',default=1024,type=int,help='the width of the generated image')
    parser.add_argument('--gpt_output_path',default="outputs/sample.json",type=str,help='json path to log the gpt output info')
    parser.add_argument("--config_dir", default=None,type=str)
    parser.add_argument("--ckpt_dir", default=None, type=str)
    parser.add_argument("--vae_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--load_typo_sdxl_pretrain_ckpt", default=None, type=str)
    
    
    opt = parser.parse_args()
    # opt.RPG_config_file="test/glyph_mixed_model_sdxl-lora-128_noise-offset_train-byt5-mapper_frominit_mix_sdxl_canva_w1-1_8ep_4x16_premiumx2.py"
    opt.all_config_file=opt.user_prompt
    opt.all_config_file="test/try_pipeline.py"
    config=read_config(opt.all_config_file)
    '''--use_base the function of this boolean variable is to activate the base prompt in diffusion process. Utilizing the base prompt signifies that we avoid the direct amalgamation of subregions as the latent representation. Instead, we use a foundational prompt that summarizes the image's key components and obatin the overall structure latent of the image. We then compute the weighted aggregate of these latents to yield the conclusive output. This method is instrumental in addressing the problems like omission of entities in complicated prompt generation tasks, and it also contributes to refining the edges of each subregion, ensuring they are seamlessly integrated and resonate harmony.

    --base_ratio the weight of the base prompt latent, if too small, it is difficult to work, if too big, it will confuse the composition and properties of subregions. We conduct ablation experiment in our paper, see our paper for more detailed information and analysis.'''
    for key in config.keys():
        setattr(opt,key,config[key])
    import random
    if opt.seed==-1:
        opt.seed=random.randint(0,10000)
    intention=opt.intention
    if isinstance(intention, str):
        intention = [intention]
    user_prompt=opt.user_prompt #This is what we need in all the situations except for demo
    model_name=opt.model_name #This is what we need in all the situations except for demo
    activate=opt.activate #If you want to try direct generation, set False
    use_base=opt.use_base # If you want to use base prompt, set True
    base_ratio=opt.base_ratio # The weight of base prompt
    base_prompt=opt.base_prompt # The base prompt, if you don't input this, we will use the user prompt as the base prompt
    batch_size=opt.batch_size # The batch size of txt2img
    seed=opt.seed # The seed of txt2img
    cfg=opt.cfg # The context-free guidance scale
    steps=opt.steps # The steps of txt2img
    height=opt.height
    width=opt.width

    if opt.config_dir is not None and opt.ckpt_dir is not None:
        # gpt_output=[intension2(itt) for itt in intention]
        gpt_output=[[ { "background": "The layer is completely light blue color, representing a clear sky." }, { "layer": 0, "category": "element", "caption": "The layer features a cartoon-style illustration of a chicken with a round body, large eyes, and a bright red comb and wattle. The chicken is standing on one leg, with the other leg lifted in a playful pose, and it's colored in a combination of white and light yellow shades.", "top_left": [ 150, 450 ], "bottom_right": [ 450, 1000 ] }, { "layer": 1, "category": "element", "caption": "Another cartoon-style chicken with a round body, large eyes, and a bright red comb and wattle. This chicken is standing on both legs, facing the left side of the poster, and is colored in various shades of brown.", "top_left": [ 600, 530 ], "bottom_right": [ 900, 1080 ] }, { "layer": 2, "category": "element", "caption": "The third cartoon chicken, featuring a round body, large eyes, and a bright red comb and wattle. The chicken is standing on both legs, facing the right side of the poster, and is colored in a combination of light orange and white shades.", "top_left": [ 1050, 460 ], "bottom_right": [ 1350, 1010 ] }, { "layer": 3, "category": "element", "caption": "The first cartoon-style duck, characterized by an elongated body, large eyes, and a bright orange beak. The duck is standing on both legs, facing the left side of the poster, and is colored in various shades of green.", "top_left": [ 300, 1100 ], "bottom_right": [ 600, 1450 ] }, { "layer": 4, "category": "element", "caption": "A second cartoon-style duck with an elongated body, large eyes, and a bright orange beak. The duck is standing on both legs, facing the right side of the poster, and is colored in a combination of light blue and white shades.", "top_left": [ 900, 1120 ], "bottom_right": [ 1200, 1470 ] } ],
                    [ { "background": "The layer is completely light beige color." }, { "layer": 0, "category": "element", "caption": "The layer features a wooden table with a realistic wood grain texture in warm brown hues, covering the bottom half of the canvas to create the impression of a cozy, intimate setting.", "top_left": [ 0, 728 ], "bottom_right": [ 1456, 1457 ] }, { "layer": 1, "category": "element", "caption": "An open book with visible pages and text, resting on the wooden table. The book's cover and binding are a soft, earthy brown color, blending harmoniously with the warm tones of the wooden surface.", "top_left": [ 488, 805 ], "bottom_right": [ 968, 1207 ] }, { "layer": 2, "category": "element", "caption": "An orange-hued cat sitting attentively in front of the open book, its body facing the viewer but its head turned to the side, as if gazing curiously at the pages. The cat's fur is rendered in rich, warm tones, with darkened shadows and highlights to suggest depth and texture. The cat's eyes are bright and inquisitive, further emphasizing the intellectual and cozy ambiance of the scene.", "top_left": [ 679, 557 ], "bottom_right": [ 1249, 1046 ] }, { "layer": 3, "category": "element", "caption": "A soft, warm light source is illustrated, casting a gentle glow onto the scene. The light is diffuse, creating a cozy and inviting atmosphere that accentuates the rich colors and textures of the cat and the wooden table.", "top_left": [ 0, 0 ], "bottom_right": [ 1456, 1457 ] } ],
                    [ { "background": "The layer is completely light blue color." }, { "layer": 0, "category": "element", "caption": "Five whole apples arranged in a neat row, with the first three being red and the last two being green. All apples have a simple leaf on top, and the colors are bright and eye-catching to appeal to young children.", "top_left": [ 578, 250 ], "bottom_right": [ 878, 450 ] }, { "layer": 1, "category": "element", "caption": "A large hand-drawn arrow, colored in yellow, pointing from the two green apples to the right, indicating that they are being taken away from the group of five apples.", "top_left": [ 889, 295 ], "bottom_right": [ 1005, 405 ] }, { "layer": 2, "category": "element", "caption": "Two green apples, separate from the original group and placed to the right of the arrow, illustrating that they have been removed from the group.", "top_left": [ 1020, 250 ], "bottom_right": [ 1245, 450 ] }, { "layer": 3, "category": "text", "caption": "Text \"Subtraction\n\" in <color-31>, <font-97>. ", "top_left": [ 50, 100 ], "bottom_right": [ 550, 200 ] }, { "layer": 4, "category": "text", "caption": "Text \"5 - 2 = 3\n\" in <color-31>, <font-97>. ", "top_left": [ 400, 500 ], "bottom_right": [ 1100, 600 ] }, { "layer": 5, "category": "text", "caption": "Text \"Take away two apples\nfrom five apples\n\" in <color-31>, <font-59>. ", "top_left": [ 50, 700 ], "bottom_right": [ 550, 800 ] }, { "layer": 6, "category": "text", "caption": "Text \"You have three apples left!\n\" in <color-31>, <font-59>. ", "top_left": [ 50, 850 ], "bottom_right": [ 550, 950 ] }, { "layer": 7, "category": "element", "caption": "A yellow sun with a smiling face, placed at the top right corner of the poster, adding a friendly and cheerful atmosphere to the design.", "top_left": [ 1256, 56 ], "bottom_right": [ 1456, 256 ] } ],
                    [ { "background": "The layer is a stylized stadium full of cheering fans, using a blend of blue and green colors to represent the atmosphere. The stadium is designed with lines and shapes to suggest depth and perspective, creating a dynamic and engaging scene." }, { "layer": 0, "category": "element", "caption": "A cartoon rabbit dressed in running gear, wearing a red tank top, blue shorts, and a white headband. The rabbit is in a running position, smiling and passing the baton to the next animal.", "top_left": [ 95, 450 ], "bottom_right": [ 395, 1050 ] }, { "layer": 1, "category": "element", "caption": "A cartoon bear dressed in soccer attire, wearing a green jersey, white shorts, and soccer cleats. The bear is standing on one foot while receiving the baton from the rabbit with an enthusiastic expression.", "top_left": [ 390, 455 ], "bottom_right": [ 690, 1100 ] }, { "layer": 2, "category": "element", "caption": "A cartoon fox dressed in biking gear, wearing a yellow cycling jersey, black shorts, a helmet, and cycling gloves. The fox is holding the handlebars of a bike with one hand and receiving the baton from the bear with the other hand, showing a determined and focused expression.", "top_left": [ 675, 400 ], "bottom_right": [ 1175, 1075 ] }, { "layer": 3, "category": "text", "caption": "Text \"Go Farther Together!\n\" in <color-2>, <font-137>. ", "top_left": [ 355, 125 ], "bottom_right": [ 1100, 250 ] }, { "layer": 4, "category": "element", "caption": "A cartoon baton in red and white colors, representing collaboration and support, being passed between the animals in the relay race.", "top_left": [ 355, 725 ], "bottom_right": [ 685, 775 ] } ]]
        
        processed_data=[]
        if isinstance(opt.base_prompt, list) and len(opt.base_prompt)==len(gpt_output):
            for i in range(len(gpt_output)):
                processed_data.append(load_gpt_output(gpt_output[i],intention[0],opt.base_prompt[i]))
        else:
            for i in range(len(gpt_output)):
                processed_data.append(load_gpt_output(gpt_output[i],intention[0],))
    else:
        raise ValueError('config_dir and ckpt_dir should be given')
    user_prompt=[i['Layer Prompt'] for i in processed_data]
    bboxes=[i['bboxes'] for i in processed_data]
    base_prompts=[i['Base Prompt'] for i in processed_data]
    index_list=[i['index'] for i in processed_data]


    ckpt_dir=[opt.load_typo_sdxl_pretrain_ckpt,opt.ckpt_dir] if opt.load_typo_sdxl_pretrain_ckpt is not None else opt.ckpt_dir
    initialize(model_name=model_name, config_dir=opt.config_dir, ckpt_dir=ckpt_dir, vae_dir=opt.vae_dir)
    if isinstance(user_prompt, str):
        user_prompts = [user_prompt]
    elif isinstance(user_prompt, list):
        user_prompts = user_prompt
    opt.gpt_output_path = f"/pyy/openseg_blob/yuyang/code/RPG/{opt.output_dir}/gpt_output.json"
    if not os.path.exists(opt.gpt_output_path):
        log_json = []
    else:
        with open(opt.gpt_output_path, 'r') as f:
            log_json = json.load(f)
    for sampler_index in [0]:
    # for cfg in [18]:
        directory = opt.output_dir
        
        for n,user_prompt in enumerate(user_prompts):
            
            bbox=bboxes[n]
            if use_base:
                base_prompt=base_prompts[n]
            else:
                base_prompt=None
            
            seed=opt.seed
            image,regional_prompt, split_ratio, textprompt=RPG(user_prompt=user_prompt,
            diffusion_model=model_name,
            split_ratio=None,
            use_base=use_base,
            base_ratio=base_ratio,
            base_prompt=base_prompt,
            batch_size=batch_size,
            seed=seed,
            use_personalized=False,
            cfg=cfg,
            steps=steps,
            height=height,
            width=width,
            bboxes=bbox,
            sampler_index=sampler_index
            )
            l=len(image)
            target_dirs=["generated_imgs","/pyy/openseg_blob/yuyang/code/RPG"]
            for tar_dir in target_dirs:
                for i in range(len(image)):
                    os.makedirs(f"{tar_dir}/{directory}", exist_ok=True)
                    index_list[n]=len(os.listdir(f"{tar_dir}/{directory}"))
                    file_name = f"{index_list[n]}.png"
                    path=f"{directory}/{file_name}"
                    image[i].save(f"{tar_dir}/{path}")
        for i in range(len(gpt_output)):
            log_json.append({"intention":intention[i],"index":index_list[i],"gpt_output":gpt_output[i]})
            

        for i in range(len(bboxes)):
            index=index_list[i]
            draw_bboxes=bboxes[i]
            source_dir=target_dirs[-1]+"/"+directory
            target_dir=source_dir+"_bbox"
            draw_bbox(index,draw_bboxes,source_dir,target_dir)

    with open(opt.gpt_output_path, 'w') as f:
        json.dump(log_json, f)
    with open(f"generated_imgs/{directory}/gpt_output.json", 'w') as f:
        json.dump(log_json, f)



