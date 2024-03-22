import numpy as np
import gradio as gr
from RPG_pipeline import *
checkpoint_model=""
SAMPLERS=["DPM++ 2M Karras", "DPM++ SDE Karras", "DPM++ 2M SDE Exponential", "DPM++ 2M SDE Karras", "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 2M SDE Heun Karras", "DPM++ 2M SDE Heun Exponential", "DPM++ 3M SDE", "DPM++ 3M SDE Karras", "DPM++ 3M SDE Exponential", "DPM fast", "DPM adaptive", "LMS Karras", "DPM2 Karras", "DPM2 a Karras", "DPM++ 2S a Karras", "Restart", "DDIM", "PLMS", "UniPC"]
from inference_data import COLOR_LIST
def generate(checkpoint, seed, intention, cfg, steps, sampler, base_prompt, base_ratio):
    global checkpoint_model
    import random
    try:
        seed=int(seed)
    except:
        seed=-1
    config_file="test/try_pipeline.py"
    opt=read_config(config_file)
    if seed==-1:
        seed=random.randint(0,10000)
    user_prompt=opt.user_prompt
    model_name=checkpoint+".safetensors"
    activate=True
    use_base=True
    batch_size=1
    height=1024
    width=1024

    gpt_output=intension2(intention)
    if len(base_prompt)>0:
        processed_data=load_gpt_output(gpt_output,intention,base_prompt)
    
    else:
        processed_data=load_gpt_output(gpt_output,intention)
    user_prompt=processed_data['Layer Prompt']
    bboxes=processed_data['bboxes']
    base_prompt=processed_data['Base Prompt']
    if checkpoint=="sd_xl_base_1.0":
        opt.vae_dir="madebyollin.safetensors"
    else:
        opt.vae_dir=None
    ckpt_dir=[opt.load_typo_sdxl_pretrain_ckpt,opt.ckpt_dir] if opt.load_typo_sdxl_pretrain_ckpt is not None else opt.ckpt_dir
    if checkpoint!=checkpoint_model:
        reload=checkpoint_model!=""
        checkpoint_model=checkpoint
        initialize(model_name=model_name, config_dir=opt.config_dir, ckpt_dir=ckpt_dir, vae_dir=opt.vae_dir, reload=reload)
    sampler=SAMPLERS.index(sampler)
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
    bboxes=bboxes,
    sampler_index=sampler
    )
    image=image[0]
    from PIL import Image, ImageDraw, ImageFont
    font=ImageFont.truetype('font/arial.ttf', size=30)
    img=image.copy()
    # img=Image.new('RGB', (1024, 1024), (255, 255, 255))
    draw=ImageDraw.Draw(img)
    for n,bbox in enumerate(bboxes):
        bbox=[int(x*height/1457) for x in bbox]
        label=f"Layer {n}"
        target_bbox=[bbox[1],bbox[0],bbox[3],bbox[2]]
        # 获取label长宽
        label_size = draw.textsize(label, font)

        # 设置label起点
        text_origin = np.array([target_bbox[0], target_bbox[1]])

        # 绘制矩形框，加入label文本
        draw.rectangle([target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]],outline=COLOR_LIST[n],width=4)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=COLOR_LIST[n])
        draw.text(text_origin, str(label), fill=(255, 255, 255), font=font)
    del draw
    return gpt_output, image, img



#ui部分
with gr.Blocks() as demo:

    gr.Markdown("RPG pipeline demo: Intention to layout to image generation.")
    with gr.Row():
        checkpoint=gr.Dropdown(["sd_xl_base_1.0","albedobaseXL_v20","playground-v2.fp16"], default="sd_xl_base_1.0", label="Checkpoint")
        seed=gr.Textbox("-1", label="Seed")

    intention=gr.Textbox(placeholder="Intention\nExample: Create an advertisement for an Autumn Super Sale event offering up to 50% off. The ad should have an eye-catching design with autumn leaves, and maybe include a shopper to suggest the idea of buying and carrying bags. Include placeholder text for additional details.", label="Intention")

    with gr.Row():
        cfg = gr.Slider(minimum=2, maximum=15, value=5, step=0.5, label="CFG")
        steps = gr.Slider(minimum=20, maximum=100, value=50, step=10, label="Steps")
        sampler = gr.Dropdown(SAMPLERS, value="DPM++ 2M Karras", label="Sampler")
    with gr.Row():
        base_prompt = gr.Textbox(placeholder="Base Prompt\nExample: A painting of a beautiful autumn landscape with a shopper carrying bags", label="Base Prompt")
        base_ratio = gr.Slider(minimum=0.0, maximum=1, value=0.0, step=0.05, label="Base Ratio")
    generate_button = gr.Button("Generate")
    layout_output=gr.Textbox(label="Layout Output")
    with gr.Row():
        image_output=gr.Image(label="Image Output")
        image_bbox=gr.Image(label="Image Bbox")
    generate_button.click(generate, inputs=[checkpoint, seed, intention, cfg, steps, sampler, base_prompt, base_ratio], outputs=[layout_output, image_output, image_bbox])
demo.launch()