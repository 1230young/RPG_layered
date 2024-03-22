      
import gradio as gr
import openai
import time
from openai import AzureOpenAI
import json
import os
from tqdm import tqdm
import base64
import cv2
import requests
import numpy as np
from PIL import Image, ImageDraw
from RPG_pipeline import *
checkpoint_model=""
Count=0
SAMPLERS=["DPM++ 2M Karras", "DPM++ SDE Karras", "DPM++ 2M SDE Exponential", "DPM++ 2M SDE Karras", "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "DPM++ 2M SDE Heun", "DPM++ 2M SDE Heun Karras", "DPM++ 2M SDE Heun Exponential", "DPM++ 3M SDE", "DPM++ 3M SDE Karras", "DPM++ 3M SDE Exponential", "DPM fast", "DPM adaptive", "LMS Karras", "DPM2 Karras", "DPM2 a Karras", "DPM++ 2S a Karras", "Restart", "DDIM", "PLMS", "UniPC"]
from inference_data import COLOR_LIST
layout_list=''
def generate_layout(intention):
    global layout_list
    try:
        gpt_output=intension2(intention)
        layout_list=gpt_output
        return gpt_output
    except Exception as e:
        return []
def generate_RPG(checkpoint, seed, intention, layout, cfg, steps, sampler, base_prompt, base_ratio):
    global checkpoint_model
    global layout_list
    global Count
    # layout=layout_list
    
    import random
    try:
        seed=int(seed)
    except:
        seed=-1
    try:
        if len(layout)==0:
            raise Exception("Please generate layout first.")
        layout=eval(layout)
        config_file="test/try_pipeline.py"
        opt=read_config(config_file)
        if seed==-1:
            seed=random.randint(0,10000)
        user_prompt=opt.user_prompt
        
        activate=True
        use_base=True
        batch_size=1
        height=1024
        width=1024

        if len(base_prompt)>0:
            processed_data=load_gpt_output(layout,intention,base_prompt)
        
        else:
            processed_data=load_gpt_output(layout,intention)
        user_prompt=processed_data['Layer Prompt']
        bboxes=processed_data['bboxes']
        base_prompt=processed_data['Base Prompt']
        if checkpoint=="sd_xl_base_1.0":
            opt.vae_dir="madebyollin.safetensors"
        else:
            opt.vae_dir=None
        ckpt_dir=[opt.load_typo_sdxl_pretrain_ckpt,opt.ckpt_dir] if opt.load_typo_sdxl_pretrain_ckpt is not None else opt.ckpt_dir
        if checkpoint!="albedobaseXL_v20" and checkpoint!="playground-v2.fp16":
            checkpoint="sd_xl_base_1.0"
        model_name=checkpoint+".safetensors"
        if checkpoint!=checkpoint_model:
            reload=checkpoint_model!=""
            checkpoint_model=checkpoint
            model_name=checkpoint+".safetensors"
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
        Count+=1
        
        del draw
        return image, img
    except:
        from PIL import Image
        image=Image.new('RGB', (1024, 1024), (255, 255, 255))
        img=image.copy()
        return image, img


def get_payload(prompt):
    payload = {
                    # "prompt": "a cute bear",
                    "prompt": prompt,
                    "negative_prompt": "ugly, bad",
                    "prompt_styles": [],
                    "steps": 20,
                    "sampler_name": "DPM++ 2M Karras",
                    "n_iter": 1,
                    "batch_size": 1,
                    "cfg_scale": 7,
                    "height": 512,
                    "width": 512,
                    "enable_hr": False,
                    "denoising_strength": 0.7,
                    "hr_scale": 2,
                    "hr_upscaler": "Latent",
                    "hr_second_pass_steps": 0,
                    "hr_resize_x": 0,
                    "hr_resize_y": 0,
                    "hr_checkpoint_name": "Use same checkpoint",
                    "hr_sampler_name": "Use same sampler",
                    "hr_prompt": "",
                    "hr_negative_prompt": "",
                    "override_settings_texts": [],
                    "force_enable_hr": False,
                    "seed": 12345,
                    "alwayson_scripts": {
                        "LayerDiffuse": {
                            "args":[
                                True,  # "enabled"
                                '(SD1.5) Only Generate Transparent Image (Attention Injection)',  # "method"
                                1,  # "weight"
                                1,  # "ending_step"
                                None,  # "fg_image"
                                None,  # "bg_image"
                                None,  # "blend_image"
                                'Crop and Resize',  # "resize_mode"
                                '',  # "fg_additional_prompt"
                                '',  # "bg_additional_prompt"
                                '',  # "blend_additional_prompt"
                            ]
                        },
                    },
                }
    return payload


def get_payload_bg(prompt):
    payload = {
                    "prompt": prompt,
                    "negative_prompt": "ugly, bad",
                    "prompt_styles": [],
                    "steps": 20,
                    "sampler_name": "DPM++ 2M Karras",
                    "n_iter": 1,
                    "batch_size": 1,
                    "cfg_scale": 7,
                    "height": 512,
                    "width": 512,
                    "enable_hr": False,
                    "denoising_strength": 0.7,
                    "hr_scale": 2,
                    "hr_upscaler": "Latent",
                    "hr_second_pass_steps": 0,
                    "hr_resize_x": 0,
                    "hr_resize_y": 0,
                    "hr_checkpoint_name": "Use same checkpoint",
                    "hr_sampler_name": "Use same sampler",
                    "hr_prompt": "",
                    "hr_negative_prompt": "",
                    "override_settings_texts": [],
                    "force_enable_hr": False,
                    "seed": 12345,
                }
    return payload


def get_layer(payload):
    resp = requests.post("http://0.0.0.0:7861/sdapi/v1/txt2img", json=payload)
    imgs = resp.json()["images"]

    img_base64 = base64.b64decode(imgs[0])
    img_bytes = bytearray(img_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_RGB = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)

    img_base64 = base64.b64decode(imgs[1])
    img_bytes = bytearray(img_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_RGBA = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    img_RGBA[:, :, :3] = img_RGB
    return img_RGBA


def get_layer_bg(payload):
    resp = requests.post("http://0.0.0.0:7861/sdapi/v1/txt2img", json=payload)
    imgs = resp.json()["images"]
    img_base64 = base64.b64decode(imgs[0])
    img_bytes = bytearray(img_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_RGB = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_RGB = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)

    img_RGBA = np.ones((512, 512, 4), dtype=np.uint8) * 255
    img_RGBA[:, :, :3] = img_RGB

    return img_RGBA


def crop_with_alpha(img_RGBA):
    alpha_channel = img_RGBA[:, :, 3]
    # Find the coordinates of the non-transparent pixels
    non_transparent_pixels = np.where(alpha_channel > 0)
    # Get the minimum and maximum coordinates
    min_x = np.min(non_transparent_pixels[1])
    min_y = np.min(non_transparent_pixels[0])
    max_x = np.max(non_transparent_pixels[1])
    max_y = np.max(non_transparent_pixels[0])
    # Crop the image to the non-transparent area
    cropped_img = img_RGBA[min_y:max_y, min_x:max_x]
    return cropped_img


def layer_to_fullsize(layer_img, location, full_size):
    layer_img_np = np.array(layer_img)
    full_img_np = np.zeros((full_size[0], full_size[1]) + (4,), dtype=np.uint8)
    full_img_np = np.transpose(full_img_np, (1, 0, 2))
    
    full_img_np[location[1]:location[3], location[0]:location[2], :] = layer_img_np
    # full_img_np = np.transpose(full_img_np, (1, 0, 2))
    full_img = Image.fromarray(full_img_np, 'RGBA')
    return full_img


def list_to_fmtstr(result_list):
    fmtstr = ""
    for element_dict in result_list:
        if "background" in element_dict.keys():
            caption = element_dict['background']
            fmtstr += f"Background: {caption}\n"
        else:
            layer_id = element_dict['layer']
            category = element_dict['category']
            caption = element_dict['caption']
            fmtstr += f"Layer {layer_id}: {caption}\n"
    return fmtstr


def postprocess_layer_list(layer_list):
    curr_len = len(layer_list)
    if curr_len < 6:
        for i in range(6 - curr_len):
            layer_list.append(Image.new('RGBA', (512, 512)))
    return layer_list


def generate_layer(result_list):
    global layout_list
    # result_list=layout_list

    try:  
        if len(result_list)==0:
            raise Exception("Please generate layout first.")
        result_list=eval(result_list)
    except:
        return Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255))
    
    layer_list = []
    object_box_list = []
    text_box_list = []
    for element_dict in result_list:
        if "background" in element_dict.keys():
            caption = element_dict['background']
            payload_bg = get_payload_bg("This is a picture of background layer. " +caption)
            backgournd_img = get_layer_bg(payload_bg)
        else:
            layer_id = element_dict['layer']
            category = element_dict['category']
            caption = element_dict['caption']
            top_left = element_dict['top_left']
            bottom_right = element_dict['bottom_right']
            if category == 'element':
                print(f"Layer {layer_id}: {caption}; top_left: {top_left}; bottom_right: {bottom_right}.")
                payload = get_payload(caption)
                layer_img = get_layer(payload)  # numpy tensor
                layer_list.append({
                    "layer_id": layer_id,
                    "layer_img": layer_img,
                    "caption": caption,
                    "top_left": top_left,
                    "bottom_right": bottom_right,
                    "box_size": (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]),
                })
                print('boxsize ', (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

                object_box_list.append((top_left, bottom_right))
            if category == 'text':
                text_box_list.append((top_left, bottom_right))

    # Create a blank canvas
    canvas_width = 1456
    canvas_height = 1457
    canvas = Image.new('RGBA', (canvas_width, canvas_height))
    draw = ImageDraw.Draw(canvas)
    
    # Iterate over each layer and paste it onto the canvas
    for layer in layer_list:
        layer_img = layer['layer_img']
        Image.fromarray(layer_img, 'RGBA').save(f"/pyy/yuyang_blob/pyy/code/sd_ui/webui/test_result/layer{layer['layer_id']}.png")
        cropped_layer_img = Image.fromarray(layer_img, 'RGBA')
        resized_layer_img = cropped_layer_img.resize((max(layer["box_size"]), max(layer["box_size"])))

        centerxy = (layer["top_left"][0] + layer["bottom_right"][0]) / 2, (layer["top_left"][1] + layer["bottom_right"][1]) / 2
        new_top_left = (int(centerxy[0] - resized_layer_img.width / 2), int(centerxy[1] - resized_layer_img.height / 2))

        canvas.paste(resized_layer_img, new_top_left)


    backgournd_img = Image.fromarray(backgournd_img, 'RGBA').resize((canvas_width, canvas_height))


    # create a bg canva, paste the canvas on it
    layer_img_list = []
    canvas_white = backgournd_img
    for layer in layer_list:
        layer_img = layer['layer_img']
        cropped_layer_img = Image.fromarray(layer_img, 'RGBA')
        layer_img_list.append(cropped_layer_img)
        resize_size = min(min(layer["box_size"]), min(canvas_height, canvas_width))
        resized_layer_img = cropped_layer_img.resize(
            (
                resize_size,
                resize_size,
            )
        )

        centerxy = (layer["top_left"][0] + layer["bottom_right"][0]) / 2, (layer["top_left"][1] + layer["bottom_right"][1]) / 2
        new_top_left = (int(centerxy[0] - resized_layer_img.width / 2), int(centerxy[1] - resized_layer_img.height / 2))
        new_bottem_right = (int(centerxy[0] + resized_layer_img.width / 2), int(centerxy[1] + resized_layer_img.height / 2))
        new_location = new_top_left + new_bottem_right
        resized_layer_img_fullsize = layer_to_fullsize(resized_layer_img, new_location, (canvas_width, canvas_height))
        canvas_white = Image.alpha_composite(canvas_white, resized_layer_img_fullsize)
        canvas_white = canvas_white.convert('RGB')
        canvas_white = canvas_white.convert('RGBA')

    # plot text boxes
    draw = ImageDraw.Draw(canvas_white)  
    for text_box in text_box_list:
        draw.rectangle([tuple(text_box[0]), tuple(text_box[1])], fill=None, outline="black", width=3)
    for object_box in object_box_list:
        draw.rectangle([tuple(object_box[0]), tuple(object_box[1])], fill=None, outline="red", width=3)

    # Save the final image
    # canvas.save('/home/pyf/code/p2canva2/invitation.png')
    # canvas_white.save('/home/pyf/code/p2canva2/invitation_white.png')
    backgournd_img = backgournd_img
    foreground_img = canvas
    whole_img = canvas_white
    layout_and_caption = list_to_fmtstr(result_list)
    layer_img_list = postprocess_layer_list(layer_img_list)
    return backgournd_img, whole_img, layer_img_list[0], layer_img_list[1], layer_img_list[2], layer_img_list[3], layer_img_list[4], layer_img_list[5]
    # except:
    #     from PIL import Image
    #     return Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255)), Image.new('RGB', (1024, 1024), (255, 255, 255))

#ui部分
with gr.Blocks() as demo:
    #RPG
    gr.Markdown("Intention to layout to image generation.")
    intention=gr.Textbox(placeholder="Intention\nExample: Create an advertisement for an Autumn Super Sale event offering up to 50% off. The ad should have an eye-catching design with autumn leaves, and maybe include a shopper to suggest the idea of buying and carrying bags. Include placeholder text for additional details.", label="Intention")
    generate_layout_button = gr.Button("Generate Layout")
    layout=gr.Textbox(label="Layout Output", interactive=False)
    generate_layout_button.click(
        generate_layout,
        inputs=[intention],
        outputs=[layout]
    )
    gr.Markdown("RPG.")
    with gr.Row():
        checkpoint=gr.Dropdown(["sd_xl_base_1.0","albedobaseXL_v20","playground-v2.fp16"], label="Checkpoint", interactive=False)
        seed=gr.Textbox("-1", label="Seed")

    with gr.Row():
        cfg = gr.Slider(minimum=2, maximum=15, value=5, step=0.5, label="CFG")
        steps = gr.Slider(minimum=20, maximum=100, value=50, step=10, label="Steps")
        sampler = gr.Dropdown(SAMPLERS, value="DPM++ 2M Karras", label="Sampler")
    with gr.Row():
        base_prompt = gr.Textbox(placeholder="Base Prompt\nExample: A painting of a beautiful autumn landscape with a shopper carrying bags", label="Base Prompt")
        base_ratio = gr.Slider(minimum=0.0, maximum=1, value=0.0, step=0.05, label="Base Ratio")
    generate_button = gr.Button("Generate RPG")
    with gr.Row():
        image_output=gr.Image(label="Image Output")
        image_bbox=gr.Image(label="Image Bbox")
    



    #Layer Diffuse
    gr.Markdown("Layer Diffusion")
   

    generate_layer_button = gr.Button("Generate Layer Diffusion", interactive=True)
    with gr.Row():
        backgournd_img = gr.Image(label="BG Image")
        # foreground_img = gr.Image(label="FG Image")
        whole_img = gr.Image(label="Whole Image")
    with gr.Row():
        layer1 = gr.Image(label="layer1")
        layer2 = gr.Image(label="layer2")
        layer3 = gr.Image(label="layer3")
        layer4 = gr.Image(label="layer4")
        layer5 = gr.Image(label="layer5")
        layer6 = gr.Image(label="layer6")

    generate_layer_button.click(
        generate_layer,
        inputs=[layout],
        outputs=[backgournd_img, whole_img, layer1, layer2, layer3, layer4, layer5, layer6]
    )
    generate_button.click(generate_RPG, inputs=[checkpoint, seed, intention, layout, cfg, steps, sampler, base_prompt, base_ratio], outputs=[image_output, image_bbox])
demo.queue(concurrency_count=3)
demo.launch(share=True)

# Create an invitation for a Valentine's Day party to be held on February 14, 2022, at 123 Anywhere St., Any City from 7 PM until the end of the event.
