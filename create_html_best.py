import json
from collections import Counter
import random
import webcolors

from inference_data import load_inference_data


index_file = "/mnt/openseg_blob/liuzeyu/datasets2/canva_poster_filtered.json"
index_file = "/openseg_blob/liuzeyu/datasets2/canva_illustration.json"
# caption_dir = "/mnt/openseg_blob/weicong/big_file/data/canva-data/metadata/"
json_path = '/openseg_blob/weicong/big_file/data/canva-data/canva_benchmark_chosen_meta/'
font_path = '/openseg_blob/weicong/big_file/data/canva-data/font-mapping.json'

group_index=[{"name":"layer<5",'index':[6, 7, 42, 53, 78, 83, 88, 97]},
             {"name":"layer5~10",'index':[2, 5, 9, 10, 14, 16, 17, 18, 20, 21, 30, 32, 37, 40, 47, 49, 52, 55, 56, 57, 58, 59, 62, 63, 65, 70, 71, 74, 77, 81, 84, 89, 90, 92, 95]},
             {"name":"layer11-15",'index':[0, 3, 11, 12, 15, 19, 22, 24, 25, 26, 28, 36, 38, 39, 41, 43, 45, 46, 60, 64, 66, 67, 69, 72, 75, 80, 91, 94, 99]},
             {"name":"layer>15",'index':[1, 4, 8, 13, 29, 34, 35, 50, 51, 61, 68, 76, 79, 96]}]
def get_caption(meta):
    prompt = meta['caption'] # ann['llava_llama2_caption']
    if "category" in meta:
        prompt = meta["category"] + ". " + prompt
    if "texts" in meta:
        texts = [f'"{text}"' for text in meta["texts"]]
        prompt += " Text: " + ', '.join(texts)
    if "tags" in meta:
        prompt += " Tags: " + ', '.join(meta["tags"])
    # if "texts" in meta:
    #     texts = [f'"{text}"' for text in meta["texts"]]
    #     prompt = ', '.join(texts)
    # else:
    #     prompt = ""
    return prompt

def get_caption_bg(meta, json_path):
    caption_path = os.path.join(json_path, f"{meta['_id']}.json")
    with open(caption_path, 'r') as f:
        caption_ann = json.load(f)
    prompt = caption_ann['caption']
    if "category" in meta:
        prompt = meta["category"] + ". " + prompt
    if "tags" in meta:
        prompt += " Tags: " + ', '.join(meta["tags"])
    return prompt

def get_caption_text(meta, font_dict, use_style=False):
    prompt = ""
    if use_style:
        prompt = "</s>"
        for text, style in zip(meta["texts"], meta["styles"]):
            text_prompt = f'Text: {text}.'
            if 'color' in style:
                hex_color = style["color"]  
                rgb_color = webcolors.hex_to_rgb(hex_color)  
                # get color name  
                color_name = convert_rgb_to_names(rgb_color)  
                text_prompt += f' Color: {color_name}.'
            if 'font-family' in style and style["font-family"] in font_dict:
                font_name = font_dict[style["font-family"]]
                text_prompt += f' Font family: {font_name}.'
            if 'font-weight' in style:
                text_prompt += f' Font weight: {style["font-weight"]}.'
            if 'font-size' in style:
                text_prompt += f' Font size: {style["font-size"]}.'
            if 'text-align' in style:
                text_prompt += f' Text align: {style["text-align"]}.'
            text_prompt = '"' + text_prompt + '"</s>'
            prompt = prompt + text_prompt
    else:
        for text in meta["texts"]:
            text_prompt = f'Text: {text}.'
            text_prompt = '"' + text_prompt + '". '
            prompt = prompt + text_prompt
    return prompt

def closest_color(requested_color):  
    min_colors = {}  
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():  
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)  
        rd = (r_c - requested_color[0]) ** 2  
        gd = (g_c - requested_color[1]) ** 2  
        bd = (b_c - requested_color[2]) ** 2  
        min_colors[(rd + gd + bd)] = name  
    return min_colors[min(min_colors.keys())]

def convert_rgb_to_names(rgb_tuple):  
    try:  
        color_name = webcolors.rgb_to_name(rgb_tuple)  
    except ValueError:  
        color_name = closest_color(rgb_tuple)  
    return color_name


# ann_path = '/openseg_blob/liuzeyu/datasets2/typo_clip_data/typoclip_onebox_200-400_simplefont_100k_val.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/typo_clip_data/typoclip_onebox_200-400_512font_100k_val.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/test_font_byt5_100_color_and_font.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/typo_clip_data/typoclip_onebox_200-400_simplefont_100k_multibox_val.json'
# ann_path = '/openseg_blob/liuzeyu/datasets2/canva_ann_json/canva_font_byt5_400_val.json'
data=load_inference_data('inference/images_100_autocaption_34b-newer.json')
# data = data[:100]

# ann_path = '/openseg_blob/liuzeyu/datasets2/canva_ann_json/font_30k_subset.json'
# with open(ann_path, 'r') as f:
#     ann_list = json.load(f)
# data = [ann_list[i] for i in range(0, 30000, 300)]


print("""
    <html>
       <head>
       <style>
       .row {
           display: flex;  
           flex-direction: row;  
           justify-content: start;  
           align-items: center;  
           margin-bottom: 16px; 
       }
       .column {
           flex: 15%;
           padding: 0 16px;
       }
       </style>
       </head>
       <body>
""")

# data = [x for x in data if "Illustration" in x['tags'] or 'illustration' in x['tags']]

# print(f"""
#         <div class="row">
#             <p> Col1: gt </p>
#         </div>
# """)
# print(f"""
#         <div class="row">
#             <p> Col1: gt </p>
#         </div>
# """)
# print(f"""
#         <div class="row">
#             <p> Col1: before tune multibox text </p>
#         </div>
# """)
# print(f"""
#         <div class="row">
#             <p> Col2: after tune multibox text </p>
#         </div>
# """)


print(f"<p>{len(data)}</p>")
# folder_list = [
#     'yuyang/code/RPG/multi_layers',
#     '
# ]
cfg_group=[3,5,7,9,12,15,18]
json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'
with open(json_file, 'r') as f:
    meta = json.load(f)

for group in group_index:
    print(f"<p style='font-size:40px; '>{group['name']}</p>")
    for index in group["index"]:
        item=data[index]

        file_prefix = f"https://openseg.blob.core.windows.net/openseg-aml/"
        file_sufix = "?sv=2021-10-04&st=2023-11-28T03%3A59%3A26Z&se=2024-11-29T03%3A59%3A00Z&sr=c&sp=rl&sig=kgIQ6YOdgXRxY9kO4grRT4Rx2YRZFHv3V3XVauteVkM%3D"

        info=meta[index]
        
        print(f"""
                <div class="row">
                    <p style='font-size:30px; '> Index: {index} <br>
                    Base Prompt: {item['Base Prompt']} </p>
                </div>
        """)
        
        if index % 1 == 0:
            print('<div class="row">')
        
        folder='v-sirui/temporary/2024-02-21/Process/images'
        img_name = f"/{index}_whole.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>GT </p>
                </div>""")
        folder='yuyang/code/RPG/original_bbox'
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>GT bbox</p>
                </div>""")
        print('</div>')
        print('<div class="row">')
        # print('</div>')
        # print('<div class="row">')
        # folder='yuyang/code/RPG/multi_layers'
        # img_name = f"/{index}.png"
        # print(f"""
        #         <div class="column">
        #             <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             <p>RPG no base no whole </p>
        #         </div>""")
        # folder='yuyang/code/RPG/multi_layers_base_0.0'
        # img_name = f"/{index}.png"
        # print(f"""
        #         <div class="column">
        #             <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             <p>RPG base ratio=0 (totally region attention)</p>
        #         </div>""")
        # folder='yuyang/code/RPG/multi_layers_base_0.0_bbox'
        # img_name = f"/{index}.png"
        # print(f"""
        #         <div class="column">
        #             <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             <p>RPG base 0 bbox</p>
        #         </div>""")
        folder='yuyang/code/RPG/multi_layers_base_0.1'
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>RPG base ratio=0.1, CFG=5, Sampler=DPM++ 2M Karras</p>
                </div>""")
        folder='yuyang/code/RPG/multi_layers_base_0.1_cfg_7_DPM++_2M_SDE_Heun_Karras'
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>RPG base ratio=0.1, CFG=7, Sampler=DPM++_2M_SDE_Heun_Karras</p>
                </div>""")
        folder='yuyang/code/RPG/multi_layers_base_0.1_cfg_7_DPM_adaptive'
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>RPG base ratio=0.1, CFG=7, Sampler=DPM_adaptive</p>
                </div>""")

    
        
        # folder='yuyang/code/RPG/multi_layers_resize_base_0.0'
        # img_name = f"/{index}.png"
        # print(f"""
        #         <div class="column">
        #             <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             <p>RPG resize base 0</p>
        #         </div>""")
        # folder='yuyang/code/RPG/multi_layers_resize_base_0.2'
        # img_name = f"/{index}.png"
        # print(f"""
        #         <div class="column">
        #             <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             <p>RPG resize base 0.2</p>
        #         </div>""")
        # print('</div>')
        # print('<div class="row">')
        # folder='yuyang/code/RPG/multi_layers_base_1'
        # img_name = f"/{index}.png"
        # print(f"""
        #         <div class="column">
        #             <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
        #             <p>RPG base ratio=1 (=totally using SDXL)</p>
        #         </div>""")
        
        
        

        if index % 1 == 0:
            print('</div>')

        print('<div class="row">')
        base_caption=info['base_image']['caption']
        img_path=info['base_image']['path'].replace('/openseg_blob/','')
        print(f"""
                <div class="column">
                    
                    <p>layer 0:{base_caption}</p>
                </div>""")
        print('</div>')

        count=0
        print('<div class="row">')
        # <img src="{file_prefix + img_path + file_sufix}" alt="image" style="max-width: 100%;">
        info['layers'].reverse()
        for n,layer in enumerate(info['layers']):
            layer_caption=layer['caption']
            img_path=layer['path'].replace('/openseg_blob/v-sirui/temporary/2024-02-21/Process/images/','yuyang/code/RPG/multi_layers_base_0.0_debug_layers/')
            img_path=img_path.replace('_box','')
            print(f"""
                <div class="column">
                    
                    <p>layer {n+1}:{layer_caption}</p>
                </div>""")
            count+=1
            if count==4:
                print('</div>')
                print('<div class="row">')
                count=0
        

        print('</div>')


        
        # print(f"""
        #         <div class="row">
        #             <p>Index: {index} <br>
        #             Category: {item["category"]} <br>
        #             {item["BGCaption_llava"]} <br>
        #             Keywords: {item["keywords"]}</p>
        #         </div>
        # """)
