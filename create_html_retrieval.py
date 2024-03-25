
import json
from collections import Counter
import random
import webcolors

from inference_data import load_inference_data, load_retrieval


index_file = "/mnt/openseg_blob/liuzeyu/datasets2/canva_poster_filtered.json"
index_file = "/openseg_blob/liuzeyu/datasets2/canva_illustration.json"
# caption_dir = "/mnt/openseg_blob/weicong/big_file/data/canva-data/metadata/"
json_path = '/openseg_blob/weicong/big_file/data/canva-data/canva_benchmark_chosen_meta/'
font_path = '/openseg_blob/weicong/big_file/data/canva-data/font-mapping.json'

# group_index=[{"name":"layer<5",'index':[6, 7, 42, 53, 78, 83, 88, 97]},
#              {"name":"layer5~10",'index':[2, 5, 9, 10, 14, 16, 17, 18, 20, 21, 30, 32, 37, 40, 47, 49, 52, 55, 56, 57, 58, 59, 62, 63, 65, 70, 71, 74, 77, 81, 84, 89, 90, 92, 95]},
#              {"name":"layer11-15",'index':[0, 3, 11, 12, 15, 19, 22, 24, 25, 26, 28, 36, 38, 39, 41, 43, 45, 46, 60, 64, 66, 67, 69, 72, 75, 80, 91, 94, 99]},
#              {"name":"layer>15",'index':[1, 4, 8, 13, 29, 34, 35, 50, 51, 61, 68, 76, 79, 96]}]

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
retrieval_json=['inference/retrieval/text.json',
                'inference/retrieval/visual.json',
                'inference/retrieval/text_prompt1.json',
                'inference/retrieval/visual_prompt1.json'
                ]
retrieval_name=['text','visual','text_prompt1', 'visual_prompt1']
retrieval_folder=['yuyang/code/RPG/mulit_layer_glyph_new_model_albedobaseXL_base_0.0_retrieval_text',
                'yuyang/code/RPG/mulit_layer_glyph_new_model_albedobaseXL_base_0.0_retrieval_visual',
                'yuyang/code/RPG/mulit_layer_glyph_new_model_albedobaseXL_base_0.0_retrieval_text_prompt1',
                'yuyang/code/RPG/mulit_layer_glyph_new_model_albedobaseXL_base_0.0_retrieval_visual_prompt1']
data=[load_retrieval(i) for i in retrieval_json]

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
print
cfg_group=[3,5,7,9,12,15,18]
json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/filter_save_color_800_tol_5_gpt_output_bg.json'

with open(json_file, 'r') as f:
    meta = json.load(f)

    for i in meta:
        index=i['index']
        item=i

        file_prefix = f"https://openseg.blob.core.windows.net/openseg-aml/"
        file_sufix = "?sv=2021-10-04&st=2023-11-28T03%3A59%3A26Z&se=2024-11-29T03%3A59%3A00Z&sr=c&sp=rl&sig=kgIQ6YOdgXRxY9kO4grRT4Rx2YRZFHv3V3XVauteVkM%3D"

        info=meta[index]
        
        print(f"""
                <div class="row">
                    <p style='font-size:30px; '> Index: {index} <br></p>
                </div>
        """)
        
        if index % 1 == 0:
            print('<div class="row">')
        
        
        
        folder='yuyang/code/RPG/intention_mulit_layer_glyph_mixed_w1-1_model_albedobaseXL_base_0.0_bg'
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>RPG Glyph SDXL old pipeline layout</p>
                </div>""")
        folder='yuyang/code/RPG/intention_mulit_layer_glyph_mixed_w1-1_model_albedobaseXL_base_0.0_bg_bbox'
        img_name = f"/{index}.png"
        print(f"""
                <div class="column">
                    <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                    <p>RPG Glyph SDXL old pipeline layout bbox</p>
                </div>""")
        print('</div>')
        print('<div class="row">')
        for n in range(len(data)):  
            if index>=len(data[n]):
                continue
            for m,item_retrieval in enumerate(data[n][index]):
                folder=retrieval_folder[n]   
                img_name = f"/{index}_{m}.png"
                print(f"""
                        <div class="column">
                            <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                            <p>{retrieval_name[n]}_{m}</p>
                        </div>""")
                folder=retrieval_folder[n]+"_bbox" 
                img_name = f"/{index}_{m}.png"
                print(f"""
                        <div class="column">
                            <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                            <p>{retrieval_name[n]}_{m} bbox</p>
                        </div>""")

            print('</div>')
            print('<div class="row">')




    
        
        

        if index % 1 == 0:
            print('</div>')

        # count=0
        # print('<div class="row">')
        # # <img src="{file_prefix + img_path + file_sufix}" alt="image" style="max-width: 100%;">

        # for n,layer in enumerate(info['gpt_output']):
        #     if n==0:
        #         print(f"""
        #         <div class="column">
                    
        #             <p>layer {n}:{layer['background']}</p>
        #         </div>""")
        #     else:
        #         layer_caption=layer['caption']
                
        #         print(f"""
        #             <div class="column">
                        
        #                 <p>layer {n}:{layer_caption}</p>
        #             </div>""")
        #     count+=1
        #     if count==4:
        #         print('</div>')
        #         print('<div class="row">')
        #         count=0
        

        # print('</div>')


        
        # print(f"""
        #         <div class="row">
        #             <p>Index: {index} <br>
        #             Category: {item["category"]} <br>
        #             {item["BGCaption_llava"]} <br>
        #             Keywords: {item["keywords"]}</p>
        #         </div>
        # """)
