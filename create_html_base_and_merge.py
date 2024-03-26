
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


# folder_list = [
#     'yuyang/code/RPG/multi_layers',
#     '
# ]
cfg_group=[3,5,7,9,12,15,18]
json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_sorted.json'
with open(json_file, 'r') as f:
    meta = json.load(f)

for item in meta:
    

    file_prefix = f"https://openseg.blob.core.windows.net/openseg-aml/"
    file_sufix = "?sv=2021-10-04&st=2023-11-28T03%3A59%3A26Z&se=2024-11-29T03%3A59%3A00Z&sr=c&sp=rl&sig=kgIQ6YOdgXRxY9kO4grRT4Rx2YRZFHv3V3XVauteVkM%3D"
    index=item['index']
    if index==10:
        continue
    if index>16 and index!=58 and index!=99:
        continue

    
    if index!=99:
        print(f"""
                <div class="row">
                    <p style='font-size:30px; '> Index: {index} <br>
                    Base Prompt: {item['whole_image']['caption']} </p>
                </div>
        """)
    else:
        
        index=101
        fox=[ { "background": "The layer is a stylized stadium full of cheering fans, using a blend of blue and green colors to represent the atmosphere. The stadium is designed with lines and shapes to suggest depth and perspective, creating a dynamic and engaging scene." }, { "layer": 0, "category": "element", "caption": "A cartoon rabbit dressed in running gear, wearing a red tank top, blue shorts, and a white headband. The rabbit is in a running position, smiling and passing the baton to the next animal.", "top_left": [ 95, 450 ], "bottom_right": [ 395, 1050 ] }, { "layer": 1, "category": "element", "caption": "A cartoon bear dressed in soccer attire, wearing a green jersey, white shorts, and soccer cleats. The bear is standing on one foot while receiving the baton from the rabbit with an enthusiastic expression.", "top_left": [ 390, 455 ], "bottom_right": [ 690, 1100 ] }, { "layer": 2, "category": "element", "caption": "A cartoon fox dressed in biking gear, wearing a yellow cycling jersey, black shorts, a helmet, and cycling gloves. The fox is holding the handlebars of a bike with one hand and receiving the baton from the bear with the other hand, showing a determined and focused expression.", "top_left": [ 675, 400 ], "bottom_right": [ 1175, 1075 ] }, { "layer": 3, "category": "text", "caption": "Text \"Go Farther Together!\n\" in <color-2>, <font-137>. ", "top_left": [ 355, 125 ], "bottom_right": [ 1100, 250 ] }, { "layer": 4, "category": "element", "caption": "A cartoon baton in red and white colors, representing collaboration and support, being passed between the animals in the relay race.", "top_left": [ 355, 725 ], "bottom_right": [ 685, 775 ] } ]
        fox_base="stylized stadium background with cheering fans in blue and green tones. In the foreground, a cartoon relay race unfolds with a rabbit in red and blue running gear smiling and passing a baton to a bear in green soccer attire, who stands on one foot with enthusiasm. Next in line, a determined fox in yellow biking gear reaches out for the baton, poised to continue the race. The rabit is on the left, the bear in the middle, and the fox on the right. Above the scene, the words \"Go Farther Together!\" are boldly displayed in a striking color and font. The baton, red and white, symbolizes the spirit of teamwork as it moves between the animated animals."
        print(f"""
                <div class="row">
                    <p style='font-size:30px; '> Index: {index} <br>
                    Base Prompt: {fox_base} </p>
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
    base_ratios=[0.0,0.2,0.4,0.6,0.8]
    merge_ratios=[0.6,0.8,1.0]
    for b_r in base_ratios:
        for m_r in merge_ratios:
            folder=f'yuyang/code/RPG/canva_1m_checkpoint_16000_base_{str(b_r)}_merge_{str(m_r)}'
            img_name = f"/{index}.png"
            print(f"""
                    <div class="column">
                        <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                        <p>base ratio={b_r} merge ratio={m_r}</p>
                    </div>""")
            folder=f'yuyang/code/RPG/canva_1m_checkpoint_16000_base_{str(b_r)}_merge_{str(m_r)}_bbox'
            img_name = f"/{index}.png"
            print(f"""
                    <div class="column">
                        <img src="{file_prefix + folder + img_name + file_sufix}" alt="image" style="max-width: 100%;">
                        <p>base ratio={b_r} merge ratio={m_r} bbox</p>
                    </div>""")
        print('</div>')
        print('<div class="row">')
    
    

    if index % 1 == 0:
        print('</div>')

    count=0
    print('<div class="row">')
    # <img src="{file_prefix + img_path + file_sufix}" alt="image" style="max-width: 100%;">

    base_caption=item['base_image']['caption']
    img_path=item['base_image']['path'].replace('/openseg_blob/','')
    print(f"""
            <div class="column">
                
                <p>layer 0:{base_caption}</p>
            </div>""")
    print('</div>')

    count=0
    print('<div class="row">')
    # <img src="{file_prefix + img_path + file_sufix}" alt="image" style="max-width: 100%;">
    if index!=101:
        for n,layer in enumerate(item['layers']):
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
    else:
        for n,layer in enumerate(fox):
            if n==0:
                layer_caption=layer['background']
            else:
                layer_caption=layer['caption']
            # img_path=layer['path'].replace('/openseg_blob/v-sirui/temporary/2024-02-21/Process/images/','yuyang/code/RPG/multi_layers_base_0.0_debug_layers/')
            # img_path=img_path.replace('_box','')
            print(f"""
                <div class="column">
                    
                    <p>layer {n}:{layer_caption}</p>
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
