import sys
sys.path.append('/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster')
import os
import json
import shutil
from TypoClipSDXL.typoclip_sdxl.datasets.utils import get_new_caption_with_special_token
# all_dir="/pyy/openseg_blob/weicong/big_file/data/canva-data/canva-render-12.29"
# example_dir="/pyy/openseg_blob/weicong/llm/DesignDiff/eval/1228/train_examples"
# target_dir="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/data_collect/fetch_2"
# already_get=os.listdir("/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/data_collect/fetch")
# for file in os.listdir(example_dir):
#     if file.endswith(".json"):
#         with open(os.path.join(example_dir, file), 'r') as f:
#             data = json.load(f)
#             png_name=data['id']+"-(0).png"
#             if png_name in already_get:
#                 continue
#             try:
#                 shutil.copyfile(os.path.join(all_dir,png_name),os.path.join(target_dir,png_name))
#             except:
#                 print(png_name)
#         shutil.copyfile(os.path.join(example_dir, file),os.path.join(target_dir,file))


text_prompt_list=[]
with open("/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/data_collect/fetch_2/match.json", 'r') as f:
    match_dict = json.load(f)
font_path = '/pyy/openseg_blob/weicong/big_file/data/canva-data/font-mapping.json'
with open(font_path, 'r') as f:
    font_dict = json.load(f)
font_ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/font_idx_512.json'
with open(font_ann_path, 'r') as f:
    font_idx_dict = json.load(f)
color_ann_path='/pyy/openseg_blob/liuzeyu/datasets2/canva_ann_json/color_idx.json'
with open(color_ann_path, 'r') as f:
    color_idx_dict = json.load(f)

for meta in os.listdir("/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/data_collect/fetch_2"):
    if not meta.endswith("meta.json"):
        continue
    with open(os.path.join("/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/data_collect/fetch_2", meta), 'r') as f:
        data = json.load(f)        
        if data['id'] in match_dict:
            text_prompt_info={}
            text_prompt_info['index']=match_dict[data['id']]
            text_prompt_info['layers']=[]
            info=json.loads(data['text_layer_info'])
            for layer in info:
                texts=[layer['text']]
                styles=[layer['style']]
                prompt=get_new_caption_with_special_token(texts, styles, font_dict, font_idx_dict=font_idx_dict, color_idx_dict=color_idx_dict)
                text_prompt_info['layers'].append({"layer_num":layer["index"],"caption":prompt})
            text_prompt_list.append(text_prompt_info)
with open("/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json", 'r') as f:
    text_prompt_list += json.load(f)
# with open("/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json", 'w') as f:
#     json.dump(text_prompt_list, f, indent=4)
