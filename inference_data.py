import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as plt


def load_inference_data(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    processed_data = []
    for item in data:
        layer_num=len(item['layers'])+1
        layer_prompt=''
        item['layers'].reverse()
        for n,layer in enumerate(item['layers']):
            if n!=layer_num-1:
                layer_prompt+=layer['caption']+' BREAK\n'
            else:
                layer_prompt+=layer['caption']+' BREAK\n'#debug
        layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
        bboxes=[[layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]] for layer in item['layers']]
        bboxes.insert(0,[0,0,1457,1457])
        index=item['index']
        base_prompt=item['whole_image']['caption']
        processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
        

    return processed_data

def load_test_data(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/test.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    processed_data = []
    for item in data:
        layer_num=len(item['layers'])+1
        layer_prompt=''
        item['layers'].reverse()
        for n,layer in enumerate(item['layers']):
            if n!=layer_num-1:
                layer_prompt+=layer['caption']+' BREAK\n'
            else:
                layer_prompt+=layer['caption']+' BREAK\n'#debug
        layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
        bboxes=[]
        for layer in item['layers']:
            new_b=[layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]]
            if "target_top_left" in layer.keys() and "target_bottom_right" in layer.keys():
                new_b+=[layer["target_top_left"][1],layer["target_top_left"][0],layer["target_bottom_right"][1],layer["target_bottom_right"][0]]
            else:
                new_b+=[0,0,1024,1024]
            bboxes.append(new_b)
        
        bboxes.insert(0,[0,0,1457,1457,0,0,1024,1024])
        index=item['index']
        base_prompt=item['whole_image']['caption']
        processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
        

    return processed_data
import os
def draw_bbox(index,bboxes,soruce_dir='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/generated_imgs/CoT-GPT-4-0229-nowidth',target_dir='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/generated_imgs/CoT-GPT-4-0229-nowidth_bbox'):
    color_list=['red','green','blue','purple','orange','pink','brown','black','gray','cyan','magenta','olive','lime','teal','navy','maroon','aqua','fuchsia','silver','gold','indigo','violet','tan','khaki','coral','salmon','tomato','orangered','darkorange','darkred','darkgoldenrod','darkkhaki','darkolivegreen','darkseagreen','darkgreen','darkcyan','darkturquoise','darkslategray','darkblue','darkviolet','darkmagenta','darkorchid','darkpink','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgray','lightgreen','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']
    # soruce_dir='/pyy/openseg_blob/v-sirui/temporary/2024-02-21/Process/images'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    font=ImageFont.truetype('font/arial.ttf', size=30)
    img=Image.open(f'{soruce_dir}/{index}.png')
    # img=Image.new('RGB', (1024, 1024), (255, 255, 255))
    draw=ImageDraw.Draw(img)
    for n,bbox in enumerate(bboxes):
        bbox=[int(x*1024/1457) for x in bbox]
        label=f"Layer {n}"
        target_bbox=[bbox[1],bbox[0],bbox[3],bbox[2]]
        # 获取label长宽
        label_size = draw.textsize(label, font)

        # 设置label起点
        text_origin = np.array([target_bbox[0], target_bbox[1]])

        # 绘制矩形框，加入label文本
        draw.rectangle([target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3]],outline=color_list[n],width=4)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color_list[n])
        draw.text(text_origin, str(label), fill=(255, 255, 255), font=font)
    del draw
    img.save(f'{target_dir}/{index}.png')


def load_layout(dir="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/CoT-GPT-4-0229-nowidth/CoT-GPT-4-0229-nowidth",meta='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'):
    processed_data=[]
    with open(meta, 'r') as f:
        meta_data = json.load(f)
    for file in os.listdir(dir):
        if file.endswith('.json'):
            with open(f'{dir}/{file}', 'r') as f:
                data = json.load(f)
            sub_data={"dir":dir.split('/')[-1]+'/'+file}
            content=[]
            for item in data:
                if len(item['gpt_output'])==0:
                    continue
                layer_num=len(item['input'])+1
                index=item['index']
                layer_prompt=''
                for n,layer in enumerate(item['input']):
                    layer_prompt+=layer['caption']+' BREAK\n'   
                layer_prompt=meta_data[index]['base_image']['caption']+' BREAK\n'+layer_prompt 
                bboxes=[[layer["top"],layer["left"],layer["bottom"],layer["right"]] for layer in item['gpt_output']]
                bboxes.insert(0,[0,0,1457,1457])
                base_prompt=meta_data[index]['whole_image']['caption']
                content.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
            sub_data['data']=content
            processed_data.append(sub_data)
    return processed_data

def load_layout2(dir="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/GPT-4-w-caption/GPT-4-w-caption",meta='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'):
    processed_data=[]
    with open(meta, 'r') as f:
        meta_data = json.load(f)
    for file in os.listdir(dir):
        if file.endswith('.json'):
            with open(f'{dir}/{file}', 'r') as f:
                data = json.load(f)
            sub_data={"dir":dir.split('/')[-1]+'/'+file}
            content=[]
            for item in data:
                if len(item['gpt_output'])==0:
                    continue
                layer_num=len(item['input'])+1
                index=item['index']
                layer_prompt=''
                item['input'].reverse()
                item['gpt_output'].reverse()
                for n,layer in enumerate(item['input']):
                    layer_prompt+=layer['caption']+' BREAK\n'   
                layer_prompt=meta_data[index]['base_image']['caption']+' BREAK\n'+layer_prompt 
                bboxes=[[layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]] for layer in item['gpt_output']]
                bboxes.insert(0,[0,0,1457,1457])
                base_prompt=meta_data[index]['whole_image']['caption']
                content.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
            sub_data['data']=content
            processed_data.append(sub_data)
    return processed_data



    

if __name__ == "__main__":
    # data = load_layout()
    # for group in data:
    #     for item in group['data']:
    #         index=item['index']
    #         bboxes=item['bboxes']
    #         draw_bbox(index,bboxes)
    data=load_inference_data()
    for item in data:
        index=item['index']
        bboxes=item['bboxes']
        source_dir='/pyy/openseg_blob/yuyang/code/RPG/multi_layers_base_0.1_sampler_2'
        target_dir='/pyy/openseg_blob/yuyang/code/RPG/multi_layers_base_0.1_sampler_2_bbox'
        draw_bbox(index,bboxes,source_dir,target_dir)


