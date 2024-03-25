import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib as plt
COLOR_LIST=['red','green','blue','purple','orange','pink','brown','black','gray','cyan','magenta','olive','lime','teal','navy','maroon','aqua','fuchsia','silver','gold','indigo','violet','tan','khaki','coral','salmon','tomato','orangered','darkorange','darkred','darkgoldenrod','darkkhaki','darkolivegreen','darkseagreen','darkgreen','darkcyan','darkturquoise','darkslategray','darkblue','darkviolet','darkmagenta','darkorchid','darkpink','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgray','lightgreen','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']

def load_inference_data(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    processed_data = []
    for item in data:
        layer_num=len(item['layers'])+1
        layer_prompt=''
        item['layers'].reverse()
        bboxes=[]
        layer_prompt_big=''
        bboxes_big=[]
        for n,layer in enumerate(item['layers']):
            if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                layer_prompt_big+=layer['caption']+' BREAK\n'
                bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
            else:
                layer_prompt+=layer['caption']+' BREAK\n'
                bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
        layer_prompt=layer_prompt_big+layer_prompt
        bboxes=bboxes_big+bboxes
        layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
        bboxes.insert(0,[0,0,1457,1457])
        index=item['index']
        base_prompt=item['whole_image']['caption']
        processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
    return processed_data

def load_inference_data_debug(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json',glyph_file="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    with open(glyph_file, 'r') as f:
        glyph_data = json.load(f)
    processed_data = []
    debug_list=[58,24,66,91,29,50]
    # debug_list=[58]
    for i in debug_list:
        item=data[i]
        layer_num=len(item['layers'])+1
        item['layers'].reverse()
        glyph_item=None
        for n,g in enumerate(glyph_data):
            if g['index']==item['index']:
                glyph_item=g
                if not all and n >=30:
                    glyph_item=None
        if glyph_item is None:
            continue
        for length in range(len(item['layers'])+1):
            layer_prompt=''
            bboxes=[]
            layer_prompt_big=''
            bboxes_big=[]
            for n,layer in enumerate(item['layers'][:length]):
                # if n in [0,3,5,8]:
                #     continue
                if layer["layer_num"] in [x["layer_num"] for x in glyph_item['layers']]:
                    caption=[x["caption"] for x in glyph_item['layers'] if x["layer_num"]==layer["layer_num"]][0]
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=caption+' GLYPH BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=caption+' GLYPH BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=layer['caption']+' BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=layer['caption']+' BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                
            layer_prompt=layer_prompt_big+layer_prompt
            bboxes=bboxes_big+bboxes
            layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
            bboxes.insert(0,[0,0,1457,1457])
            index=item['index']
            if length!=len(item['layers']):
                index=str(index)+'_'+str(length)
            base_prompt=item['whole_image']['caption']
            processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
            

    return processed_data


def load_inference_data_debug_backup(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json',glyph_file="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    with open(glyph_file, 'r') as f:
        glyph_data = json.load(f)
    processed_data = []
    debug_list=[58,24,66,91,29,50]
    # debug_list=[58]
    for i in debug_list:
        item=data[i]
        layer_num=len(item['layers'])+1
        item['layers'].reverse()
        glyph_item=None
        for n,g in enumerate(glyph_data):
            if g['index']==item['index']:
                glyph_item=g
                if not all and n >=30:
                    glyph_item=None
        if glyph_item is None:
            continue
        for length in range(len(item['layers'])+1):
            layer_prompt=''
            
            for n,layer in enumerate(item['layers'][:length]):
                # if n in [0,3,5,8]:
                #     continue
                if layer["layer_num"] in [x["layer_num"] for x in glyph_item['layers']]:
                    caption=[x["caption"] for x in glyph_item['layers'] if x["layer_num"]==layer["layer_num"]][0]
                    layer_prompt+=caption+' GLYPH BREAK\n'
                else:
                    layer_prompt+=layer['caption']+' BREAK\n'

            layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
            bboxes=[]
            for n,layer in enumerate(item['layers'][:length]):
                # if n in [0,3,5,8]:
                #     continue
                # else:
                bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])

            bboxes.insert(0,[0,0,1457,1457])
            index=item['index']
            if length!=len(item['layers']):
                index=str(index)+'_'+str(length)
            base_prompt=item['whole_image']['caption']
            processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
            

    return processed_data

def load_inference_data_with_glyph(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json', glyph_file="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json", all=True):
    with open(json_file, 'r') as f:
        data = json.load(f)
    with open(glyph_file, 'r') as f:
        glyph_data = json.load(f)
    processed_data = []
    for item in data:
        glyph_item=None
        for n,g in enumerate(glyph_data):
            if g['index']==item['index']:
                glyph_item=g
                if not all and n >=30:
                    glyph_item=None
        if glyph_item is None:
            continue
       
        layer_num=len(item['layers'])+1
        layer_prompt=''
        bboxes=[]
        item['layers'].reverse()
        layer_prompt_big=''
        bboxes_big=[]
        for n,layer in enumerate(item['layers']):
            if layer["layer_num"] in [x["layer_num"] for x in glyph_item['layers']]:
                caption=[x["caption"] for x in glyph_item['layers'] if x["layer_num"]==layer["layer_num"]][0]
                if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                    layer_prompt_big+=caption+' GLYPH BREAK\n'
                    bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    layer_prompt+=caption+' GLYPH BREAK\n'
                    bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
            else:
                if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                    layer_prompt_big+=layer['caption']+' BREAK\n'
                    bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    layer_prompt+=layer['caption']+' BREAK\n'
                    bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
        layer_prompt=layer_prompt_big+layer_prompt
        bboxes=bboxes_big+bboxes
        layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
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

def load_intention_output_data(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/filter_save_color_800_tol_5_gpt_output_bg.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    processed_data = []
    for item in data:
        if len(item['gpt_output'])==0:
            continue
        layer_num=len(item['gpt_output'])
        background_prompt=''
        background_bbox=[]
        layer_prompt=''
        bboxes=[]
        layer_prompt_big=''
        bboxes_big=[]
        for n,layer in enumerate(item['gpt_output']):
            if "background" in layer.keys():
                background_prompt+=layer['background']+' BREAK\n'
                background_bbox.append([0,0,1457,1457])
                continue
            if layer['category']=='text':
                if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                    layer_prompt_big+=layer['caption']+' GLYPH BREAK\n'
                    bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    layer_prompt+=layer['caption']+' GLYPH BREAK\n'
                    bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
            else:
                if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                    layer_prompt_big+=layer['caption']+' BREAK\n'
                    bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    layer_prompt+=layer['caption']+' BREAK\n'
                    bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
        layer_prompt=background_prompt+layer_prompt_big+layer_prompt
        bboxes=background_bbox+bboxes_big+bboxes
        
        index=item['index']
        base_prompt=item['intension']
        processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt, "intention": item['intension']})
        

    return processed_data

def load_gpt_output(gpt_output, intention, base_prompt=""):
    print(gpt_output)
    item=gpt_output
    index=101
    layer_num=len(item)
    layer_prompt=''
    bboxes=[]
    background_prompt=''
    background_bbox=[]
    layer_prompt_big=''
    bboxes_big=[]
    for n,layer in enumerate(item):
        if n==0:
            background_prompt+=layer['background']+' BREAK\n'
            background_bbox.append([0,0,1457,1457])
            continue
        if layer['category']=='text':
            if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                layer_prompt_big+=layer['caption']+' GLYPH BREAK\n'
                bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
            else:
                layer_prompt+=layer['caption']+' GLYPH BREAK\n'
                bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
        else:
            if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                layer_prompt_big+=layer['caption']+' BREAK\n'
                bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
            else:
                layer_prompt+=layer['caption']+' BREAK\n'
                bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
    layer_prompt=background_prompt+layer_prompt_big+layer_prompt
    bboxes=background_bbox+bboxes_big+bboxes
    base_prompt=base_prompt if len(base_prompt)>0 else intention
    return {'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt}

def load_retrieval(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/retrieval/text.json', start_index=0):
    with open(json_file, 'r') as f:
        data = json.load(f)[start_index:]
    processed_data = []
    for example in data:
        processed_example=[]
        for item in example['gptoutput']:
            if len(item)==0:
                continue
            layer_num=len(item)
            layer_prompt=''
            bboxes=[]
            background_prompt=''
            background_bbox=[]
            layer_prompt_big=''
            bboxes_big=[]
            for n,layer in enumerate(item):
                if layer['category']=='text':
                    caption="Text "+ " ".join(layer["caption"].split()[1:])
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=caption+' GLYPH BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=caption+' GLYPH BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=layer['caption']+' BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=layer['caption']+' BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])

            layer_prompt=background_prompt+layer_prompt_big+layer_prompt
            bboxes=background_bbox+bboxes_big+bboxes
            index=example['index']
            base_prompt=example['intension']
            processed_example.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt, "intention": example['intension']})
        processed_data.append(processed_example)
        # for item in example["supporting_examples"]:
        #     if len(item['output'])==0:
        #         continue
        #     layer_num=len(item['output'])
        #     layer_prompt=''
        #     bboxes=[]
        #     for n,layer in enumerate(item['output']):
        #         if layer['category']=='text':
        #             caption="Text "+ " ".join(layer["caption"].split()[1:])
        #             layer_prompt+=caption+' GLYPH BREAK\n'
        #         else:
        #             layer_prompt+=layer['caption']+' BREAK\n'
        #         bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
        #     base_prompt=item['intension']
        #     index=item['index']
        #     processed_example.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'support_index':index,'Base Prompt':base_prompt, "intention": item['intension']})
        # processed_data.append(processed_example)


    return processed_data

def load_customized_debug(json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json',glyph_file="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    with open(glyph_file, 'r') as f:
        glyph_data = json.load(f)
    processed_data = []
    debug_list=[58,58,58]
    # debug_list=[58]
    # fox=[ { "background": "The layer is a stylized stadium full of cheering fans, using a blend of blue and green colors to represent the atmosphere. The stadium is designed with lines and shapes to suggest depth and perspective, creating a dynamic and engaging scene." }, { "layer": 0, "category": "element", "caption": "A cartoon rabbit dressed in running gear, wearing a red tank top, blue shorts, and a white headband. The rabbit is in a running position, smiling and passing the baton to the next animal.", "top_left": [ 95, 450 ], "bottom_right": [ 395, 1050 ] }, { "layer": 1, "category": "element", "caption": "A cartoon bear dressed in soccer attire, wearing a green jersey, white shorts, and soccer cleats. The bear is standing on one foot while receiving the baton from the rabbit with an enthusiastic expression.", "top_left": [ 390, 455 ], "bottom_right": [ 690, 1100 ] }, { "layer": 2, "category": "element", "caption": "A cartoon fox dressed in biking gear, wearing a yellow cycling jersey, black shorts, a helmet, and cycling gloves. The fox is holding the handlebars of a bike with one hand and receiving the baton from the bear with the other hand, showing a determined and focused expression.", "top_left": [ 675, 400 ], "bottom_right": [ 1175, 1075 ] }, { "layer": 3, "category": "text", "caption": "Text \"Go Farther Together!\n\" in <color-2>, <font-137>. ", "top_left": [ 355, 125 ], "bottom_right": [ 1100, 250 ] }, { "layer": 4, "category": "element", "caption": "A cartoon baton in red and white colors, representing collaboration and support, being passed between the animals in the relay race.", "top_left": [ 355, 725 ], "bottom_right": [ 685, 775 ] } ]
    # fox_base="stylized stadium background with cheering fans in blue and green tones. In the foreground, a cartoon relay race unfolds with a rabbit in red and blue running gear smiling and passing a baton to a bear in green soccer attire, who stands on one foot with enthusiasm. Next in line, a determined fox in yellow biking gear reaches out for the baton, poised to continue the race. Above the scene, the words \"Go Farther Together!\" are boldly displayed in a striking color and font. The baton, red and white, symbolizes the spirit of teamwork as it moves between the animated animals."
    # data.append(load_gpt_output(fox,"", fox_base))
    for count,i in enumerate(debug_list):
        if i==0:
            item=data[-1].copy()
        else:
            item=data[i].copy()
        layer_num=len(item['layers'])+1
        item['layers'].reverse()
        glyph_item=None
        for n,g in enumerate(glyph_data):
            if g['index']==item['index']:
                glyph_item=g
                if not all and n >=30:
                    glyph_item=None
        if glyph_item is None:
            continue
        for length in range(len(item['layers'])+1):
            layer_prompt=''
            bboxes=[]
            layer_prompt_big=''
            bboxes_big=[]
            background_prompt=''
            background_bbox=[]
            for n,layer in enumerate(item['layers'][:length].copy()):
                # if n in [0,3,5,8]:
                #     continue
                if count==0 and n==4:
                    continue
                if count==1 and n==4:
                    layer['top_left'][0]+=600
                    layer['top_left'][1]-=400
                    layer['bottom_right'][0]+=600
                    layer['bottom_right'][1]-=400
                if "background" in layer.keys():
                    background_prompt+=layer['background']+' BREAK\n'
                    background_bbox.append([0,0,1457,1457])
                    continue
                if layer["layer_num"] in [x["layer_num"] for x in glyph_item['layers']]:
                    caption=[x["caption"] for x in glyph_item['layers'] if x["layer_num"]==layer["layer_num"]][0]
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=caption+' GLYPH BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=caption+' GLYPH BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=layer['caption']+' BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=layer['caption']+' BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                
            layer_prompt=background_prompt+layer_prompt_big+layer_prompt
            bboxes=background_bbox+bboxes_big+bboxes
            layer_prompt=item['base_image']['caption']+' BREAK\n'+layer_prompt
            bboxes.insert(0,[0,0,1457,1457])
            index=item['index']
            
            if length!=len(item['layers']):
                index=str(index)+'_'+str(length)
            if count==0:
                index=str(index)+'_move1'
            if count==1:
                index=str(index)+'_move2'
            base_prompt=item['whole_image']['caption']
            processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
    


    return processed_data

def load_customized_debug2():
    fox=[ { "background": "The layer is a stylized stadium full of cheering fans, using a blend of blue and green colors to represent the atmosphere. The stadium is designed with lines and shapes to suggest depth and perspective, creating a dynamic and engaging scene." }, { "layer": 0, "category": "element", "caption": "A cartoon rabbit dressed in running gear, wearing a red tank top, blue shorts, and a white headband. The rabbit is in a running position, smiling and passing the baton to the next animal.", "top_left": [ 95, 450 ], "bottom_right": [ 395, 1050 ] }, { "layer": 1, "category": "element", "caption": "A cartoon bear dressed in soccer attire, wearing a green jersey, white shorts, and soccer cleats. The bear is standing on one foot while receiving the baton from the rabbit with an enthusiastic expression.", "top_left": [ 390, 455 ], "bottom_right": [ 690, 1100 ] }, { "layer": 2, "category": "element", "caption": "A cartoon fox dressed in biking gear, wearing a yellow cycling jersey, black shorts, a helmet, and cycling gloves. The fox is holding the handlebars of a bike with one hand and receiving the baton from the bear with the other hand, showing a determined and focused expression.", "top_left": [ 675, 400 ], "bottom_right": [ 1175, 1075 ] }, { "layer": 3, "category": "text", "caption": "Text \"Go Farther Together!\n\" in <color-2>, <font-137>. ", "top_left": [ 355, 125 ], "bottom_right": [ 1100, 250 ] }, { "layer": 4, "category": "element", "caption": "A cartoon baton in red and white colors, representing collaboration and support, being passed between the animals in the relay race.", "top_left": [ 355, 725 ], "bottom_right": [ 685, 775 ] } ]
    fox_base="stylized stadium background with cheering fans in blue and green tones. In the foreground, a cartoon relay race unfolds with a rabbit in red and blue running gear smiling and passing a baton to a bear in green soccer attire, who stands on one foot with enthusiasm. Next in line, a determined fox in yellow biking gear reaches out for the baton, poised to continue the race. Above the scene, the words \"Go Farther Together!\" are boldly displayed in a striking color and font. The baton, red and white, symbolizes the spirit of teamwork as it moves between the animated animals."
    data=[]
    data.append(fox)
    processed_data=[]
    for item in data:
        
        layer_num=len(item)
        for length in range(1,len(item)):
            layer_prompt=''
            bboxes=[]
            layer_prompt_big=''
            bboxes_big=[]
            background_prompt=''
            background_bbox=[]
            for n,layer in enumerate(item.copy()):
                if "background" in layer.keys():
                    background_prompt+=layer['background']+' BREAK\n'
                    background_bbox.append([0,0,1457,1457])
                    continue
                if layer["category"]=='text':
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=layer['caption']+' GLYPH BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=layer['caption']+' GLYPH BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                else:
                    if layer["bottom_right"][1]-layer["top_left"][1]>1200 and layer["bottom_right"][0]-layer["top_left"][0]>1200:
                        layer_prompt_big+=layer['caption']+' BREAK\n'
                        bboxes_big.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                    else:
                        layer_prompt+=layer['caption']+' BREAK\n'
                        bboxes.append([layer["top_left"][1],layer["top_left"][0],layer["bottom_right"][1],layer["bottom_right"][0]])
                
            layer_prompt=background_prompt+layer_prompt_big+layer_prompt
            bboxes=background_bbox+bboxes_big+bboxes

            index=101
            
            if length!=len(item):
                index=str(index)+'_'+str(length)
            base_prompt=fox_base
            processed_data.append({'layer_num':layer_num,'Layer Prompt':layer_prompt,'bboxes':bboxes,'index':index,'Base Prompt':base_prompt})
    


    return processed_data



if __name__ == "__main__":
    # data = load_layout()
    # for group in data:
    #     for item in group['data']:
    #         index=item['index']
    #         bboxes=item['bboxes']
    #         draw_bbox(index,bboxes)
    # data=load_intention_output_data()
    # for item in data:
    #     index=item['index']
    #     bboxes=item['bboxes']
    #     source_dir='/pyy/openseg_blob/yuyang/code/RPG/try_pipeline'
    #     target_dir='/pyy/openseg_blob/yuyang/code/RPG/try_pipeline_bbox'
    #     draw_bbox(index,bboxes,source_dir,target_dir)

    # glyph_file="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts.json"
    # with open(glyph_file, 'r') as f:
    #     glyph_data = json.load(f) 
    # for i in range(len(glyph_data)):
    #     for j in range(len(glyph_data[i]['layers'])):
    #         glyph_data[i]['layers'][j]['caption']=glyph_data[i]['layers'][j]['caption'].split("\"")[1]
    # glyph_file="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_text_layer_prompts_ocr.json"
    # with open(glyph_file, 'w') as f:
    #     json.dump(glyph_data, f, indent=4)

    load_customized_debug2()

    # load_inference_data_with_glyph()
    # sample=[[ { "background": "The layer is completely light blue color, representing a clear sky." }, { "layer": 0, "category": "element", "caption": "The layer features a cartoon-style illustration of a chicken with a round body, large eyes, and a bright red comb and wattle. The chicken is standing on one leg, with the other leg lifted in a playful pose, and it's colored in a combination of white and light yellow shades.", "top_left": [ 150, 450 ], "bottom_right": [ 450, 1000 ] }, { "layer": 1, "category": "element", "caption": "Another cartoon-style chicken with a round body, large eyes, and a bright red comb and wattle. This chicken is standing on both legs, facing the left side of the poster, and is colored in various shades of brown.", "top_left": [ 600, 530 ], "bottom_right": [ 900, 1080 ] }, { "layer": 2, "category": "element", "caption": "The third cartoon chicken, featuring a round body, large eyes, and a bright red comb and wattle. The chicken is standing on both legs, facing the right side of the poster, and is colored in a combination of light orange and white shades.", "top_left": [ 1050, 460 ], "bottom_right": [ 1350, 1010 ] }, { "layer": 3, "category": "element", "caption": "The first cartoon-style duck, characterized by an elongated body, large eyes, and a bright orange beak. The duck is standing on both legs, facing the left side of the poster, and is colored in various shades of green.", "top_left": [ 300, 1100 ], "bottom_right": [ 600, 1450 ] }, { "layer": 4, "category": "element", "caption": "A second cartoon-style duck with an elongated body, large eyes, and a bright orange beak. The duck is standing on both legs, facing the right side of the poster, and is colored in a combination of light blue and white shades.", "top_left": [ 900, 1120 ], "bottom_right": [ 1200, 1470 ] } ],
    # [ { "background": "The layer is completely light beige color." }, { "layer": 0, "category": "element", "caption": "The layer features a wooden table with a realistic wood grain texture in warm brown hues, covering the bottom half of the canvas to create the impression of a cozy, intimate setting.", "top_left": [ 0, 728 ], "bottom_right": [ 1456, 1457 ] }, { "layer": 1, "category": "element", "caption": "An open book with visible pages and text, resting on the wooden table. The book's cover and binding are a soft, earthy brown color, blending harmoniously with the warm tones of the wooden surface.", "top_left": [ 488, 805 ], "bottom_right": [ 968, 1207 ] }, { "layer": 2, "category": "element", "caption": "An orange-hued cat sitting attentively in front of the open book, its body facing the viewer but its head turned to the side, as if gazing curiously at the pages. The cat's fur is rendered in rich, warm tones, with darkened shadows and highlights to suggest depth and texture. The cat's eyes are bright and inquisitive, further emphasizing the intellectual and cozy ambiance of the scene.", "top_left": [ 679, 557 ], "bottom_right": [ 1249, 1046 ] }, { "layer": 3, "category": "element", "caption": "A soft, warm light source is illustrated, casting a gentle glow onto the scene. The light is diffuse, creating a cozy and inviting atmosphere that accentuates the rich colors and textures of the cat and the wooden table.", "top_left": [ 0, 0 ], "bottom_right": [ 1456, 1457 ] } ],
    # [ { "background": "The layer is completely light blue color." }, { "layer": 0, "category": "element", "caption": "Five whole apples arranged in a neat row, with the first three being red and the last two being green. All apples have a simple leaf on top, and the colors are bright and eye-catching to appeal to young children.", "top_left": [ 578, 250 ], "bottom_right": [ 878, 450 ] }, { "layer": 1, "category": "element", "caption": "A large hand-drawn arrow, colored in yellow, pointing from the two green apples to the right, indicating that they are being taken away from the group of five apples.", "top_left": [ 889, 295 ], "bottom_right": [ 1005, 405 ] }, { "layer": 2, "category": "element", "caption": "Two green apples, separate from the original group and placed to the right of the arrow, illustrating that they have been removed from the group.", "top_left": [ 1020, 250 ], "bottom_right": [ 1245, 450 ] }, { "layer": 3, "category": "text", "caption": "Text \"Subtraction\n\" in <color-31>, <font-97>. ", "top_left": [ 50, 100 ], "bottom_right": [ 550, 200 ] }, { "layer": 4, "category": "text", "caption": "Text \"5 - 2 = 3\n\" in <color-31>, <font-97>. ", "top_left": [ 400, 500 ], "bottom_right": [ 1100, 600 ] }, { "layer": 5, "category": "text", "caption": "Text \"Take away two apples\nfrom five apples\n\" in <color-31>, <font-59>. ", "top_left": [ 50, 700 ], "bottom_right": [ 550, 800 ] }, { "layer": 6, "category": "text", "caption": "Text \"You have three apples left!\n\" in <color-31>, <font-59>. ", "top_left": [ 50, 850 ], "bottom_right": [ 550, 950 ] }, { "layer": 7, "category": "element", "caption": "A yellow sun with a smiling face, placed at the top right corner of the poster, adding a friendly and cheerful atmosphere to the design.", "top_left": [ 1256, 56 ], "bottom_right": [ 1456, 256 ] } ],
    # [ { "background": "The layer is a stylized stadium full of cheering fans, using a blend of blue and green colors to represent the atmosphere. The stadium is designed with lines and shapes to suggest depth and perspective, creating a dynamic and engaging scene." }, { "layer": 0, "category": "element", "caption": "A cartoon rabbit dressed in running gear, wearing a red tank top, blue shorts, and a white headband. The rabbit is in a running position, smiling and passing the baton to the next animal.", "top_left": [ 95, 450 ], "bottom_right": [ 395, 1050 ] }, { "layer": 1, "category": "element", "caption": "A cartoon bear dressed in soccer attire, wearing a green jersey, white shorts, and soccer cleats. The bear is standing on one foot while receiving the baton from the rabbit with an enthusiastic expression.", "top_left": [ 390, 455 ], "bottom_right": [ 690, 1100 ] }, { "layer": 2, "category": "element", "caption": "A cartoon fox dressed in biking gear, wearing a yellow cycling jersey, black shorts, a helmet, and cycling gloves. The fox is holding the handlebars of a bike with one hand and receiving the baton from the bear with the other hand, showing a determined and focused expression.", "top_left": [ 675, 400 ], "bottom_right": [ 1175, 1075 ] }, { "layer": 3, "category": "text", "caption": "Text \"Go Farther Together!\n\" in <color-2>, <font-137>. ", "top_left": [ 355, 125 ], "bottom_right": [ 1100, 250 ] }, { "layer": 4, "category": "element", "caption": "A cartoon baton in red and white colors, representing collaboration and support, being passed between the animals in the relay race.", "top_left": [ 355, 725 ], "bottom_right": [ 685, 775 ] } ]]
    # index=[4,5,6,7]
    # from RPG_pipeline import resize_bbox
    # bboxes=[]
    # for i in sample:
    #     bbox=[]
    #     for j in i:
    #         if 'background' in j.keys():
    #             bbox.append([0,0,1457,1457])
    #             continue
    #         temp=[j['top_left'][1],j['top_left'][0],j['bottom_right'][1],j['bottom_right'][0]]
    #         bbox.append(temp)
    #     bboxes.append(bbox)
    # source_dir='/pyy/openseg_blob/yuyang/code/RPG/try_pipeline'
    # target_dir='/pyy/openseg_blob/yuyang/code/RPG/try_pipeline_bbox'
    # for i in range(len(index)):
    #     draw_bbox(index[i],bboxes[i],source_dir,target_dir)

