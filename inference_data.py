import json

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

if __name__ == "__main__":
    data = load_inference_data()
    print(data[0])