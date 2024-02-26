import json
from PIL import Image, ImageDraw
import os
import numpy as np
# json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'
# dir="/pyy/openseg_blob/yuyang/code/RPG/bboxes/"
# with open(json_file, 'r') as f:
#     data = json.load(f)
# for item in data:
#     bboxes=[layer["top_left"]+layer["bottom_right"] for layer in item['layers']]
#     img_name=[layer["path"].split('/')[-1] for layer in item['layers']]
#     for i in range(len(bboxes)):
#         img=np.zeros((1457,1457,3),dtype=np.uint8)
#         img[bboxes[i][1]:bboxes[i][3],bboxes[i][0]:bboxes[i][2],:]=255


#         img=Image.fromarray(img).resize((1024,1024))
#         img.save(dir+img_name[i])

bbox_dir="/pyy/openseg_blob/yuyang/code/RPG/bboxes/"
layer_dir="/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/generated_imgs/multi_layers_base_0.0_debug_layers"
target_dir="/pyy/openseg_blob/yuyang/code/RPG/layers"
for img in os.listdir(bbox_dir):
    bbox=Image.open(bbox_dir+"/"+img)
    img_name=img.replace("_box","")
    layer=Image.open(layer_dir+"/"+img_name)
    bbox=np.array(bbox,dtype=np.float32)/255
    layer=np.array(layer,dtype=np.float32)/255
    final=(0.6*layer+0.4*bbox*layer)*255
    final=Image.fromarray(final.astype(np.uint8))
    final.save(target_dir+"/"+img_name)


