import json
from PIL import Image, ImageDraw
import os
import numpy as np
json_file='/pyy/yuyang_blob/pyy/code/RPG-DiffusionMaster/inference/images_100_autocaption_34b-newer.json'
dir="/pyy/openseg_blob/yuyang/code/RPG/bboxes/"
with open(json_file, 'r') as f:
    data = json.load(f)
for item in data:
    bboxes=[layer["top_left"]+layer["bottom_right"] for layer in item['layers']]
    img_name=[layer["path"].split('/')[-1] for layer in item['layers']]
    for i in range(len(bboxes)):
        img=np.zeros((1458,1458,3),dtype=np.uint8)
        img[bboxes[i][1]:bboxes[i][3],bboxes[i][0]:bboxes[i][2],:]=255
        img=Image.fromarray(img)
        img.save(dir+img_name[i])
