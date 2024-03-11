import pandas as pd
import json
import shutil
json_file="/pyy/openseg_blob/yuyang/datasets/RPG/meta_shape1k.json"

with open(json_file, 'r') as f:
    meta = json.load(f)
for n,item in enumerate(meta):
    if item['text_prompt']=="":
        meta[n]['path']=meta[n]['path'].replace("shape1k","val")
    else:
        meta[n]['path']=meta[n]['path'].replace("shape1k","shape1k/train")
with open(json_file, 'w') as f:
    json.dump(meta, f, indent=4)
        
# file_name=[]
# text=[]
# for item in meta:
#     if item['text_prompt']=="":
#         continue
#     file_name.append(item['path'].split("/")[-1])
#     text.append(item['text_prompt'])
# #字典中的key值即为csv中列名
# dataframe = pd.DataFrame({'file_name':file_name,'text':text})

# #将DataFrame存储为csv,index表示是否显示行名，default=True
# dataframe.to_csv("/pyy/openseg_blob/yuyang/datasets/RPG/shape1k/train/metadata.csv",index=False,sep=',')
    
