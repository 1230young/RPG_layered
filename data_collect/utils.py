import json
#这里输入你的文件路径
file_name = '/pyy/openseg_blob/weicong/big_file/data/canva-data/canva-render-11.30/cdfs_with_masks_v2.json'
save_example_file = '/pyy/openseg_blob/yuyang/big_file/canva/cdfs_with_masks_v2_example.json'
json_dump=[]
with open(file_name, 'r', encoding='utf-8') as f:
    for i in range(100):
        line = f.readline()
        if not line: # 到 EOF，返回空字符串，则终止循环
            break
        js = json.loads(line)
        json_dump.append(js)
with open(save_example_file, 'w', encoding='utf-8') as f:
    json.dump(json_dump, f, ensure_ascii=False, indent=4)
