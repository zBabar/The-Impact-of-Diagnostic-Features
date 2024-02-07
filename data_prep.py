import pandas as pd
import pickle
import json

### For 5 samples
# path ='/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/Two_Images/word/Data/Sample5/'
#
# def load_transform(split='train'):
#     with open(path +split+'/'+ split+'.json', 'rb') as f:
#         records = json.load(f)
#     print(records[1])
#     new_data=[]
#
#     for record in records:
#
#         new_record={}
#         new_record['id']='-'.join(record['images'][0].split('/')[-1].split('-')[:-1])
#         print(new_record['id'])
#         new_record['report']=record['caption']
#         new_record['split']=split
#         new_record['image_path']=[new_record['id']+'/0.png',new_record['id']+'/1.png']
#         new_data.append(new_record)
#
#
#
#     return new_data
#
# data={}
#
# data['train']=load_transform('train')
# data['test']=load_transform('test')
# data['val']=load_transform('test')
# #
# with open("./data/iu_xray/annotation_5.json", "w") as write_file:
#     json.dump(data, write_file)

####### For file given with the code

# with open('/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/normal/Sample1/binary_tags_chex.pkl','rb') as file:
#     tags=pickle.load(file)

with open('/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/r2gen_annotations/binary_tags_chex_r2gen.pkl','rb') as file:
    tags=pickle.load(file)

with open('./data/iu_xray/annotation.json', 'rb') as f:
        full_records = json.load(f)

splits=['train','val','test']
new_records={}

for s in splits:
    split_records = []
    records=full_records[s]
    for r in records:
        tag=tags[r['id'].split('_')[0]]
        if tag==1:
            split_records.append(r)
        else:
            continue
    new_records[s]=split_records

with open("./data/iu_xray/annotation_01.json", "w") as write_file:
    json.dump(new_records, write_file)