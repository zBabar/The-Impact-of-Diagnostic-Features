import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
import pickle

scaler = MinMaxScaler()

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
#data_path='/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/normal/Sample1/'

with open('data/mimic/chexnet_mimic_features.pickle', 'rb') as handle: # loading chexnet based predicted scores
     image_features = pickle.load(handle)

feature_size = 1

def feature_append(feature_map, semantic_feature):
    #print(feature_map.shape)

    feature_map = np.reshape(feature_map,(49,feature_size))

    new_feature_map = np.ndarray([49, feature_size+14], dtype=np.float32)
    for i in range(feature_map.shape[0]):
        new_feature_map[i,:] = np.append(feature_map[i,:], semantic_feature)
    print(new_feature_map.shape)
    new_feature_map = scaler.fit_transform(new_feature_map)
    new_feature_map = np.reshape(new_feature_map,(1,7,7,feature_size+14))

    return new_feature_map

#def get_features_chexbert(visual_features, semantic_features): # using chexbert based features
def get_features_chexbert(image_features): # using chexnet based predicted features
    new_features={}

    for key, value in image_features.items():


        value1= np.zeros((1,7,7,1))
        value = feature_append(value1, image_features[key]) # semantic_features[report])

        new_features[key] = value
            #print(new_features[key][0].shape)
    return new_features



new_features_chexbert = get_features_chexbert(image_features)

with open('images_semantic_features_2d_report.pickle', 'wb') as handle:
    pickle.dump(new_features_chexbert, handle, protocol=2)




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

