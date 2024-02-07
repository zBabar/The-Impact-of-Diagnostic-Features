from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
import pickle
import numpy as np


scaler = MinMaxScaler()

path = '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/r2gen_annotations/'

with open(path + 'images_features_2d.pickle', 'rb') as handle:
#with open(path + 'densenet_image_features.pkl', 'rb') as handle:
    visual_features = pickle.load(handle)
feature_size = 1024

with open(path + 'images_features_new.pickle', 'rb') as handle:
    image_features = pickle.load(handle)

with open('disease_embeddings_log.pickle', 'rb') as handle:
    disease_embeddings = pickle.load(handle)

with open(path + 'chexbert_chex_r2gen.pkl', 'rb') as handle:
    semantic_features = pickle.load(handle)

def get_features(visual_features):
    new_features={}
    for key, value in visual_features.items():
        report=key.split('_')[0]
        if report not in new_features:
            print(report)
            new_features[report]=[value]
        else:
            print(report)
            print(type(new_features[report]))
            temp=(new_features[report])
            new_features[report]=temp+[value]
        #print(new_features[key][0].shape)
    return new_features

def feature_embedding_append(feature_map, semantic_feature):
    print(feature_map.shape)

    feature_map = np.reshape(feature_map,(49,feature_size))

    new_feature_map = np.ndarray([49, feature_size+200], dtype=np.float32)
    for i in range(feature_map.shape[0]):
        new_feature_map[i,:] = np.append(feature_map[i,:], semantic_feature)

    new_feature_map = scaler.fit_transform(new_feature_map)
    new_feature_map = np.reshape(new_feature_map,(1,7,7,feature_size+200))

    return new_feature_map

def feature_semantic_append(feature_map, semantic_feature):
    feature_map = np.reshape(feature_map, (49, feature_size+200))

    new_feature_map = np.ndarray([49, feature_size+214], dtype=np.float32)
    for i in range(feature_map.shape[0]):
        new_feature_map[i,:] = np.append(feature_map[i,:], semantic_feature)

    new_feature_map = scaler.fit_transform(new_feature_map)
    new_feature_map = np.reshape(new_feature_map,(1,7,7,feature_size+214))

    return new_feature_map

def get_features_chexbert(visual_features, disease_embeddings , semantic_features): # ysing chexnet based predicted features
    new_features={}

    for key, value in visual_features.items():

        report = key.split('_')[0]


        if report in semantic_features:

            value = feature_embedding_append(value,disease_embeddings[key]) # chexnet based predictions
            #value = feature_embedding_append(value, disease_embeddings[report]) # chexbert based prediction
            #value = feature_semantic_append(value, image_features[key])

            if report not in new_features:
                new_features[report] = [value]
            else:
                # print(report)
                # print(type(new_features[report]))
                temp = new_features[report]
                new_features[report] = temp + [value]
            print(new_features[report][0].shape)
    return new_features

new_features = get_features(visual_features)

with open('images_features_2d_report.pickle', 'wb') as handle:
    pickle.dump(new_features, handle, protocol=2)


new_features_chexbert = get_features_chexbert(visual_features, disease_embeddings, semantic_features)

with open('images_semantic_features_2d_report.pickle', 'wb') as handle:
    pickle.dump(new_features_chexbert, handle, protocol=2)
