#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import f1_score, roc_auc_score
import json


# In[2]:


#X = pd.read_csv('./IU_Xray/clean_data.csv')

#Y = pd.read_csv('./IU_Xray/labeled_reports.csv')

#Y['report_id'] = X['uid'].apply(lambda x: 'CXR'+str(x))

#Y = Y[['report_id','Enlarged Cardiomediastinum', 'Cardiomegaly',
#       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
#       'Fracture', 'Support Devices', 'No Finding']]#.to_csv('chexbert_labels.csv')

#Y = Y.fillna(0)
#Y.replace(-1,1)
#Y.to_csv('./IU_Xray/chexbert_labels.csv')


# In[3]:


labels_path = '/home/zaheer/pythonCode/MIMIC_CXR/'


# In[4]:


# X = pd.read_csv(labels_path+'mimic_impression.csv')
X = pd.read_json(labels_path+'mimic_50k_all.json')


Y = pd.read_csv(labels_path+'labeled_reports.csv')

Y['report_id'] = [image[0] for image in X['image_path'].tolist()]

Y = Y[['report_id','Enlarged Cardiomediastinum', 'Cardiomegaly',
      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
      'Fracture', 'Support Devices', 'No Finding']]#.to_csv('chexbert_labels.csv')

Y = Y.fillna(0)
Y = Y.replace(-1,1)
Y.to_csv(labels_path+'chexbert_labels.csv', index = False)


# In[5]:


Y.head(6)


# In[6]:


path = labels_path


# In[7]:


with open(path+'mimic_densenet_image_features.pkl','rb') as features:
#with open(path+'densenet_report_features.pkl','rb') as features:
    data=pickle.load(features)


# In[8]:


feature_shape = 1024


# In[9]:


# def clean_r2gen_annotations():
    
#     tags = pd.read_csv('./IU_Xray/chexbert_labels.csv')
    
#     reports = tags['report_id'].tolist()
    
#     #print(reports)
    
#     with open(path+'annotation.json', 'rb') as f:
#         full_records = json.load(f)

#     splits=['train','val','test']
    
#     new_images_records={}
    
#     for s in splits:
#         new_records=[]
#         records=full_records[s]
#         for r in records:
#             for image in r['image_path']:
#                 if image.split('_')[0] in reports: 
#                     new_records.append(r)
#                     break
                
#         new_images_records[s]=new_records
        
#     with open(path+'new_annotation.json', 'w') as f:
#         json.dump(new_images_records , f, ensure_ascii = False )
        
#     return new_images_records

#annot = clean_r2gen_annotations()

cols = ['Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
      'Fracture', 'Support Devices', 'No Finding']

def fetch_tags(image_id, tags):

    new_tags=[]
    image_ids =[]

    for image in image_id:
        tag = tags[tags.report_id == image][cols]
        
        if tag.shape[0] != 0:
            tag = tag.values.tolist()[0]
            image_ids.append(image)
            new_tags.append(tag)

    return image_ids, new_tags


def convert_r2gen_annotations():
    with open(path+'mimic_annotation_10000.json', 'rb') as f:
        full_records = json.load(f)

    splits=['train','val','test']
    
    new_images_records={}
    
    for s in splits:
        images=[]
        records=full_records[s]
        for r in records:
            images.append(r['image_path'][0])
                
        new_images_records[s]=images
    return new_images_records

convert_r2gen_annotations()

def load_preprocess_data():
    tags = pd.read_csv(labels_path+'chexbert_labels.csv')
    converted_records = convert_r2gen_annotations()
    x_train = converted_records['train']
    x_test = converted_records['test']
    x_val = converted_records['val']
    # print(x_train[:5])

    x_train, train_tags=fetch_tags(x_train,tags)
    #print(train_tags)
    #y_train = mlb.fit_transform([train_tags])
    y_train=np.array(train_tags)
    #print(y_train)

    #print(y_train.head())
    
    
    x_test, test_tags=fetch_tags(x_test,tags)
    y_test = np.array(test_tags)
    
    
    x_val, val_tags=fetch_tags(x_val,tags)
    y_val = np.array(val_tags)
    return x_train,y_train,x_test,y_test, x_val,y_val



# In[10]:


def load_features(split):
    features = []#np.empty((0,7,7,1024))
    for image in split:
        features.append(data[image])
    features=np.array(features)
    features=np.reshape(features,(-1,49,feature_shape))
    
    return features


# In[11]:


def class_model(n):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(49,feature_shape)))
    
    model.add(layers.GlobalAveragePooling1D())
    #model.add(layers.Dense(128,activation='relu'))
    #model.add(layers.LeakyReLU(alpha=0.05))
    #model.add(layers.Dense(64,activation='relu'))
    #model.add(layers.Dropout(0.4))
    model.add(layers.Dense(n, activation="sigmoid"))

    model.summary()
    
    return model



# In[ ]:


images_train,y_train,images_test,y_test, images_val,y_val=load_preprocess_data()


# In[ ]:



train_features=load_features(images_train)
test_features=load_features(images_test)
val_features=load_features(images_val)


# In[ ]:


y_train.shape


# In[ ]:


model=class_model(14)


# In[ ]:


earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()
])

model.fit(train_features, y_train, epochs=20, callbacks=[earlyStopping],validation_data=(val_features, y_val), batch_size=16)

test_tags=model.predict(test_features)
val_tags=model.predict(val_features)
train_tag = model.predict(train_features)


# In[ ]:


#test_tags=np.where(test_tags<0.50,0,1)
test_tags


# In[ ]:


#val_tags=np.where(val_tags<0.50,0,1)
val_tags


# In[ ]:


y_test_0=np.where(y_test==-1.0,0,y_test)


# In[ ]:


y_test_1=np.where(y_test==-1.0,1,y_test)


# In[ ]:


#val_tags=[i[0] for i in val_tags.tolist()]
#val_tags=pd.Series(val_tags)


# In[ ]:


#print(f1_score(y_test,test_tags))


# In[ ]:


#np.max(y_test_1[:,cidx])


# In[ ]:


for cidx in range(y_test.shape[1]):
    print(roc_auc_score(pd.Series(y_test_1[:,cidx]), pd.Series(test_tags[:,cidx])))


# In[ ]:


# for cidx in range(y_test.shape[1]):
#     print(f1_score(pd.Series(y_test_0[:,cidx]), pd.Series(test_tags[:,cidx])))


# In[ ]:


m=tf.keras.metrics.Recall()
m.update_state(y_test,test_tags)
m.result().numpy()


# In[ ]:


import pickle

all_tags={}
all_reports = []
test_true = np.array(y_test)
for idx,image in enumerate(images_test):
    print(image)
    all_tags[image] = test_tags[idx,:]
    #all_tags[image] = test_tags[idx,:]
    #all_tags[image.split('_')[0]]=test_true[idx,:]
    all_reports.append(image.split('_')[0])

val_true = np.array(y_val)
for idx,image in enumerate(images_val):
    all_tags[image] = val_tags[idx,:]
    #all_tags[image] = val_tags[idx,:]
    #all_tags[image.split('_')[0]]=val_true[idx,:]
    all_reports.append(image.split('_')[0])
    
train_true=np.array(y_train)
for idx,image in enumerate(images_train):
    # all_tags[image.split('_')[0]]=train_true[idx,:]
    all_tags[image] = train_tag[idx,:]
    #all_tags[image] = train_tag[idx,:]
    all_reports.append(image.split('_')[0])


all_reports = list(set(all_reports))

#print(all_tags['CXR3614'])
#with open(path+'chexbert_chex_r2gen.pkl','wb') as file:
with open(path+'mimic_chexbert_dense.pkl','wb') as file:
    pickle.dump(all_tags, file, protocol=2)
    


# In[ ]:


import numpy as np    
arr = np.empty((0,3), int)
print("Empty array:")
print(arr)
arr = np.append(arr, np.array([[10,20,30]]), axis=0)
arr = np.append(arr, np.array([[40,50,60]]), axis=0)
print("After adding two new arrays:")
print(arr)


# In[ ]:


B = 2*[[3,4]]
A = arr
print(B)
np.insert(A,A.shape[1],B,axis=1)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script mimic_2d_classification_chexbert.ipynb')


# In[ ]:




