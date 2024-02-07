#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pickle
from tqdm import tqdm
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import VGG19,DenseNet121
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model, Sequential
#print(tf.config.list_physical_devices('GPU'))


# In[2]:


base_model = DenseNet121(include_top = False, weights='imagenet',input_shape=(224,224,3), pooling = 'avg')
model = Model(base_model.input, base_model.layers[-4].output)


# In[3]:


# model.summary()


# In[4]:



data = pd.read_json('/home/zaheer/pythonCode/MIMIC_CXR/mimic_10k_all.json')
images_path='/home/zaheer/pythonCode/R2Gen-main/data/mimic/images/'

data.columns


# In[5]:


image_features = {}
for images in data['image_path']:
    image_path=images_path+images[0]
    img = image.load_img(image_path, target_size=(224, 224,3))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    #x=x/255.0
    # Add one more dimension
#         x = np.expand_dims(x, axis=0)

    x=preprocess_input(x)
    x = np.reshape(x,(-1,224,224,3))
    feature = model.predict(x)
    
    image_features[images[0]] = feature


# In[6]:


path='/home/zaheer/pythonCode/MIMIC_CXR/'


# In[7]:


with open(path+'mimic_densenet_image_features.pkl','wb') as file:
    pickle.dump(image_features, file, protocol=2)

