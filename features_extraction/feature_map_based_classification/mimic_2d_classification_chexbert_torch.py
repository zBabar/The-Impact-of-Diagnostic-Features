#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install torcheval
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import f1_score, roc_auc_score
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


# In[2]:


# !pip install torcheval


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
num_classes = 14


# In[9]:


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


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len


# In[12]:


images_train,y_train,images_test,y_test, images_val,y_val=load_preprocess_data()

train_features=load_features(images_train)
test_features=load_features(images_test)
val_features=load_features(images_val)


# In[13]:


print(len(images_test))


# In[14]:


# Instantiate training and test data

batch_size = 16

train_data = Data(train_features, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
train_dataloader_infer = DataLoader(dataset=train_data, batch_size=len(images_train), shuffle=False)

test_data = Data(test_features, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=len(images_test), shuffle=False)


val_data = Data(val_features, y_test)
val_dataloader = DataLoader(dataset=val_data, batch_size=len(images_val), shuffle=False)


# In[15]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(feature_shape, num_classes).cuda()
        #self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.mean(x,1)
        #x= self.drop(x)
        # x = self.pool(x)
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.sigmoid(self.fc1(x))
        return x
model = Net()
print(model)


# In[16]:


loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# In[17]:



def testAccuracy():
    
    net.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            #print(outputs)
            # print(labels)
            # the label with the highest energy will be our prediction
            
            # print(roc_auc_score(labels, outputs))

num_epochs = 20
loss_values = []

def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    #net.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        for inputs, labels in train_dataloader:
            
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # get the inputs
            # images = Variable(inputs.to(device))
            # labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(inputs)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            
            loss_values.append(loss.item())
            # backpropagate the loss
            loss.backward()
            
           
            optimizer.step()

train(num_epochs)
print('Finished Training')

# testAccuracy()


# In[18]:


step = np.linspace(0, 20, 12500)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# In[19]:


y_train.shape


# In[20]:


"""
We're not training so we don't need to calculate the gradients for our outputs
"""

with torch.no_grad():
    for X, y in test_dataloader:
        
        if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
        outputs = model(X)
        test_tags = outputs.cpu().numpy()
    for X, y in val_dataloader:
        
        if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
        outputs = model(X)
        val_tags = outputs.cpu().numpy()
    
    for X, y in train_dataloader_infer:
        
        if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
        outputs = model(X)
        train_tags = outputs.cpu().numpy()


# In[21]:


import pickle

all_tags={}
all_reports = []
test_true = np.array(y_test)
for idx,image in enumerate(images_test):
    print(image)
    all_tags[image] = test_tags[idx,:]
    #all_tags[image] = test_tags[idx,:]
    #all_tags[image.split('_')[0]]=test_true[idx,:]
    #all_reports.append(image.split('_')[0])

val_true = np.array(y_val)
for idx,image in enumerate(images_val):
    all_tags[image] = val_tags[idx,:]
    #all_tags[image] = val_tags[idx,:]
    #all_tags[image.split('_')[0]]=val_true[idx,:]
    #all_reports.append(image.split('_')[0])
    
train_true=np.array(y_train)
for idx,image in enumerate(images_train):
    # all_tags[image.split('_')[0]]=train_true[idx,:]
    all_tags[image] = train_tags[idx,:]
    #all_tags[image] = train_tag[idx,:]
    #all_reports.append(image.split('_')[0])
#print(all_tags['CXR3614'])
#with open(path+'chexbert_chex_r2gen.pkl','wb') as file:
with open(path+'mimic_chexbert_dense.pkl','wb') as file:
    pickle.dump(all_tags, file, protocol=2)
    


# In[22]:


get_ipython().system('jupyter nbconvert --to script mimic_2d_classification_chexbert.ipynb')


# In[ ]:




