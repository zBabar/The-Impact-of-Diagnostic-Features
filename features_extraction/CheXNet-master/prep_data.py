import pickle
import pandas as pd
import numpy as np

# pred=pd.read_csv('predictions.csv')[['filename', 'Atelectasis', 'Cardiomegaly',
#        'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
#        'Pleural_Thickening', 'Hernia']]
#
#
# new={}

# for i,image in enumerate(pred['filename'].to_list()):
#     #new['CXR'+image.split('.')[0]+'.png']=pred.iloc[i,1:].to_numpy()
#     new[image]=pred.iloc[i,1:].to_numpy()
#
# # print(new['CXR3855_IM-1950-1001.png'])
# # print(pred.columns)
#
#
with open('images_features.pickle','rb') as features:

    data=pickle.load(features)


images=list(data.keys())

print(len(images))

new={}
print(data[images[0]][0])


for i,image in enumerate(images):

    print(i)

    new[image]=data[image][0]


with open('images_features_new.pickle', 'wb') as handle:
    pickle.dump(new, handle, protocol=2)