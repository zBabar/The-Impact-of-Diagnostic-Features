import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
import pickle

scaler = MinMaxScaler()

CLASS_NAMES = [ 'atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass', 'nodule', 'pneumonia',
                'pneumothorax', 'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural_thickening', 'hernia','normal']
path = '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/r2gen_annotations/'

embeddings = {}

with open(path + 'images_features_new.pickle', 'rb') as handle:
    image_features = pickle.load(handle)

# with open(path + 'chexbert_chex_r2gen.pkl', 'rb') as handle:
#     semenatic_features = pickle.load(handle)

pre_trained_embeddings = {}
def embedding_for_vocab(filepath, embedding_dim):

    embedding_matrix_vocab = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in CLASS_NAMES:
                embedding_matrix_vocab[word] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]


    return embedding_matrix_vocab


def process_embeddings(diseases, embedding_matrix_vocab):

    if len(diseases) == 1:
        return embedding_matrix_vocab[diseases[0]]
    else:
        temp_embeddings = [embedding_matrix_vocab[disease] for disease in diseases]
        return np.mean(temp_embeddings, axis=0 )


def get_embeddings(image_features): # using chexnet based predicted features
    embeddings = {}

    with open('/home/zaheer/pythonCode/embeddings/pretrained_14disease_embeddings.pickle', 'rb') as handle:
        embedding_matrix_vocab = pickle.load(handle)
    # embedding_matrix_vocab = embedding_for_vocab('/home/zaheer/pythonCode/embeddings/glove.6B.100d.txt', 100)
    print(embedding_matrix_vocab.keys())

    embedding_matrix_vocab = {k.lower(): v for k, v in embedding_matrix_vocab.items()}
    for key, value in  image_features.items():

        if len(np.where(value > 0.6)[0]) > 0:
            diseases = [CLASS_NAMES[idx] for idx in (np.where((value > 0.6))[0]).tolist()]

        elif len(np.where((value < 0.6) & (value > 0.4))[0]) > 0:
            print((np.where((value < 0.6) & (value > 0.4))[0]).tolist())
            diseases = [CLASS_NAMES[idx] for idx in (np.where((value < 0.6) & (value > 0.4))[0]).tolist()]
        else:
            diseases =['normal']
        print(diseases)

        embeddings[key] = process_embeddings(diseases, embedding_matrix_vocab)

    return embeddings

diseases_embeddings = get_embeddings(image_features)

with open('disease_embeddings.pickle', 'wb') as handle:
    pickle.dump(diseases_embeddings, handle, protocol=2)
