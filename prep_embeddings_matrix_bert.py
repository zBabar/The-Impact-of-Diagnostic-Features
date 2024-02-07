import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json
import pickle

scaler = MinMaxScaler()

CLASS_NAMES = [ 'enlarged cardiomediastinum', 'cardiomegaly', 'lung opacity', 'lung lesion', 'edema', 'consolidation', 'pneumonia',
                'atelectasis', 'pneumothorax', 'pleural effusion', 'pleural other','fracture', 'support devices', 'no finding']


path = '/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image_norm_abnorm_split/r2gen_annotations/'

embeddings = {}

with open(path + 'chexbert_chex_r2gen.pkl', 'rb') as handle:
    semantic_features = pickle.load(handle)

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

def create_embed_matrix(embedding_matrix_vocab):

    embed_matrix = np.ndarray([14, 200], dtype=np.float32)

    for i,disease in enumerate(CLASS_NAMES):

        embed_matrix[i,:] = embedding_matrix_vocab[disease]

    return embed_matrix

def process_embeddings(diseases_weights, embedding_matrix):

        x = np.average(embedding_matrix, axis=0 , weights= diseases_weights)

        return x #1 / (1 + np.exp(-x))


def get_embeddings(image_features): # using chexnet based predicted features
    embeddings = {}

    with open('/home/zaheer/pythonCode/embeddings/pretrained_14disease_embeddings_bert.pickle', 'rb') as handle:
        embedding_matrix_vocab = pickle.load(handle)
    # embedding_matrix_vocab = embedding_for_vocab('/home/zaheer/pythonCode/embeddings/glove.6B.100d.txt', 100)
    print(embedding_matrix_vocab.keys())

    embedding_matrix_vocab = {k.lower(): v for k, v in embedding_matrix_vocab.items()}

    embed_matrix = create_embed_matrix(embedding_matrix_vocab)

    for key, value in  semantic_features.items():


        embeddings[key] = process_embeddings(value, embed_matrix)

    return embeddings

diseases_embeddings = get_embeddings(semantic_features)

with open('disease_embeddings_log_bert.pickle', 'wb') as handle:
    pickle.dump(diseases_embeddings, handle, protocol=2)
