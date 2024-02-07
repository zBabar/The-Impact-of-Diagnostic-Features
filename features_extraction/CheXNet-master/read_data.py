# encoding: utf-8

"""
Read images and corresponding labels.
"""
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
        """
        #data=pd.read_csv(image_list_file)
        data = pd.read_json(image_list_file)
        #image_names=data['Image Index'].to_list()
        image_names = data['image_path'].to_list()
        self.image_names = [data_dir+image[0] for image in image_names]
        self.labels = [[0]*12+[1]*2]*data.shape[0]#labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        file=image_name.split('/')[-1]
        if self.transform is not None:
            image = self.transform(image)
        return file, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

