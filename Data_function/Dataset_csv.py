import glob
from matplotlib import image
import pandas as pd
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data

'''
    Customer class dataset
'''
class Dataset(data.Dataset):
    def __init__(self, csv = None, transformer = None, phase = None) -> None:
        
        self.transformer = transformer
        self.phase = phase
        if not csv is None:
            data_csv = pd.read_csv(csv)
        else:
            print('No dataset')
        #create a label maker
        self.labels = data_csv['label']
        self.classes = set(self.labels)
        self.class_mapping = {}
        for idx, clas in enumerate(self.classes):
            self.class_mapping[clas] = idx

        #create image_paths
        self.img_paths = []
        for img_path in data_csv['img_path']:
            self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        #get image, label_idx
        image = Image.open(self.img_paths[index]).convert('RGB')
        target = self.class_mapping[self.labels[index]]

        #transform image
        if not self.transformer is None: 
            image = self.transformer(image, self.phase)
        return image, target 
