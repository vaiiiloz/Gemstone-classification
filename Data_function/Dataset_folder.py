import glob
from matplotlib import image
import pandas as pd
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from glob import glob
'''
    Customer class dataset
'''
class Dataset(data.Dataset):
    def __init__(self, class_mapping = None, root_folder = None, transformer = None, phase = None) -> None:
        
        self.transformer = transformer
        self.phase = phase
        #create a label maker
        self.labels = []
        self.classes = set()
        
        self.class_name = {}
        if not class_mapping is None:
            self.class_mapping = class_mapping
            for clas, idx in class_mapping.items():
                self.class_name[idx] = clas
        else:
            self.class_mapping = {}
            for idx, class_folder in enumerate(glob(root_folder+"/*")):
                
                if osp.isdir(class_folder):
                    clas = class_folder.split('/')[-1]
                    self.class_mapping[clas] = idx
                    self.class_name[idx] = clas 
        
        #create image_paths
        self.img_paths = []
        
        for clas in self.class_mapping.keys():
            for img_path in glob(osp.join(root_folder, clas)+'/*'):
                # print(img_path)
                # print()
                self.img_paths.append(img_path)
                self.labels.append(self.class_mapping[clas])

    def __len__(self):
        return len(self.img_paths)
    
    def num_classes(self):
        return len(self.class_mapping)

    def __getitem__(self, index):

        #get image, label_idx
        image = Image.open(self.img_paths[index]).convert('RGB')
        target = self.labels[index]

        #transform image
        if not self.transformer is None: 
            image = self.transformer(image, self.phase)
        return image, target 
