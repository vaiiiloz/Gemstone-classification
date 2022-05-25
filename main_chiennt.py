import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from models.Resnet.Resnet50 import Resnet as Model
import os
import numpy as np
import pandas
from Frame.Framework import Framework
from Frame.ModelParalellBase import ModelParalellBase
from Data_function.Dataset_folder import Dataset
from Data_function.Transfomer import  Transform
import configparser


#get data from config file
config = configparser.ConfigParser()
config.read('./config/ConfigFile.properties')
width = int(config["config"]['width'])
height = int(config['config']['height'])
resize = (128, 128)
mean = [float(x) for x in config['config']['mean'].split(', ')]
std = [float(x) for x in config['config']['std'].split(', ')]
batch = int(config['config']['batch'])
batch = 256
num_worker = int(config['config']['num_worker'])
lr = float(config['config']['lr'])
lr = 0.01
epochs = int(config['config']['epochs'])



#config function
framework = Framework()
transformers = Transform(resize=resize, mean= mean, std=std)
loss_function = F.cross_entropy
opt_func = torch.optim.Adam


#Create dataset
train_dataset = Dataset(root_folder='/home/saplap/Desktop/Model/cifar-100/train', transformer=transformers, phase='train')
val_dataset = Dataset(class_mapping = train_dataset.class_mapping, root_folder='/home/saplap/Desktop/Model/cifar-100/val', transformer=transformers, phase = 'val')
test_dataset = Dataset(class_mapping = train_dataset.class_mapping, root_folder='/home/saplap/Desktop/Model/cifar-100/test', transformer=transformers, phase = 'val')

print(len(train_dataset.class_mapping))
#Create model
print(Model)
model = Model(len_label= len(train_dataset.class_mapping))
#Run multiple gpu
model = ModelParalellBase(model)

#Create dataloader
train_dl = DataLoader(dataset = train_dataset, batch_size = batch, num_workers=num_worker, sampler=RandomSampler(train_dataset))
val_dl = DataLoader(dataset = val_dataset, batch_size = batch*2, num_workers=num_worker, sampler=SequentialSampler(val_dataset))
test_dl = DataLoader(dataset = test_dataset, batch_size = batch*2, num_workers=num_worker, sampler=SequentialSampler(test_dataset))

print(model)
framework.fit(model, train_dl= train_dl, val_dl = val_dl, lr= lr, epochs=epochs, opt_func= opt_func, loss_func=loss_function)
