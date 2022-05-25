import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from models.Resnet import Resnet50 as Model
import os
import numpy as np
import pandas
from Frame.Framework import Framework
from Data_function.Dataset_csv import Dataset
from Data_function.Transfomer import  Transform
import configparser

if __name__ == '__main__':
    #get data from config file
    config = configparser.ConfigParser()
    config.read('./config/ConfigFile.properties')
    width = int(config["config"]['width'])
    height = int(config['config']['height'])
    resize = (width, height)
    mean = [float(x) for x in config['config']['mean'].split(', ')]
    std = [float(x) for x in config['config']['std'].split(', ')]
    batch = int(config['config']['batch'])
    num_worker = int(config['config']['num_worker'])
    lr = float(config['config']['lr'])
    epochs = int(config['config']['epochs'])
    
    #config function
    framework = Framework()
    transformers = Transform(resize=resize, mean= mean, std=std)
    loss_function = F.cross_entropy
    opt_func = torch.optim.Adam
    
    #Create dataset
    train_dataset = Dataset('./data/classification/train.csv', transformer=transformers, phase='train')
    val_dataset = Dataset('./data/classification/val.csv', transformer=transformers, phase = 'val')
    test_dataset = Dataset('./data/classification/test.csv', transformer=transformers, phase = 'val')
    
    #Create dataloader
    train_dl = DataLoader(dataset = train_dataset, batch_size = batch, num_workers=num_worker, sampler=RandomSampler(train_dataset))
    val_dl = DataLoader(dataset = val_dataset, batch_size = batch*2, num_workers=num_worker, sampler=SequentialSampler(val_dataset))
    test_dl = DataLoader(dataset = test_dataset, batch_size = batch*2, num_workers=num_worker, sampler=SequentialSampler(test_dataset))
    
    #Create model
    model = Model(2)
    framework.fit(model, train_dl= train_dl, val_dl = val_dl, lr= lr, epochs=epochs, opt_func= opt_func, loss_func=loss_function)