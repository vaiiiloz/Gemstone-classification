from turtle import forward
import torch
import torch.nn as nn
import timm 
from ...Frame.ModelBase import ModelBase
class Tresnet(ModelBase):
    def __init__(self, num_labels, model = 'tresnet_l'):
        self.network = timm.create_model(model, pretrained = True, num_classes = num_labels)
    
    def forward(self, x):
        return self.network(x)