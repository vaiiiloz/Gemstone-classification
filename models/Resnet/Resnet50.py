from turtle import forward
import torch
import torch.nn as nn
from torchvision import models

from Frame.ModelBase import ModelBase

class Resnet(ModelBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        
        self.network.fc = nn.Linear(in_features=2048,out_features=num_classes, bias=True)
        
    def forward(self, x):
        return self.network(x)
    
        