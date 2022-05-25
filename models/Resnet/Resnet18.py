from turtle import forward
import torch
import torch.nn as nn
from torchvision import models

from Frame.ModelBase import ModelBase

    
class Resnet18(ModelBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.network(x)
    