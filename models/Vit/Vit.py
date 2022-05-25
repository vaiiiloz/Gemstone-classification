


import torch
import torch.nn as nn
import timm
class ViT(nn.Module):
    def __init__(self, len_label):
        super().__init__()
        self.network = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
        self.network.fc = nn.Linear(in_features=2048,out_features=len_label, bias=True)
        
    def forward(self, x):
        return self.network(x)