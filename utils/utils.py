import torch
from sklearn.metrics import confusion_matrix
import os
'''
    Calculate accuracy
'''
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds)), preds

'''
    Calculate F1_score
'''
def F1_score(outputs, lables):
    _, preds = torch.max(outputs, dim = 1)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))

    return precision, recall, f1, preds

'''
    get learning rate from optimizer
'''
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)
    