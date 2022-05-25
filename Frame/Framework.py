from pickletools import optimize
import torch
import torch.nn as nn
import pandas as pd
import os
from Data_function.DeviceDataLoader import DeviceDataLoader
import torch.nn.functional as F
# import torch.utils.tensorboard as SummaryWriter
from Frame.ModelBase import ModelBase
from Frame.ModelParalellBase import ModelParalellBase
from tqdm import tqdm
import copy
from utils.utils import get_default_device, get_lr, to_device
import numpy as np
import matplotlib.pyplot as plt
import math

class Framework:
    def __init__(self):
        print('Create frame work')
        
    def visual(self):
        return
    
    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001) 
        
    def visualize_model(self, model, dl, class_names, device = torch.device("cuda"), num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dl):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
        
    def evaluate(self, model: ModelBase, val_loader):
        model.eval()
        with torch.no_grad():
            
            outputs = [model.validation_step(batch, model) for batch in tqdm(val_loader)]
            return model.validation_epoch_end(outputs, model)
    
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc:{:.4f}'
            .format(epoch+1, train_result['train_loss'], train_result['train_acc'], val_result['val_loss'], val_result['val_acc']))
        
    def fit(self, model, train_dl, optimizer, val_dl, lr = 0.01, epochs = 1, device = None,
             loss_func = nn.CrossEntropyLoss, scheduler = None,  
            save_each_model = None, save_model = None, 
            save_history = None, writer = None):
        
        #create loss if none
        
        #add to divice
        if device is None:
            device = get_default_device()
        elif isinstance(device, str):
            device = torch.device(device)
            
        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)
        model = to_device(model, device)

        
        torch.cuda.empty_cache()
        history = {}
        
        #generate optimizer and scheduler
        # optimizer = opt_func(model.parameters(), lr, momentum=0.9)
        
        # scheduler = scheduler_func(optimizer,  step_size = 10, gamma=0.5)
        
        best_model_wts = copy.deepcopy(model.state_dict())
        
        best_loss = 10
        dl = {'train': train_dl, 'val': val_dl}
        #each epochs
        for epoch in range(epochs):
            
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                #run train_loader
                for batch in tqdm(dl[phase]):
                    optimizer.zero_grad()
                    
                    #perform a forward pass (evaluate the model on this training batch)
                    inputs, labels = batch
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = loss_func(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            #Clip the norm of the gradients to 1.0
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            #Perform a backward pass to calculate the gradients
                            loss.backward()
                            #Update parameters and take a stop suing the computed gradient
                            optimizer.step()
                            
                    if math.isnan(loss.item()):
                        exit(0)
                    #statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                        
                        
                        #clear calculated gradients
                    
                ##Update the learning rate
                if phase == 'train' and not scheduler is None:
                    scheduler.step()
                
                #statistics epoch
                epoch_loss = running_loss/len(dl[phase].dl.dataset)
                epoch_acc = running_corrects.double()/len(dl[phase].dl.dataset)
                
                if phase == 'train':
                    train_results = {'train_loss':epoch_loss, 'train_acc':epoch_acc}
                else:
                    val_results = {'val_loss':epoch_loss, 'val_acc':epoch_acc}
                
                #save model with best loss
                if phase == 'val':
                    if val_results['val_loss'] < best_loss:
                        best_loss = min(best_loss, val_results['val_loss'])
                        best_model_wts = copy.deepcopy(model.state_dict())
            
            #print result in each epoch  
            self.epoch_end(epoch, train_results, val_results)
            print("best_loss :" + str(best_loss)+"; lr: "+str(get_lr(optimizer)))
            
            #save model in each epoch
            if save_each_model:
                if not os.path.exists(save_each_model):
                    os.makedirs(save_each_model)
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(save_each_model, f'model_epoch_{epoch}.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(save_each_model, f'model_epoch_{epoch}.pt'))

                
            #save result in each epoch
            to_add = {'train_loss': train_results['train_loss'],
                'train_acc': train_results['train_acc'],
                'val_loss': val_results['val_loss'],
                'val_acc': val_results['val_acc'], 'lrs':get_lr(optimizer)}
            
            if not writer is None:
                writer.add_scalar('Loss/train', train_results['train_loss'], epoch)
                writer.add_scalar('Loss/test', val_results['val_loss'], epoch)
                writer.add_scalar('Accuracy/train', train_results['train_acc'], epoch)
                writer.add_scalar('Accuracy/test', val_results['val_acc'], epoch)

            for key,val in to_add.items():
                if key in history:
                    history[key].append(val)
                else:
                    history[key] = [val]      
                    
            # model.load_state_dict(best_mode_wts)        
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(best_model_wts)
            else:
                model.load_state_dict(best_model_wts)        
                
            

        
        #save history 
        if save_history:
            pd.DataFrame(history).to_csv(save_history)
        
        #save model
        if save_model:
            if isinstance(model, ModelParalellBase):
                    torch.jit.save(model.module.state_dict(), save_model)
            else:
                    torch.jit.save(model.state_dict(), save_model)
        return history, optimizer, best_loss  
    
                
        