import torch
import torch.nn as nn
from utils.utils import accuracy
import torch.nn.functional as F
from tqdm import tqdm

class ModelBase(nn.Module):
    def training_step(self, batch, loss_function):
        with torch.enable_grad():
        #go though model
            images, labels = batch
            out = self(images)
            
            #call loss and accu
                
            loss = loss_function(out, labels)
            acc, preds = accuracy(out, labels)
            
            return {'train_loss': loss, 'train_acc': acc}
    
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]
        
        epoch_loss = torch.stack(batch_losses).mean()
        
        batch_accs = [x['train_acc'] for x in outputs]
        
        epoch_acc = torch.stack(batch_accs).mean()
        
        return {'train_loss':epoch_loss.item(), 'train_acc':epoch_acc.item()}
    
    def validation_step(self, batch, loss_function ):
        with torch.no_grad():
            #go though model
            images, labels = batch
            out  = self(images)

            
            #call loss and accu
            loss = loss_function(out, labels)
            acc, preds = accuracy(out, labels)
            
            return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        return {'val_loss': torch.stack([x['val_loss'] for x in outputs]).mean(), 
                'val_acc': torch.stack([x['val_acc'] for x in outputs]).mean()}
        
    def evaluate(self, val_loader, loss_function):
        # self.eval()
        with torch.no_grad():
            
            outputs = [self.validation_step(batch, loss_function) for batch in tqdm(val_loader)]
            return self.validation_epoch_end(outputs)
        
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc:{:.4f}'
            .format(epoch+1, train_result['train_loss'], train_result['train_acc'], val_result['val_loss'], val_result['val_acc']))
        
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()

        batch_accs = [x['val_acc'] for x in outputs]
        
        epoch_acc = torch.stack(batch_accs).mean()

        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]

        batch_labels = [pred for x in outputs for pred in x['labels'].tolist()]

        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                'test_preds': batch_preds, 'test_labels': batch_labels}  
        