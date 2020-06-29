

import numpy as np
import os
from PIL import Image,ImageFilter


import torch
from torchvision import *

from torch import *

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid





class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)     
        # _,out = torch.max(out,dim = 1)                 
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)      

                                                # Generate predictions
        loss = F.binary_cross_entropy(torch.sigmoid(out), targets) 
       
        score = accuracy(out, targets)
        return {'val_loss': loss.detach(),'val_score':score.detach()}
        
    #this 2 methods will not change .
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))


class Densenet169(ImageClassificationBase):

    def __init__(self):
      super().__init__()
      self.pretrained_model = models.densenet169(pretrained = True)
      
      feature_in = self.pretrained_model.classifier.in_features
      self.pretrained_model.classifier = nn.Linear(feature_in,2)

    def forward(self,x):
      return self.pretrained_model(x)


