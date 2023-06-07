import model.Dataset as Dataset

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as pl
from pytorch_lightning import Trainer

class Net(pl.LightningModule):
 
    def __init__(self, batch_size= 2):
        super(Net, self).__init__()
        self.batch_size = batch_size
        # resnet
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        
    def forward(self, x):
        x = self.model(x)
        return x
 
    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer
    
    def lossfun(self, y, t):
        return F.cross_entropy(y, t)
   
    #traindata
    def train_dataloader(self):
        return torch.utils.data.DataLoader(Dataset.train, self.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results
    
    #valdata
    def val_dataloader(self):
        return torch.utils.data.DataLoader(Dataset.val, self.batch_size)
    
    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)        
        results = {'val_loss': loss, 'val_acc': acc}
        return results
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc  =torch.stack([x['val_acc'] for x in outputs]).mean()
        results = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return results
    
    # New: テストデータセットの設定
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(Dataset.test, self.batch_size)
    
    # New: テストデータに対するイテレーションごとの処理
    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results
    
    # New: テストデータに対するエポックごとの処理
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results