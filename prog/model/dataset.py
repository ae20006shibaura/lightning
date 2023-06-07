import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, dir_path,transform=None):
        
        super().__init__()
        
        self.dir_path = dir_path
        self.transform = transform
        self.image_paths = [str(p) for p in Path(self.dir_path).glob("**/*.jpg")]
        self.len = len(self.image_paths)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        p = self.image_paths[index]

        image = Image.open(p)
        image = image.convert("RGB")
        image = self.transform(image)  

        
        # ラベル (「1」: 1, 「2」: 2)
        label = p.split("\\")[4]
        label = 1 if label == "not" else 2
        
        return image, label
    
transform = transforms.Compose([transforms.ToTensor(), 
                                  transforms.Resize(size=(64, 64)),
                              transforms.Normalize(
                                  [0.5, 0.5, 0.5],  
                                  [0.5, 0.5, 0.5], 
                               )]) 


dir_path = 'prog/model/mask/train'
dataset = MyDataset(dir_path,transform)
train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[15, 5], generator=torch.Generator().manual_seed(42))