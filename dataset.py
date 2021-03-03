from PIL import Image
import torch 
import os 
import torchvision.transforms.functional as F 

class MNISTDiscriminatorDataset():
    def __init__(self, transform, digit=3):
        self.path = f'./data/{digit}.pt'

        self.digit = digit
        self.transform = transform

        self.tensor = transform(torch.load(self.path))
    
    def __len__(self):
        return self.tensor.shape[0]
    
    def __getitem__(self, i):
        assert i >= 0 and i < len(self)
        return self.tensor[i]