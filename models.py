import torch
import torchvision
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequantial = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout2d(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.sequantial(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequantial = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout2d(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.sequantial(x)