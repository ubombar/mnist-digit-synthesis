import torch
import torchvision
import torch.nn as nn 
from tqdm import tqdm
import dataset as ds 
import models as m
import random
import torch.optim as optim 
import os 

ONE_EPOCH = 100
BATCH_SIZE = 1
DIGIT = 1
CHANCE_DELUDE = 0.5
LR1 = 0.001
LR2 = 0.001

# define transform and dataloader
transform = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
dataloader = torch.utils.data.DataLoader(
    ds.MNISTDiscriminatorDataset(transform, digit=DIGIT), 
    batch_size=BATCH_SIZE, 
    shuffle=True)

# define model and optimizer
discriminator = m.Discriminator()
generator = m.Generator()
criterion = nn.CrossEntropyLoss()

optimizer1 = optim.SGD(discriminator.parameters(), lr=LR1)
optimizer2 = optim.SGD(generator.parameters(), lr=LR2)

snaps = os.listdir('./snaps')
snaps.remove('.gitkeep')

if len(snaps) > 0:
    latest = max(snaps)
    path = f'./snaps/{latest}'

    torch_object = torch.load(path)

    discriminator.load_state_dict(torch_object['discriminator'])
    generator.load_state_dict(torch_object['generator'])

no_generated = 0
no_original = 0

for i in range(ONE_EPOCH):
    is_fake = Truerandom.random() > CHANCE_DELUDE

    if is_fake: # run generator
        no_generated += 1
        input_image = generator()
    else:
        no_original += 1
        input_image = next(dataloader.__iter__()) # infinite dataloader

    pred_fake = discriminator(input_image)
    loss = criterion(pred_fake, is_fake)
    loss.backwards()

# TODO needs more mork not finished yet

