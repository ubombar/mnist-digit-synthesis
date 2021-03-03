from PIL import Image
import os 
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
import torch

# small preprocess script!

os.mkdir('./data')
for i in range(0, 10):
    trains = os.listdir(f'./raw/train/{i}')
    tests = os.listdir(f'./raw/test/{i}')

    all_files = []

    for name in trains:
        full_path = os.path.join(f'./raw/train/{i}', name)
        all_files.append((full_path, name))

    for name in tests:
        full_path = os.path.join(f'./raw/test/{i}', name)
        all_files.append((full_path, name))

    tensor_list = []

    for full_path, name in tqdm(all_files):
        tensor_list.append(to_tensor(Image.open(full_path)))
    
    total_tensor = torch.cat(tensor_list)

    torch.save(total_tensor, f'./data/{i}.pt')
