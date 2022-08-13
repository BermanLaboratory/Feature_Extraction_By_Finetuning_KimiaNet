import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import cv2

from tqdm import tqdm
from torchvision import transforms

from PIL import Image
import json

from ..tiles_dataset import Tiles_Selected_CSV

device      = torch.device('cpu') 
num_workers = 4
image_size  = 1000 
batch_size  = 8
data_path_csv = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_clustering_200.json'


class Tiles_Selected_CSV(Dataset):

    def __init__(self,transform,selected_patches):
        """
        Args: 
            data_path : path to input data
            transform : transformation function
            selected_patches: list of selected_patches with their file paths
            labels_dict : labels associated with each whole slide
        """
        self.image_patches = selected_patches
        self.transform = transform


    def __len__(self):
        
        return len(self.image_patches)
    
    def __getitem__(self,index):

        image = Image.open(self.image_patches[index]).convert('RGB')
        image = self.transform(image)
        
        return image


transform = transforms.Compose([
		transforms.ToTensor()
		
	])

with open('/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_clustering_200.json', 'r') as f:
        selected_patches = json.load(f)

dataset = Tiles_Selected_CSV(transform,selected_patches)

image_loader = DataLoader(dataset,batch_size = 16,shuffle = False,num_workers = 50)

psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])


for inputs in tqdm(image_loader):
    psum    += inputs.sum(axis        = [0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])


# pixel count
count = len(selected_patches) * image_size * image_size

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))