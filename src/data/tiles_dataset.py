import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import pandas as pd
import torchstain
import cv2
from torchvision import transforms
import numpy as np
from histomicstk.preprocessing.color_normalization import reinhard

class Tiles_Selected_CSV(Dataset):

    def __init__(self,data_path,transform,labels_dict,selected_patches,):
        """
        Args: 
            data_path : path to input dataset
            transform : transformation function
            selected_patches: list of selected_patches with their file paths
            labels_dict : labels associated with each whole slide
        """
        self.data_path = data_path
        self.image_patches = selected_patches
        self.transform = transform
        self.labels_dict = labels_dict
        # self.cnorm = {
        #     'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
        #     'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
        # }


    def __len__(self):
        
        return len(self.image_patches)
    
    def __getitem__(self,index):

        image = Image.open(self.image_patches[index]).convert('RGB')
        patch_name = self.image_patches[index].split('/')[-1]
        image_name = self.image_patches[index].split('/')[-2]
        label = self.labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]

        # tissue_normalized = reinhard(np.array(image), target_mu=self.cnorm['mu'], target_sigma=self.cnorm['sigma'])

        # image = Image.fromarray(np.uint8(tissue_normalized)).convert('RGB')
        image = self.transform(image)
        
        return image,label




class Tiles_Selected(Dataset):

    def __init__(self,data_path,transform,labels_dict):
        """
        Args: 
            data_path : path to input data
            transform : transformation function
        """
        self.data_path = data_path
        self.image_patches = glob(self.data_path+'/**/*.png',recursive = True)
        self.transform = transform
        self.labels_dict = labels_dict
    
    def __len__(self):
        
        return len(self.image_patches)
    
    def __getitem__(self,index):
        image = Image.open(self.image_patches[index]).convert('RGB')
        patch_name = self.image_patches[index].split('/')[-1]
        image_name = self.image_patches[index].split('/')[-2]
        label = self.labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]
        image = self.transform(image)

        return image,label
