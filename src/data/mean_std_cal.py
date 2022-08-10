

import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt



####### PARAMS

device      = torch.device('cpu') 
num_workers = 4
image_size  = 512 
batch_size  = 8
data_path   = '/kaggle/input/cassava-leaf-disease-classification/'


augs = A.Compose([A.Resize(height = image_size, 
                           width  = image_size),
                  A.Normalize(mean = (0, 0, 0),
                              std  = (1, 1, 1)),
                  ToTensorV2()])
    