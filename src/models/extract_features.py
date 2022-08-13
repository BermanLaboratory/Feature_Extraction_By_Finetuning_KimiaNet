from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage import io, transform
import torch.nn.functional as F
from PIL import Image
import pickle	
from models.architechture.model_interface import model_interface
import json
from data.dataloader import dataset_labels
from torch.utils.data.sampler import SubsetRandomSampler

plt.ion()   # interactive mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
save_address_1024 = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'
labels_dict = dataset_labels('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv')
data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class Tiles_Selected_CSV(Dataset):

    def __init__(self,data_path,transform,labels_dict,selected_patches):
        """
        Args: 
            data_path : path to input data
            transform : transformation function
        """
        self.data_path = data_path
        self.image_patches = selected_patches
        # self.image_patches = [patch for patch in self.image_patches if patch.split('/')[-1] in selected_patches]
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

        return image,patch_name

with open("/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_180_with_new.json", 'r') as f:
        selected = json.load(f)
dataset = Tiles_Selected_CSV(train_dir,data_transform, labels_dict,selected)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers = 40)

validation_split = .2
shuffle_dataset = True
random_seed= 13

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
print(len(train_indices))
sampler = {'train':train_sampler,'val':valid_sampler}

def extract_features(model):

    since = time.time()
    model.eval()
    slide_patches_dict_1024 = {}   
    count = 0
    for ii, (inputs, img_name) in enumerate(dataloader):
        inputs = inputs.to(device)
        output1, outputs = model(inputs)
        count = count +1
        # print(count)
        output_1024 = output1.cpu().detach().numpy()
        # output_128 = output2.cpu().detach().numpy()
        for j in range(len(outputs)):
            slide_patches_dict_1024[img_name[j]] = output_1024[j]
        print(len(slide_patches_dict_1024))
    outfile_1024 = open(save_address_1024+'FineTuned_Model_Features_dict.pickle','wb')
    pickle.dump(slide_patches_dict_1024, outfile_1024)
    outfile_1024.close() 

    time_elapsed = time.time() - since
    print('Evaluation completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


weights = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/models/KimiaNetPyTorchWeights.pth'
model = model_interface.load_from_checkpoint('/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/pytorchlightning_lightning_logs/2mndwf27_4/checkpoints/epoch=19-step=56260.ckpt',sampler=sampler,dataset=dataset,kimianet_weights = weights,learning_rate = 0.0001,batch_size = 8)
model = model.to(device)
print(model.learning_rate)

model.eval()

for param in model.parameters():
	param.requires_grad = False

extract_features(model)