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
from architechture.model_interface import model_interface
import json

from torch.utils.data.sampler import SubsetRandomSampler
from data.data_interface import *
from utils.utils import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

parser = argparse.ArgumentParser(description='Script for Feature Extraction From A Trained Model')
parser.add_argument("save_add",help = 'Path of Directory where to store the extracted features. Add '/' at the end)
parser.add_argument('model_weights',help= 'Path of the checkpoint file containing model weights)
parser.add_argument('config',help = 'Path to the config file')
parser.add_argument('selected',help = 'Path to csv file that contains paths of images whose features are to be extracted')
args = parser.parse_args()
config = vars(args)


class Tiles_Selected_CSV(Dataset):

    def __init__(self,data_path,transform,labels_dict,selected_patches):
        """
        Args: 
            data_path : path to input data
            transform : transformation function
        """
        self.data_path = data_path
        self.image_patches = selected_patches
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


def extract_features(model):

    feature_dict = {}   
    # count = 0
    for ii, (inputs, img_name) in enumerate(dataloader):
        inputs = inputs.to(device)
        output1, output_2 = model(inputs)
        # count = count +1
        output_features = output1.cpu().detach().numpy()
        
        for j in range(len(outputs)):
            feature_dict[img_name[j]] = output_features[j]
        # print(len(feature_dict))
    save_file = open(save_address+'FineTuned_Model_Features_dict.pickle','wb')
    pickle.dump(feature_dict, save_file)
    save_file.close() 

    print("Feature Extraction Is Complete")




if __name__ == '__main__':

    # config_file_path = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/config/bermanlab.yaml'
   
    save_address = config['save_add']
    checkpoint_path = config['model_weights']
    # save_address = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/'
    config_file_path = config['config']
    selected_csv = config['selected']


    cfg = read_yaml(config_file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = cfg.Data.data_dir
    labels_dict = dataset_labels(cfg.Data.label_dir)

    with open(selected_csv, 'r') as f:
        selected = json.load(f)
    dataset = Tiles_Selected_CSV(train_dir,data_transform, labels_dict,selected)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers = 40)

    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # checkpoint_path = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/models/feature_extraction-epoch=08-val_loss=0.8935.ckpt'
    model = model_interface.load_from_checkpoint(checkpoint_path,kimianet_weights = cfg.Model.pretrained_weights,num_classes = cfg.Model.n_classes,learning_rate = cfg.Optimizer.lr)

    model = model.to(device)
    model.eval()

    for param in model.parameters(): 
        param.requires_grad = False

    extract_features(model)
