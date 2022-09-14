from __future__ import print_function, division
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import pickle	
from architechture.model_interface import model_interface
import json

from data.data_interface import *
from utils.utils import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

parser = argparse.ArgumentParser(description='Script for Feature Extraction From A Trained Model')
parser.add_argument("save_add",help = 'Path of Directory where to store the extracted features')
parser.add_argument('model_weights',help= 'Path of the checkpoint file containing model weights')
parser.add_argument('config',help = 'Path to the config file')
parser.add_argument('selected',help = 'Path to json file that contains paths of images whose features are to be extracted')
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
        # patch_name = self.image_patches[index].split('/')[-1]
        image_name = self.image_patches[index].split('/')[-2]
        label = self.labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]
        image = self.transform(image)

        return image,self.image_patches[index]


def extract_features(model):

    feature_dict = {}   
    # count = 0
    print('Starting Feature Extraction')
    for ii, (inputs, img_name) in enumerate(dataloader):
        print(ii)
        inputs = inputs.to(device)
        output1, output_2 = model(inputs)
        # count = count +1
        output_features = output1.cpu().detach().numpy()
        
        for j in range(len(output_features)):
            feature_dict[img_name[j]] = output_features[j]
        # print(len(feature_dict))
    final_save_address = os.path.join(save_address,'FineTuned_Model_Features_dict.pickle')
    save_file = open(final_save_address,'wb')
    pickle.dump(feature_dict, save_file)
    save_file.close() 

    print("Feature Extraction Is Complete")




if __name__ == '__main__':

    
   
    save_address = config['save_add']
    checkpoint_path = config['model_weights']
    config_file_path = config['config']
    selected_csv = config['selected']


    cfg = read_yaml(config_file_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = cfg.Data.data_dir
    labels_dict = dataset_labels(cfg.Data.label_dir)
    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    with open(selected_csv, 'r') as f:
        selected = json.load(f)
    dataset = Tiles_Selected_CSV(train_dir,data_transform, labels_dict,selected)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers = 40)


    model = model_interface.load_from_checkpoint(checkpoint_path,kimianet_weights = cfg.Model.pretrained_weights,num_classes = cfg.Model.n_classes,learning_rate = cfg.Optimizer.lr)

    model = model.to(device)
    model.eval()

    for param in model.parameters(): 
        param.requires_grad = False

    extract_features(model)
