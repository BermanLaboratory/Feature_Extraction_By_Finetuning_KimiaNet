import json
import yaml
import numpy as np
from addict import Dict
from data.dataloader import dataset_labels
from torchvision import transforms
from sklearn.model_selection import train_test_split
#----> pytorch
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from glob import glob


def dataset_labels(csv_file_path):

    labels_df = pd.read_csv(csv_file_path)
    labels_df = labels_df.dropna()
    labels_df.astype(int)
    labels_dict = {}
    files_list = labels_df['Sample ID'].to_list()
    grade = labels_df['Sample Grade'].to_list()

    for i in range(len(files_list)):
        labels_dict[int(files_list[i])] = int(grade[i])
    
    return labels_dict

def patch_labels(selected_json_file_path,data_csv_file_path):

    with open(selected_json_file_path, 'r') as f:
        selected = json.load(f)
    
    labels_dict = dataset_labels(data_csv_file_path)

    labels = []

    for select in selected:

        image_name = select.split('/')[-2]
        labels = labels + [labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]]


    return labels

def data_transforms_dict():

    data_transforms_dict = {
	'train': transforms.Compose([
        # transforms.Resize(1000),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# transforms.Lambda(stain_normalization),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
	]),
	'val': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),

    'val': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
    }

    return data_transforms_dict


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

def data_split_random(seed,indices,dataset_size,validation_split = 0.1,shuffle_dataset = False):

    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices,val_indices

def data_split_balanced(seed,indices,labels_list,validation_split = 0.1):

    train_indices, val_indices, _, _ = train_test_split(
    indices,
    labels_list,
    stratify=labels_list,
    test_size=validation_split,
    random_state=seed
    )

    return train_indices,val_indices

def data_sampler_dict(split_type,indices,random_seed,len_dataset,patch_labels_list,validation_split=0.1,data_shuffle = False):


    if split_type == 'random':
        train_indices,val_indices = data_split_random(random_seed,indices,len_dataset,validation_split,data_shuffle)
    else:
        train_indices,val_indices = data_split_balanced(random_seed,indices,patch_labels_list,validation_split)


    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    sampler = {'train':train_sampler,'val':valid_sampler}

    return sampler


def selected_patches(selected_csv_folder):

    '''
        input: selected_csv_folder -> folder that contains csv files with data of patches from each whole slide image
        ouput: Sorted And top 500 selected patches from each whole slide image.
    '''

    csv_files = glob(selected_csv_folder+'/*')
    selected = []
    for file in csv_files:

        nuclei_ratio = pd.read_csv(file)
        nuclei_ratio = nuclei_ratio.sort_values(by = 'Nuclei Ratio',ascending = False)
        nuclei_ratio = nuclei_ratio.head(500)
        selected_patches = nuclei_ratio['Patch'].to_list()
        selected = selected + selected_patches
    
    return selected
