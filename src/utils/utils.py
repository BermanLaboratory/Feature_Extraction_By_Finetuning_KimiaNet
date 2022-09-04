import json
import yaml
import numpy as np
from addict import Dict
from torchvision import transforms
from sklearn.model_selection import train_test_split
#----> pytorch
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from glob import glob


def dataset_labels(csv_file_path):

    """
    Inputs:
        csv_file_path: Path to the Data CSV File

    Ouputs:
        Returns the dictionary with sample ids and Grades
    """

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

    """
        Inputs:
            selected_json_file_path: Path to Selected Patches
            data_csv_file_path: Path to the Data lables CSV File

        Ouputs:
            Returns the Patch wise labels list
    """

    with open(selected_json_file_path, 'r') as f:
        selected = json.load(f)
    
    labels_dict = dataset_labels(data_csv_file_path)

    labels = []
    for select in selected:

        image_name = select.split('/')[-2]
        labels = labels + [labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]]


    return labels

def data_transforms_dict():

    """
        Returns the data transforms dictionary that are required to be 
        applied to train, test and val dataset splits
    """

    data_transforms_dict = {
	'train': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
		# transforms.Lambda(stain_normalization),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		
	]),
	'val': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),

    'test': transforms.Compose([
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

    """
        Inputs:
            seed: Seed to produce random numbers from
            indices: The List of Indices of dataset to split
            dataset_size: The Size of dataset
            validation_split: val split ratio
            shuffle_dataset: To Shuffle datset or not
        Returns:
            Indices split based on the ratio (The split is random)
                train_indices,val_indices
    
    """

    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices,val_indices

def data_split_balanced(seed,indices,labels_list,validation_split = 0.1):

    """
        Inputs:
            seed: seed to generate random numbers.
            indices: The List of Indices of dataset to split
            labels_list: List of labels for each image patch
            validation_slit: split ratio required.

        Returns: 
            Indices split based on the ratio(The Dataset Split is balanced)
    """

    train_indices, val_indices, train_labels, val_labels = train_test_split(
                                            indices,
                                            labels_list,
                                            stratify=labels_list,
                                            test_size=validation_split,
                                            random_state=seed
                                            )

    return train_indices,val_indices,train_labels,val_labels

def data_sampler_dict(split_type,indices,random_seed,len_dataset,patch_labels_list,train_split = 0.8 ,validation_split=0.1,test_split = 0.1,data_shuffle = True):
    
    """
        Inputs:
            split_type: can be random or balanced.
            indices: Indices of the dataset
            patch_labels_list: list of labels of each image patch

        Returns:
            train,val and test sampler dictionary is returned based on the type and ratio of split chosen.

    """

    ratio_remaining = 1.0 - validation_split
    ratio_test_adjusted = test_split / ratio_remaining


    if split_type == 'random':
        train_indices_remaining,val_indices = data_split_random(random_seed,indices,len_dataset,validation_split,data_shuffle)
        train_indices,test_indices = data_split_random(random_seed,train_indices_remaining,len(train_indices_remaining),ratio_test_adjusted,data_shuffle)
        
    else:
        train_indices_remaining,val_indices,train_labels_remaining,val_labels = data_split_balanced(random_seed,indices,patch_labels_list,validation_split)
        train_indices,test_indices,_,_ = data_split_balanced(random_seed,train_indices_remaining,train_labels_remaining,ratio_test_adjusted)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    

    sampler = {'train':train_sampler,'val':valid_sampler,'test':test_sampler}


    return sampler



def freeze_dense_blocks(block_list,model):
    """
        Inputs:
            block_list: list of dense blocks to freeze while training
            model: Model whole dense blocks need to be frozen
    """

    if(1 in block_list):
        for param in model.model.model[0].denseblock1.parameters():
            param.requires_grad = False
	
    if(2 in block_list):
        for param in model.model.model[0].denseblock3.parameters():
            param.requires_grad = False

    if(3 in block_list):

        for param in model.model.model[0].denseblock3.parameters():
            param.requires_grad = False

    if(4 in block_list):

        for param in model.model.model[0].denseblock4.parameters():
            param.requires_grad = False


