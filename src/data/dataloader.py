import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import pandas as pd

class Tumor_Samples(Dataset):

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