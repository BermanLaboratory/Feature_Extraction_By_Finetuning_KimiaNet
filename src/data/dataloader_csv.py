import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import pandas as pd

class Tumor_Samples_Selected(Dataset):

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

        return image,label

def selected_patches(selected_csv_folder):

    csv_files = glob(selected_csv_folder+'/*')
    selected = []
    for file in csv_files:

        nuclei_ratio = pd.read_csv(file)
        nuclei_ratio = nuclei_ratio.sort_values(by = 'Nuclei Ratio',ascending = False)
        nuclei_ratio = nuclei_ratio.head(500)
        selected_patches = nuclei_ratio['Patch'].to_list()
        selected = selected + selected_patches
    
    return selected








