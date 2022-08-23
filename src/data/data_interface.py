import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.utils import *
from data.tiles_dataset import *


class WSI_Data_Interface(pl.LightningDataModule):

    def __init__(self,cfg):


        """
        
        """
        super().__init__()
        self.cfg = cfg.Data
        self.random_seed = cfg.General.seed

    def setup(self,stage = None):

        labels_dict = dataset_labels(self.cfg.label_dir)

        self.data_transforms = data_transforms_dict()
        with open(self.cfg.selected_patches_json, 'r') as f:
            self.selected = json.load(f)

        self.dataset_train = Tiles_Selected_CSV(self.cfg.data_dir,self.data_transforms['train'], labels_dict,self.selected)
        self.dataset_val = Tiles_Selected_CSV(self.cfg.data_dir,self.data_transforms['val'], labels_dict,self.selected)
        self.dataset_test = Tiles_Selected_CSV(self.cfg.data_dir,self.data_transforms['test'], labels_dict,self.selected)
        # self.dataset_train = Tiles_Selected_Image_Array(self.cfg.data_dir,self.data_transforms['train'], labels_dict,self.selected)
        # self.dataset_val = Tiles_Selected_Image_Array(self.cfg.data_dir,self.data_transforms['val'], labels_dict,self.selected)
        # self.dataset_test = Tiles_Selected_Image_Array(self.cfg.data_dir,self.data_transforms['test'], labels_dict,self.selected)
        patch_labels_list = patch_labels(self.cfg.selected_patches_json,self.cfg.label_dir)
        indices = list(range(len(self.dataset_train)))

        # Setting Up data Samplers
        self.sampler = data_sampler_dict(self.cfg.split_type,indices,self.random_seed,len(self.dataset_train),patch_labels_list,self.cfg.train_slit,self.cfg.validation_split,self.cfg.test_split,self.cfg.data_shuffle)


    def train_dataloader(self):

        return DataLoader(self.dataset_train, batch_size=self.cfg.train_dataloader.batch_size, 
                                           sampler=self.sampler['train'],num_workers = self.cfg.train_dataloader.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.val_dataloader.batch_size,
                                                sampler=self.sampler['val'],num_workers = self.cfg.val_dataloader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,batch_size = self.cfg.test_dataloader.batch_size,sampler = self.sampler['test'],num_workers = self.cfg.test_dataloader.num_workers)


        