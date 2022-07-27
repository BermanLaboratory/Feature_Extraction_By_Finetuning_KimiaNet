from data.dataloader import Tumor_Samples, dataset_labels
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from architechture.kimianet_modified  import kimianet_modified
from glob import glob
from data.dataloader_csv import *
import json

#----> pytorch_lightning
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers


from sklearn.model_selection import train_test_split
from utils.utils import final_labels

'''Todo:
1. Create A Common configuration file using yaml
2. Create A Dataset Interface And transfer Dataset Configuration and loaders to that file
'''


#----> load loggers
wandb_logger = WandbLogger(name='Adam-8-0.0001-200_images-0.1-random_seed(66)-balanced',project='pytorchlightning')
tb_logger = pl_loggers.TensorBoardLogger(save_dir="/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/")
labels_dict = dataset_labels('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv')

#----> Data transforms and configuration

train_dir = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'
selected_image_patches_json = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_180_with_new.json'
data_csv = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_180_with_new.json','/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv'
logging_dir = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/'
pretrained_weights = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/models/KimiaNetPyTorchWeights.pth'
batch_size = 8
learning_rate = 0.0001
validation_split = .1
shuffle_dataset = True
random_seed= 66

with open(selected_image_patches_json, 'r') as f:
    selected = json.load(f)


data_transforms = {
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
}

# Dataset Initialization

# dataset = Tumor_Samples(train_dir,data_transforms['train'], labels_dict)
dataset = Tumor_Samples_Selected(train_dir,data_transforms['train'], labels_dict,selected)
final_labels_list = final_labels(data_csv)

dataset_size = len(dataset)
indices = list(range(dataset_size))
# Creating data indices for training and validation splits:

# Random Validation and Training Split

split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

#Testing Dataset Sampling with Equal distribution of classes

train_indices, val_indices, _, _ = train_test_split(
    indices,
    final_labels_list,
    stratify=final_labels_list,
    test_size=validation_split,
    random_state=random_seed
)

# Setting Up data Samplers
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
sampler = {'train':train_sampler,'val':valid_sampler}

#----> Model Initialization

model = kimianet_modified(pretrained_weights,8,0.0001,sampler,dataset)

# summary(model.model, input_size=(4, 3,1000,1000))


#----> Instantiate Trainer

trainer = pl.Trainer(
    accelerator = "gpu",
	devices = [2],
    max_epochs = 10,
    progress_bar_refresh_rate = 10,
	check_val_every_n_epoch = 1,
	logger = [wandb_logger,tb_logger],
	default_root_dir=logging_dir
	# accumulate_grad_batches=2
)

# trainer.tune(model)
# trainer.validate(model,ckpt_path='/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/version_6/checkpoints/epoch=3-step=45000.ckpt')
trainer.fit(model)
