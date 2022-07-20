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
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers

train_dir = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'
test_dir = './test'
wandb_logger = WandbLogger(name='Adam-8-0.0001',project='pytorchlightning')

tb_logger = pl_loggers.TensorBoardLogger(save_dir="/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/")
labels_dict = dataset_labels('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv')

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

# dataset = Tumor_Samples(train_dir,data_transforms['train'], labels_dict)

with open("/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_180.json", 'r') as f:
    selected = json.load(f)
dataset = Tumor_Samples_Selected(train_dir,data_transforms['train'], labels_dict,selected)

validation_split = .1
shuffle_dataset = True
random_seed= 42

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
model = kimianet_modified('/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/models/KimiaNetPyTorchWeights.pth',8,0.0001,sampler,dataset)
# summary(model.model, input_size=(4, 3,1000,1000))
trainer = pl.Trainer(
    accelerator = "gpu",
	devices = [2],
    max_epochs = 10,
    progress_bar_refresh_rate = 10,
	check_val_every_n_epoch = 1,
	logger = [wandb_logger,tb_logger],
	default_root_dir="/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/"
)

# trainer.tune(model)
# trainer.validate(model,ckpt_path='/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/version_6/checkpoints/epoch=3-step=45000.ckpt')
trainer.fit(model)
