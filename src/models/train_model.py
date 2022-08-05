#----> Locally Created utils and packages import
from utils.utils import *
from architechture.kimianet_modified  import kimianet_modified
# from data.dataloader import Tumor_Samples
from data.dataloader_csv import *

#----> Helper Libraries
import json


#----> pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


#----> torchinfo to find structure of model
from torchinfo import summary



'''Todo:
1. Create A Common configuration file using yaml
	File Created -> Test the File
	Make parameters updatable using command line
2. Create A Dataset Interface And transfer Dataset Configuration and loaders to that file
3. Pytorch Profiler
4. Environment Requirements File
5. Create Utils files
	Utils files created -> Shift functions to utils folder
'''



def main(cfg):

	random_seed = cfg.General.seed

	#----> load loggers
	wandb_logger = WandbLogger(name='Adam-16-0.0001-200_images-0.1-random_seed(66)-balanced',project='pytorchlightning')
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.General.log_path)
	labels_dict = dataset_labels(cfg.Data.label_dir)

	#----> Dataset And Interface Intitialization

	data_transforms = data_transforms_dict()
	with open(cfg.Data.selected_patches_json, 'r') as f:
		selected = json.load(f)

	dataset = Tumor_Samples_Selected(cfg.Data.data_dir,data_transforms['train'], labels_dict,selected)
	patch_labels_list = patch_labels(cfg.Data.selected_patches_json,cfg.Data.label_dir)
	indices = list(range(len(dataset)))

	# Setting Up data Samplers
	sampler = data_sampler_dict(cfg.Data.split_type,indices,random_seed,len(dataset),patch_labels_list,cfg.Data.validation_split,cfg.Data.data_shuffle)

	#----> Model Initialization

	model = kimianet_modified(cfg.Model.pretrained_weights,cfg.train_dataloader.batch_size,cfg.Optimizer.lr,sampler,dataset)

	# summary(model.model, input_size=(4, 3,1000,1000))

	#-----> Model Checkpoint
	
	# callback_checkpoint = ModelCheckpoint(
	# 	monitor = 'val_loss',
	# 	dir_path=model_weights_path,
	# 	filename='feature_extraction-{epoch:02d}-{val_loss:0.4f}',
	# 	save_top_k = 1,
	# 	mode = 'min,',
	# 	save_weights_only = True,
	# 	verbose = True
	# 	)

	#----> Eary Stopping Callback

	# early_stop_callback = EarlyStopping(
	# 	monitor = 'val_loss',
	# 	patience = ,
	# 	min_delta = 0.00,
	# 	verbose = True,
	# 	mode = 'min'
		
	# )
	# # checkinpoint only when training
	# Mycallbacks = [callback_checkpoint,early_stop_callback]


	#----> Instantiate Trainer

	trainer = pl.Trainer(
		accelerator = "gpu",
		devices = cfg.General.gpus,
		max_epochs = cfg.General.epochs,
		check_val_every_n_epoch = 1,
		logger = [tb_logger,wandb_logger],
		# accumulate_grad_batches=2
		# callbacks = Mycallbacks
	)

	trainer.fit(model)

	# trainer.tune(model)
	# trainer.validate(model,ckpt_path='/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/version_6/checkpoints/epoch=3-step=45000.ckpt')
	# trainer.fit(model)


if __name__ == '__main__':

	config_file_path = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/config/bermanlab.yaml'
	cfg = read_yaml(config_file_path)

	main(cfg)

"""
	Load from Best Checkpoint And save models to load in any environment

"""

# script = model.to_torchscript()
# torch.jit.save(script,'Model.pt')


