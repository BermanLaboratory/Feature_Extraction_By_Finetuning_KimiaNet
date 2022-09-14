#----> Locally Created utils and packages import
from data.data_interface import WSI_Data_Interface
from utils.utils import *
from architechture.model_interface  import model_interface
# from data.dataloader import Tiles_Selected
from data.data_interface import *
#----> Helper Libraries
import json
import argparse
import os

#----> pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


#----> torchinfo to find structure of model
from torchinfo import summary


def parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('config',default = '',type = str)
	args = parser.parse_args()
	return args

def main(cfg):

	# random_seed = cfg.General.seed

	#----> load loggers
	wandb_logger = WandbLogger(name=cfg.General.run_name,project=cfg.General.project_name)
	# tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.General.log_path)
	

	#----> Dataset And Interface Intitialization
	
	data_module = WSI_Data_Interface(cfg)

	#----> Model Initialization

	model = model_interface(cfg.Model.pretrained_weights,cfg.Optimizer.lr,cfg.Model.n_classes)

	# summary(model.model, input_size=(4, 3,1000,1000))


	#-----> Selecting DenseBlocks to Train and Freeze

	freeze_dense_blocks(cfg.Model.layers_to_freeze,model)
	
	#-----> Model Checkpoint
	
	checkpoints_callback = ModelCheckpoint(
		monitor = 'val_loss',
		dirpath=cfg.Model.fine_tuned_weights_dir,
		filename='feature_extraction-{epoch:02d}-{val_loss:0.4f}',
		save_top_k = 1,
		mode = 'min',
		save_weights_only = True,
		verbose = True
		)

	#----> Eary Stopping Callback

	early_stop_callback = EarlyStopping(
		monitor = 'val_loss',
		patience = cfg.General.patience,
		min_delta = 0.00,
		verbose = True,
		mode = 'min'
		
	)


	callbacks = [early_stop_callback]
	if cfg.General.mode == 'train':
		callbacks.append(checkpoints_callback)



	#----> Instantiate Trainer

	trainer = pl.Trainer(
		accelerator = "gpu",
		devices = cfg.General.gpus,
		max_epochs = cfg.General.epochs,
		check_val_every_n_epoch = 1,
		logger = [wandb_logger],
		accumulate_grad_batches=cfg.General.grad_accumulation,
		callbacks = callbacks
	)


	if cfg.General.mode == 'lr_find':
		lr_finder = trainer.tuner.lr_find(model,datamodule=data_module)
		lr_finder.results
		fig = lr_finder.plot(suggest=True)
		fig.savefig(os.path.join(cfg.Optimizer.lr_finder_path,'Learning_Rate.png'))
		new_lr = lr_finder.suggestion()
		print(new_lr)

	elif cfg.General.mode == 'train':
		trainer.fit(model = model,datamodule = data_module)

	elif cfg.General.mode == 'test':
		test_model = model.load_from_checkpoint(checkpoint_path=cfg.General.weights_file_path,kimianet_weights = cfg.Model.pretrained_weights,learning_rate= cfg.Optimizer.lr)
		trainer.test(test_model,datamodule = data_module)
	

if __name__ == '__main__':

	args = parse()
	config_file_path = args.config
	cfg = read_yaml(config_file_path)

	main(cfg)







