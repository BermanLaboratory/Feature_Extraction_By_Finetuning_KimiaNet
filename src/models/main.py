#----> Locally Created utils and packages import
from data.data_interface import WSI_Data_Interface
from utils.utils import *
from architechture.model_interface  import model_interface
# from data.dataloader import Tiles_Selected
from data.data_interface import *
#----> Helper Libraries
import json
import argparse

#----> pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


#----> torchinfo to find structure of model
from torchinfo import summary



'''Todo:
3. Test the -> test Code.
5. Feature Visualisation Code
6. Feature Importance Code:
	Added The Code -> Make it to work with Command Line
7. Extracting the heatmaps for each feature.
10. Clustering for patch selection
 -> code Added 
13. Mean and Standard deviation of dataset: For Reinhard Normalization
		Added in the Dataloader
		Using Mean and Standard dev already given
14. Add seaborn visualization code here
15. Add Statistics Calculation to the code
16. Learning Rate and Batch Size Optimizer
		Learning Rate added -> Move it to utils file

17. Add docstrings to all the functions in the code
18. Write all the input and output of each file in the code.
19. Add Python  Path addition to the code
20. Description and use of each file with references and citation.
21. Sample outputs add to the github page.
22. Test Runs for all the code.
23. Directory structure required to run the files
23. Update the requirements file -> add that to readme as well
24. Project Directory Readme complete
25. What do you mean by fine tuning
26. Yaml file description
27. why pytorch lightning
28. What is hyperparameter tuning and why?
29. Short QuPath Videos
30. Add capacity for n>2
31. wandb connection check
32. Table for all the files
33. Python multiprocessing module -> info reference
34. Using Screen for running stuff
'''

def parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('config',default = '',type = str)
	args = parser.parse_args()
	return args

def main(cfg):

	random_seed = cfg.General.seed

	#----> load loggers
	wandb_logger = WandbLogger(name=cfg.General.run_name,project=cfg.General.project_name)
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.General.log_path)
	

	#----> Dataset And Interface Intitialization
	
	data_module = WSI_Data_Interface(cfg)

	#----> Model Initialization

	model = model_interface(cfg.Model.pretrained_weights,cfg.Optimizer.lr,cfg.Model.n_classes)

	# summary(model.model, input_size=(4, 3,1000,1000))


	#-----> Selecting DenseBlocks to Train and Freeze

	freeze_dense_blocks(cfg.Model.layers_to_freeze,model)
	
	#-----> Model Checkpoint
	
	callback_checkpoint = ModelCheckpoint(
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
		patience = 5,
		min_delta = 0.00,
		verbose = True,
		mode = 'min'
		
	)

	# # checkinpoint only when training

	Mycallbacks = [callback_checkpoint]


	#----> Instantiate Trainer

	trainer = pl.Trainer(
		accelerator = "gpu",
		devices = cfg.General.gpus,
		max_epochs = cfg.General.epochs,
		check_val_every_n_epoch = 1,
		logger = [tb_logger,wandb_logger],
		accumulate_grad_batches=cfg.General.grad_accumulation,
		callbacks = Mycallbacks
	)


	# lr_finder = trainer.tuner.lr_find(model,datamodule=data_module)
	# # Results can be found in
	# lr_finder.results
	# # Plot with
	# fig = lr_finder.plot(suggest=True)
	# fig.savefig('/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/models/Learning_Rate.png')

	# # Pick point based on plot, or get suggestion
	# new_lr = lr_finder.suggestion()
	# print(new_lr)

	if cfg.General.mode == 'train':
		trainer.fit(model = model,datamodule = data_module)
	else:
		print('Using the {} for testing the model on dataset')
		test_model = model.load_from_checkpoint(checkpoint_path=cfg.General.weights_file_path)
		trainer.test(test_model,datamodule = data_module)
	

if __name__ == '__main__':

	args = parse()
	# config_file_path = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/config/bermanlab.yaml'
	config_file_path = args.config
	cfg = read_yaml(config_file_path)

	main(cfg)







