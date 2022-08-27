import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def model_initializer(kimianet_weights,num_classes):

    model = torchvision.models.densenet121(pretrained=True) #Loading Pretrained DenseNet (Trained on ImageNet Dataset)
    model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
    num_features = model.classifier.in_features
    model = fully_connected(model.features,num_features,num_classes)


    # Loading the kimianet weights.
    # The Fully connected layers have been changed so all the kimianet weights cannot be loaded.
    # The DataParallel Layer has also been removed becuase only one GPU is being used for training.
    model_dict = model.state_dict()
    pretrained_dict = torch.load(kimianet_weights,map_location=torch.device('cpu'))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    return model

class fully_connected(nn.Module):

        def __init__(self,model,num_features,num_classes):

            super(fully_connected,self).__init__()
            self.model = model
            self.fc_4 = nn.Linear(num_features,num_classes)
        
        def forward(self,x):

            x = self.model(x)
            x = torch.flatten(x,1)
            out_1 = x
            out_2 = self.fc_4(x)
            return out_1,out_2

class model_interface(pl.LightningModule):

    def __init__(self,kimianet_weights,learning_rate,num_classes=2):

        super().__init__()

    
        self.learning_rate = learning_rate 
        self.train_accuracy = torchmetrics.Accuracy() 
        self.val_accuracy = torchmetrics.Accuracy()

        self.criterion_train = nn.CrossEntropyLoss()
        self.criterion_val = nn.CrossEntropyLoss()

        self.model = model_initializer(kimianet_weights,num_classes)

    
    def forward(self,inputs):
        output_1,output_2 = self.model(inputs)
        return output_1,output_2

    def training_step(self,batch,batch_idx):

        data,label = batch
        output_1,output_2 = self.forward(data)

        loss = self.criterion_train(output_2,label)
        self.train_accuracy(output_2,label)
        self.log('train_loss',loss,on_step = True,on_epoch = True)
        self.log('train_acc',self.train_accuracy,on_step = True,on_epoch = True,prog_bar = True)

        return {'loss':loss,'log':self.log}
      

    def validation_step(self,batch,batch_index):

        val_data,val_label = batch
        temp,val_output = self.forward(val_data)

        val_loss = self.criterion_val(val_output,val_label)
        self.val_accuracy(val_output,val_label)
        self.log('val_acc', self.val_accuracy,on_step = True,on_epoch = True,prog_bar = True)
        self.log('val_loss', val_loss,on_step = True,on_epoch = True)


    def configure_optimizers(self):
        
        #Type of optimizer used is Adam . Can be Changed as per requirements.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    