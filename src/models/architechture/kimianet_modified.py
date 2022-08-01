import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

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

class kimianet_modified(pl.LightningModule):

    def __init__(self,kimianet_weights,batch_size,learning_rate,sampler,dataset):

        super().__init__()

        #Initialize the model here

        self.batch_size = batch_size
        self.learning_rate = learning_rate #
        self.train_accuracy = torchmetrics.Accuracy() #
        self.val_accuracy = torchmetrics.Accuracy()
        self.sampler = sampler
        self.dataset = dataset

        self.model = torchvision.models.densenet121(pretrained=True)
        self.model.features = nn.Sequential(self.model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
        num_features = self.model.classifier.in_features
        self.model = fully_connected(self.model.features,num_features,2)
        self.criterion_train = nn.CrossEntropyLoss()
        self.criterion_val = nn.CrossEntropyLoss()
        
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(kimianet_weights,map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 

        self.model.load_state_dict(model_dict)


    
    def forward(self,inputs):
        output_1,output_2 = self.model(inputs)
        return output_1,output_2

    def training_step(self,batch,batch_idx):

        data,label = batch
        output_1,output_2 = self.forward(data)

        loss = self.criterion_train(output_2,label)
        self.train_accuracy(output_2,label)
        # self.log('train_loss',loss,on_step = True,on_epoch = True)
        # self.log('train_acc',self.train_accuracy,on_step = True,on_epoch = True,prog_bar = True)

        # return {'loss':loss,'log':self.log}
        return{'loss':loss}

    # def training_epoch_end(self,outs):

        # self.log('train_acc_epoch',self.accuracy.compute())
        # print(self.accuracy.compute())

    def validation_step(self,batch,batch_index):

        val_data,val_label = batch
        temp,val_output = self.forward(val_data)

        val_loss = self.criterion_val(val_output,val_label)
        self.val_accuracy(val_output,val_label)
        # self.log('val_acc', self.val_accuracy,on_step = True,on_epoch = True,prog_bar = True)
        # self.log('val_loss', val_loss,on_step = True,on_epoch = True)

    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('val_acc_epoch', self.accuracy.compute())
    #     print(self.accuracy.compute())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                           sampler=self.sampler['train'],num_workers = 40)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                sampler=self.sampler['val'],num_workers = 40)


    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=(self.learning_rate))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    