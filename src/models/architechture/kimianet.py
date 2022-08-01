
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import pytorch_lightning as pl

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

class kimianet():
        
    def __init__(self,kimianet_weights,batch_size,learning_rate,sampler,dataset):

        super().__init__()

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