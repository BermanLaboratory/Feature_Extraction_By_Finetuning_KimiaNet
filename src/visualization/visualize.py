from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.architechture.model_interface import model_interface
from PIL import Image
from torchvision import transforms
import numpy as np
from data.dataloader import Tiles_Selected, dataset_labels
from torch.utils.data.sampler import SubsetRandomSampler
from data.tiles_dataset import *
import json
train_dir = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'
test_dir = './test' 
labels_dict = dataset_labels('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv')
print(labels_dict[342])
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
with open("/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_180.json", 'r') as f:
    selected = json.load(f)
dataset = Tiles_Selected_Selected(train_dir,data_transforms['train'], labels_dict,selected)

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

image = Image.open('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled/Sample 226.vsi - 20x/Sample 226.vsi - 20x [x=8000,y=30000,w=1000,h=1000].png').convert('RGB')
args = ['/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/models/KimiaNetPyTorchWeights.pth',4,0.0001,sampler,dataset]
weights_file = '/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/models/KimiaNetPyTorchWeights.pth'

model = model_interface.load_from_checkpoint("/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/lightning_logs/version_6/checkpoints/epoch=3-step=45000.ckpt",kimianet_weights=weights_file,batch_size=4,learning_rate=0.001,sampler=sampler,dataset=dataset)
image = data_transforms['val'](image)
model.eval()
# target_layers = [model.model.features[1]]
print(model.model.model[0].norm5)
target_layers = [model.model.model[0].norm5]
image = image.unsqueeze(0)
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)


targets = [ClassifierOutputTarget(0)]
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grad_cam = cam(input_tensor=image)

grayscale_cam = grad_cam[0, :]
visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
im_save = Image.fromarray(visualization)
im_save.save('test.png')