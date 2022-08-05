import json
from data.dataloader import dataset_labels
from torchvision import transforms

def final_labels(selected_json_file_path,data_csv_file_path):

    with open(selected_json_file_path, 'r') as f:
        selected = json.load(f)
    
    labels_dict = dataset_labels(data_csv_file_path)

    labels = []

    for select in selected:

        image_name = select.split('/')[-2]
        labels = labels + [labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]]


    return labels

def data_transforms_dict():

    data_transforms_dict = {
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

    'val': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
    }

    return data_transforms_dict



