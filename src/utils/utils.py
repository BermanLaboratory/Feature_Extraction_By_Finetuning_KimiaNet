import json
from data.dataloader import dataset_labels

def final_labels(selected_json_file_path,data_csv_file_path):

    with open(selected_json_file_path, 'r') as f:
        selected = json.load(f)
    
    labels_dict = dataset_labels(data_csv_file_path)

    labels = []

    for select in selected:

        image_name = select.split('/')[-2]
        labels = labels + [labels_dict[int(((image_name).split(' ')[1]).split('.')[0])]]


    return labels
