import numpy as np
import os
from glob import glob
from img2vec_pytorch import Img2Vec
from PIL import Image
import json

data_path = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'

folders = []
with os.scandir(data_path) as folder_list:
    for folder in folder_list:
        if(folder.is_dir()):
            folders = folders + [folder.path]

img2vec = Img2Vec(cuda=True,model='densenet')


def image_feature_extractor(path_tiled):

    patches = []
    patches = glob(path_tiled+'/**/*.png',recursive = True)
    print("Number of Patches For the Image are : {}".format(len(patches)))

    feature_dictionary = {}
    for patch in patches:
        patch_pil = Image.open(patch)
        feature_dictionary[patch] = (img2vec.get_vec(patch_pil, tensor=False)).tolist()
        # print(len(feature_dictionary[patch]))
    
    print("Feature Extraction for Image {} Done! ".format(path_tiled))

    print(len(feature_dictionary))
    file_name = path_tiled.split('/')[-1] + '_feature_vectors_densenet.json'
    with open(os.path.join('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/DenseNet_Features',file_name),"w") as file:
        file.write(json.dumps(feature_dictionary)) 


for folder in folders:
    image_feature_extractor(folder)