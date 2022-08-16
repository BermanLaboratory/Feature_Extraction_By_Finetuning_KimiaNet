import numpy as np
import os
from glob import glob
from img2vec_pytorch import Img2Vec
from PIL import Image
import json

import argparse

parser = argparse.ArgumentParser(description='Script to extract features from using pretrained densenet on ImageNet -> Generates a json file for each wsi with features of all patches from a wsi')
parser.add_argument("src",help = 'Dataset Source')
parser.add_argument('dst',help='Destination Folder to store the Extracted Features')
args = parser.parse_args()
config = vars(args)




def image_feature_extractor(path_tiled,img2vec,dst_path):

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
    with open(os.path.join(dst_path,file_name),"w") as file:
        file.write(json.dumps(feature_dictionary)) 


if __name__ == '__main__':

    src_path = config['src']
    dst_path = config['dst']

    folders = []
    with os.scandir(src_path) as folder_list:
        for folder in folder_list:
            if(folder.is_dir()):
                folders = folders + [folder.path]

    img2vec = Img2Vec(cuda=True,model='densenet')
    for folder in folders:
        image_feature_extractor(folder,img2vec,dst_path)