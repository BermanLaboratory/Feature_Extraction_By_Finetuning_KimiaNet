import numpy as np
import os
from glob import glob
from img2vec_pytorch import Img2Vec
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='Script to extract features using pretrained densenet on ImageNet -> Generates a json file for each wsi with features of all patches from a wsi')
parser.add_argument("src",help = 'Dataset Source')
parser.add_argument('dst',help='Destination Folder to store the Extracted Features')
args = parser.parse_args()
config = vars(args)


def image_feature_extractor(path_tiled,img2vec,dst_path):

    """
        The Function Will Extract the Features of Each Patch in the folder and Save the Features to a single json file corresponding
        to Each WSI
        Inputs: 
            path_tiled: Path To the Directory Containing Image Patches. 
            img2vec: The img2vec Instance(object) for Feature Extraction
            dst_path: Path to Store The Extracted Features.
        Outputs:
    """
    

    patches = []
    patches = glob(path_tiled+'/**/*.png',recursive = True) #Extract the Paths of Patches in the folder using glob
    print("Number of Patches For the Image are : {}".format(len(patches)))

    feature_dictionary = {}
    for patch in patches:
        patch_pil = Image.open(patch)
        feature_dictionary[patch] = (img2vec.get_vec(patch_pil, tensor=False)).tolist()
    
    print("Feature Extraction for Image {} Done! ".format(path_tiled))

    
    file_name = path_tiled.split('/')[-1] + '_feature_vectors_densenet.json'

    # Save The Final Dictionary As Json file
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

    img2vec = Img2Vec(cuda=True,model='densenet') # The Model Used Can be Changed.
    # Further documentation for img2vec can be found at https://github.com/christiansafka/img2vec

    for folder in folders: #Extract Features for Each WSI
        image_feature_extractor(folder,img2vec,dst_path)