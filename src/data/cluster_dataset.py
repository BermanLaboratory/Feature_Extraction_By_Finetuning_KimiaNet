# for loading/processing the images  
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import json
import matplotlib.backends.backend_pdf
import shutil
from glob import glob
import multiprocessing
import time

data_path = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)


folders = []
with os.scandir(data_path) as folder_list:
    for folder in folder_list:
        if(folder.is_dir()):
            folders = folders + [folder.path]

def image_feature_extractor(path_tiled,model):

    patches = []
    patches = glob(path_tiled+'/**/*.png',recursive = True)
    print("Number of Patches For the Image are : {}".format(len(patches)))
    feature_dictionary = {}

    i = 0
    for patch in patches:
        try:
            image = load_img(patch,target_size=(224,224))
            image = np.array(image)
            reshaped_image = image.reshape(1,224,224,3)
            image_processed = preprocess_input(reshaped_image)
            feature_vector = model.predict(image_processed, use_multiprocessing = True,workers = 200)
            feature_dictionary[patch] = feature_vector[0].tolist()
            # print(patch)
        except:
            print(patch)
            continue

        if(i % 100 == 0):
            print("Features Extrated of {} Patches".format(i))
        i = i + 1
    print("Feature Extraction for Image {} Done! ".format(path_tiled))

    file_name = path_tiled.split('/')[-1] + '_feature_vector_vgg16.json'
    with open(os.path.join('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/VGG_features',file_name),"w") as file:
        file.write(json.dumps(feature_dictionary)) 


# processes = []
# # manager = multiprocessing.Manager()
# # return_dict = manager.dict()

# for folder in folders:

#     # feature_analyzer(i)
#     p = multiprocessing.Process(target = image_feature_extractor,args=(folder,model))
#     processes.append(p)
#     p.start()

# for process in processes:
#     process.join()

for folder in folders:
    image_feature_extractor(folder,model)



