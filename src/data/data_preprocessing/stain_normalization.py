from histomicstk.preprocessing.color_normalization import reinhard
import numpy as np
import time
from PIL import Image
import cv2
import staintools
import cv2
import random
from matplotlib import pyplot as plt
import os
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
import argparse

norm = {
    'mean': np.array([0.8388, 0.6859,  0.8174]),
    'sd': np.array([0.0755, 0.1486, 0.0631]),
}
cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

tissue_rgb = np.array(Image.open('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled/Sample 128.vsi - 20x/Sample 128.vsi - 20x [x=12000,y=30000,w=1000,h=1000].png').convert('RGB'))
tissue_rgb_normalized = reinhard(
    tissue_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])

# print(type(tissue_rgb_normalized))

# image = Image.fromarray(tissue_rgb_normalized.astype(np.uint8))
cv2.imwrite(file_path, cv2.cvtColor(tissue_rgb_normalized, cv2.COLOR_RGB2BGR))

def Reinhard_Using_mean_and_sd(norm,image_path):

    tissue_rgb = np.array(Image.open(image_path))
    return reinhard(tissue_rgb, target_mu=norm['mu'], target_sigma=norm['sigma']))

def Reinhard(target_image, sample_image):
    target_image = staintools.read_image(target_image)
    sample_image = staintools.read_image(sample_image)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target_image)
    return normalizer.transform(sample_image)


def Vahadne(target_image, sample_image):
    target_image = staintools.read_image(target_image)
    sample_image = staintools.read_image(sample_image)
    normalizer = staintools.StainNormalizer(method = "vahadane")
    normalizer.fit(target_image)
    return normalizer.transform(sample_image)


def Macenko(target_image, sample_image):
    target_image = staintools.read_image(target_image)
    sample_image = staintools.read_image(sample_image)
    normalizer = staintools.StainNormalizer(method = "macenko")
    normalizer.fit(target_image)
    return normalizer.transform(sample_image)

def Normalize(method, target_image, sample_image):

    result = ""
    if(method == "Macenko"):
        result = Macenko(target_image,sample_image)
    elif (method == "Vahadne"):
        result = Vahadne(target_image,sample_image)
    else:
        result = Reinhard(target_image,sample_image)
    
    return result

def RandomNormalize(target_image,sample_image):
    number = random.random()
    if(number < 0.333):
        return Normalize("Vahadne",target_image,sample_image)
    elif(number > 0.666):
        return Normalize("Macenko",target_image,sample_image)
    else: return Normalize("Reinhard",target_image,sample_image)




parser = argparse.ArgumentParser(description='Script to Remove Empty Tiles from the Images Using File Size')
parser.add_argument("src",help = 'Source to run the Script on - Can be a single folder or directory with muliple folders')
parser.add_argument('--file_size',type = int,default = 1,help='The Tiles having file size less than this argument will be removed. (Pass an Integer in mb)')
args = parser.parse_args()
config = vars(args)

if __name__ == '__main__':

    size = config['file_size']

    folder_paths = []
    with os.scandir(config['src']) as folder_list:
        for folder in folder_list:
            if(folder.is_dir()):
                folder_paths.append(folder.path)

    if (len(folder_paths) == 0):

        print('Source Is A Single Folder')

        count = 0
        
        with os.scandir(config['src']) as files:
            for file in files:
                
        folder_name = config['src'].split('/')[-1]
        print('{} Files Removed from folder {}'.format(count,folder_name))

    else:

        print('There are {} folders in the directory to process'.format(len(folder_paths)))
        for folder in folder_paths:
            count = 0
            folder_name = folder.split('/')[-1]
            print('File Removal Started for folder : {}'.format(folder_name))
            with os.scandir(folder) as files:
                for file in files:
                    if (os.path.getsize(file.path)/(1024*1024)) <= size :
                        os.remove(file.path)
                        count = count + 1
                        if count % 100 == 0 :
                            print('         {} files removed'.format(count))
            
            print('{} Files Removed from folder {}'.format(count,folder_name))


    print('ALL FILES HAVING FILE SIZE <= {} mb ARE SUCCESSFULLY REMOVED.'.format(size))