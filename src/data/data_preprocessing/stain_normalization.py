from histomicstk.preprocessing.color_normalization import reinhard
import numpy as np
import time
from PIL import Image
import cv2
# import staintools
import cv2
import random
from matplotlib import pyplot as plt
import os
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration
import argparse
from glob import glob
import multiprocessing

parser = argparse.ArgumentParser(description='Script for Stain Normalization')
parser.add_argument("src",help = 'Source to run the Script on - Can be a single folder or directory with muliple folders')
parser.add_argument('dst',help= 'Destination for saving the normalized images')
parser.add_argument('--type',default='mean_and_std',help = 'Due Stain Norm using mean_std or using a reference image')
parser.add_argument('--standard',help = 'Image Path for which to standardize images with')
args = parser.parse_args()
config = vars(args)


# StainTools Installation is Facing some issues on the Server.
# If staintools installed And It's Use is required. Uncomment 'import staintools'

# norm = {
#     'mean': np.array([0.8388, 0.6859,  0.8174]),
#     'sd': np.array([0.0755, 0.1486, 0.0631]),
# }

# tcga_h_and_e std and mean
tcga_norm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

def Reinhard_Using_mean_and_sd(target_folder,norm,image_path):

    """
        Function will normalize the image and store it in the target_folder
        Inputs:
            target_folder: Folder To Store the Normalized Images
            norm: dictionary with std and mean
            image_path: Path to Image to be normalized
        Ouputs:
    
    """

    tissue_rgb = np.array(Image.open(image_path))
    image =  reinhard(tissue_rgb, target_mu=norm['mu'], target_sigma=norm['sigma'])
    final_path = os.path.join(target_folder,image_path.split('/')[-1])
    cv2.imwrite(final_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def Reinhard(target_image, sample_image):
    """
        Inputs:
            target_image: Path of the Target Image.
            sample_image: The Path of the Image to be normalized

        Outputs:
            Normalized Image Array
    
    """

    target_image = staintools.read_image(target_image)
    sample_image = staintools.read_image(sample_image)
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target_image)
    return normalizer.transform(sample_image)


def Vahadne(target_image, sample_image):
    """
    Inputs:
        target_image: Path of the Target Image.
        sample_image: The Path of the Image to be normalized

    Outputs:
        Normalized Image Array

    """
    target_image = staintools.read_image(target_image)
    sample_image = staintools.read_image(sample_image)
    normalizer = staintools.StainNormalizer(method = "vahadane")
    normalizer.fit(target_image)
    return normalizer.transform(sample_image)


def Macenko(target_image, sample_image):

    """
    Inputs:
        target_image: Path of the Target Image.
        sample_image: The Path of the Image to be normalized

    Outputs:
        Normalized Image Array
    
    """
    target_image = staintools.read_image(target_image)
    sample_image = staintools.read_image(sample_image)
    normalizer = staintools.StainNormalizer(method = "macenko")
    normalizer.fit(target_image)
    return normalizer.transform(sample_image)

def Normalize(method, target_image, sample_image):

    """
        Inputs:
            method: Which Method to use for Normalization
            target_image: Path of the Target Image.
            sample_image: The Path of the Image to be normalized

        Outputs:
            Normalized Image
    """
    result = ""
    if(method == "Macenko"):
        result = Macenko(target_image,sample_image)
    elif (method == "Vahadne"):
        result = Vahadne(target_image,sample_image)
    else:
        result = Reinhard(target_image,sample_image)
    
    return result

def RandomNormalize(target_image,sample_image,target_folder):

    """
        The Function will Use a Random Normalization Method from Macenko, Vahadne And Reinhard.
        The Normalized Image will be saved in the target Folder
    """
    number = random.random()
    image = ''
    if(number < 0.333):
        image =  Normalize("Vahadne",target_image,sample_image)
    elif(number > 0.666):
        image =  Normalize("Macenko",target_image,sample_image)
    else: image = Normalize("Reinhard",target_image,sample_image)

    cv2.imwrite(os.path.join(target_folder,sample_image.split('/')[-1]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def normalize_dataset(type_norm,dataset_path,target_path,target_image,norm):

    """
        Inputs: 
            type_norm : Type of Normalization to Use
            dataset_path: Path to the Directory with Image Tiles
            target_path: Path to Directory to store normalized images
            target_image: Path to the target image (The image to normalize dataset to)
            norm: Dict with mean and std to normalize dataset to
    
    """
    processes = []
    files = glob(dataset_path+'/**/*.png',recursive = True)

    if(type_norm == 'mean_and_std'):

        for file in files:
            print(file)

            p = multiprocessing.Process(target = Reinhard_Using_mean_and_sd,args = (target_path,norm,file))
            processes.append(p)
            p.start()
        
        for process in processes:
            process.join()

    else:
        for file in files:

            p = multiprocessing.Process(target = RandomNormalize,args = (target_image,file,target_path))
            processes.append(p)
            p.start()
        
        for process in processes:
            process.join()


if __name__ == '__main__':


    folder_paths = []
    with os.scandir(config['src']) as folder_list:
        for folder in folder_list:
            if(folder.is_dir()):
                folder_paths.append(folder.path)

    
    src_path = config['src']
    dst_path = config['dst']
    type_norm = config['type']
    target_image = config['standard']

    for folder in folder_paths:
        normalize_dataset(type_norm,folder,dst_path,target_image,tcga_norm)