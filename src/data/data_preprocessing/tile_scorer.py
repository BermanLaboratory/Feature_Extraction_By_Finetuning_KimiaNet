import histomicstk as htk
import skimage.io
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
from glob import glob
import multiprocessing

#----> Histolab import
from histolab.tile import Tile
from histolab.scorer import NucleiScorer

import argparse

parser = argparse.ArgumentParser(description='Script to Find The Nuclei Ratio Of Each Image Patch -> Generates A Csv file with Ratios of all patches of a whole slide image')
parser.add_argument("src",help = 'Source to run the Script on - Can be a single folder or directory with muliple folders')
parser.add_argument('dst',help='Destination Folder to store the Calculated Scores CSV Files')
parser.add_argument('--type',default = 'nuclei_and_tissue',help = 'To Select if to use just the nuceli ratio or nuceli ratio + tissue ratio' )
args = parser.parse_args()
config = vars(args)

def nuclei_ratio_for_patch(path_of_patch):

    """
        Inputs: path_of_patch: Patch to calculate nuclei_ratio for
        Ouputs: nuclei_ratio
    """
    
    input_image = skimage.io.imread(path_of_patch)[:,:,:3]
    
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    stains = ['hematoxylin','eosin','null']
    W_matrix = np.array([stain_color_map[stain] for stain in stains]).T
    
    stain_deconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(input_image,W_matrix)
    hematoxylin_channel = stain_deconvolved.Stains[:,:,0] # Only H channel is used for Nuclei Ratio Calculation
    (threshold,black_white_image) = cv2.threshold(hematoxylin_channel,180,255,cv2.THRESH_BINARY)

    
    #COUNTING THE NUMBER OF BLACK PIXELS IN THE FINAL IMAGE FOR NUCLEI APPROXIMATION
    number_of_white_pixels = np.sum(black_white_image == 255)
    number_of_black_pixels = np.sum(black_white_image == 0)
    
    return (number_of_black_pixels/(number_of_black_pixels+number_of_white_pixels))


def nuclei_ratio_for_wsi(wsi_path,dst_path):

    """
        The Nuclei Ratio will be stored in a CSV file in the dst_path

        Inputs:
            wsi_path: Path To Directory That contains patches of one WSI
            dst_path: Path to directory to store the calculated scores
    """
    
    patches = []
    name = wsi_path.split('/')[-1]
    patches = glob(wsi_path+'/**/*.png',recursive = True)
    
    print('Nuclei Ratio Calculaton Started for : {}'.format(name))
    print('Number of Patches in {} are : {}'.format(name,len(patches)))

    nuclei_percent = {}

    for patch in patches:

        try :

            patch_nuclei_percent = nuclei_ratio_for_patch(patch)
            nuclei_percent[patch] = patch_nuclei_percent

        except:
            print('Error With Patch : {}'.format(patch))
            continue

    data_frame = pd.DataFrame(patches,columns = ['Patch'])
    data_frame['Nuclei Ratio'] = list(nuclei_percent.values())

    temp = name
    name = name + '.csv'
    data_frame.to_csv(os.path.join(dst_path,name))

    print('Nuclei Ratio Calculation Completed For : {}'.format(temp))



def nuclei_ratio_for_dataset(dataset_path,dst_path):

    """
        This Function Runs nuclei_ratio_for_wsi for each WSI
        Simultaneously Runs for Multiple WSI for faster processing.
    """
    
    folders = []
    with os.scandir(dataset_path) as folder_list:
        for folder in folder_list:
            if(folder.is_dir()):
                folders = folders + [folder.path]


    processes = []

    for folder in folders:
    
        p = multiprocessing.Process(target = nuclei_ratio_for_wsi,args=(folder,dst_path))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
        

def nuclei_ratio_for_patch_histolab(patch_path,scorer):

    """
        Using Histolab's Implementation for Patch Score Calculation.
        Inputs:
            patch_path: Path of The Patch to calculate score for
            scorer: Histolab Scorer Object for score calculation

        Ouputs:
            Score for each patch
    """

    Tile_object = Tile(Image.open(patch_path).convert('RGB'),[0,0])
    return scorer(Tile_object)

def nuclei_ratio_for_wsi_histolab(wsi_path,dst_path):

    """
        Using Histolab's Patch Score Implementation for calculation of score for each Pathc in a WSI
    """

    patches = glob(wsi_path+'/**/*.png',recursive = True)
    scorer = NucleiScorer()
    name = wsi_path.split('/')[-1]
    nuclei_percent = {}
    print('Nuclei Ratio Calculaton Started for : {}'.format(name))

    for patch in patches:

        patch_nuclei_percent = nuclei_ratio_for_patch_histolab(patch,scorer)
        nuclei_percent[patch] = patch_nuclei_percent
   
    data_frame = pd.DataFrame(patches,columns = ['Patch'])
    data_frame['Nuclei Ratio'] = list(nuclei_percent.values())

    
    temp = name
    name = name + '.csv'
    data_frame.to_csv(os.path.join(dst_path,name))

    print('Nuclei Ratio Calculation Completed For : {}'.format(temp))



def nuclei_ratio_for_dataset_histolab(dataset_path,dst_path):

    """
        Running The Histolab Patch Scorer for complete Dataset.
        The Calculation for Each WSI is done simultaneously usng python's multiprocessing module
    """

    folders = []
    with os.scandir(dataset_path) as folder_list:
        for folder in folder_list:
            if(folder.is_dir()):
                folders = folders + [folder.path]
    
    processes = []

    for folder in folders:
      
        p = multiprocessing.Process(target = nuclei_ratio_for_wsi_histolab,args=(folder,dst_path))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()


if __name__ == '__main__':

    src_path = config['src']
    dst_path = config['dst']
    scorer_type = config['type']

    if scorer_type == 'nuclei_and_tissue':

        nuclei_ratio_for_dataset_histolab(src_path,dst_path)
    
    else:
        nuclei_ratio_for_dataset(src_path,dst_path)

