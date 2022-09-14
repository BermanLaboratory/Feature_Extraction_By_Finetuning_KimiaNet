from glob import glob
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='Select The Final Dataset for Training on Basis of Patch Score and the results of patch clustering(artifact removal)')
parser.add_argument("cluster",help = 'Path to csv file that has patches selected after clustering and removing patches with artifacts')
parser.add_argument('patch_score',help = 'Path to the directory that contains the patch score calculated files')
parser.add_argument('dst',help = 'The Path to store the final json file. Path format : directory/file_name.json')
args = parser.parse_args()
config = vars(args)

def selected_patches(selected_csv_folder):

    """
        Selecting Top Patches just on the basis of Patch Score
    """

    csv_files = glob(selected_csv_folder+'/*')
    selected = []
    for file in csv_files:

        nuclei_ratio = pd.read_csv(file)
        nuclei_ratio = nuclei_ratio.sort_values(by = 'Nuclei Ratio',ascending = False)
        nuclei_ratio = nuclei_ratio.head(1000)
        selected_patches = nuclei_ratio['Patch'].to_list()
        selected = selected + selected_patches
    
    return selected


def select_patches_cluster_nuclei_ratio(cluster_selected,nuclei_ratio_folder):

    """
        Inputs: 
            cluster_selected: List of Selected Image Patches
            nuclei_ratio_folder: Path to Directory Which Contain the Nuclei Ratios or Nuclei Plus Tissue Ratio Calculated for Each Image Patch
        Output:
            List That contains the final selected Patches
    """

    csv_files = glob(nuclei_ratio_folder+'/*')
    selected = [] #List for final selected Patches (File Patches)
    select_patches_cluster = pd.read_csv(cluster_selected)
    # i = 0
    for file in csv_files:

        nuclei_ratio = pd.read_csv(file)
        merged_df = pd.merge(select_patches_cluster,nuclei_ratio,on='Patch') #Find the Selected Patches for Each WSI
        merged_df = merged_df.sort_values(by = 'Nuclei Ratio',ascending = False) # Sort The Patches Based on the Score
        final_df = merged_df.head(500)  #Select Top 500 Patches
        selected_patches = final_df['Patch'].to_list()
        selected = selected + selected_patches
        # print(i)
        # i = i+1
    return selected



if __name__ == '__main__':

    select_cluster = config['cluster']
    patch_score = config['patch_score']
    dst = config['dst']

    selected = select_patches_cluster_nuclei_ratio(select_cluster,patch_score)
    with open(dst, 'w') as f:
        json.dump(selected, f, indent=2) 