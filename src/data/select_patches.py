from glob import glob
import pandas as pd
import json

def selected_patches(selected_csv_folder):

    csv_files = glob(selected_csv_folder+'/*')
    selected = []
    for file in csv_files:

        nuclei_ratio = pd.read_csv(file)
        nuclei_ratio = nuclei_ratio.sort_values(by = 'Nuclei Ratio',ascending = False)
        nuclei_ratio = nuclei_ratio.head(1000)
        selected_patches = nuclei_ratio['Patch'].to_list()
        selected = selected + selected_patches
    
    return selected

image_patches = glob('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Images_Tiled'+'/**/*.png',recursive = True)
selected = selected_patches('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Nuclei_Ratio_180')
image_patches = [patch for patch in image_patches if patch.split('/')[-1] in selected]
with open("/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_1000_180_with_new.json", 'w') as f:
    json.dump(image_patches, f, indent=2) 
