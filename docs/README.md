Feature Extractor
==============================

Finetuning DenseNet121 architechture using weights of the model provided by KimiaNet for extracting features From Whole Slide Image Patches relevant to Cancer Grades

## Updates:

## Testing And Evaluation Script

For training the model on custom dataset
```shell

python train --config CONFIG_FILE_PATH --mode train

```

For Testing the model of custom model using pretrained weights:

```shell

python train --config CONFIG_FILE_PATH --mode test
```

Additional flags that can be passed :
* `--config`: configuration file path
* `--mode` : train or test mode
* `--gpus` : gpus to use (default : [2]). Use nvidia-smi to check for free gpus.
* `--batch_size_train` : batch_size for training the model
* `--batch_size_test`
: batch_size for testing the model

## Running External Cohorts on This Code

## Preprocessing Image Dataset (.vsi Images) :

<img src="/docs/WSI_Processing.png"  height = '700px' align="center" />

### Creating Image Tiles/Patches Using QuPath:
1. Open QuPath -> File -> Project -> Create Project -> New Folder -> Open
2. Add Images -> Choose Files -> Select Images to Add to Project (only the .vsi files)
3. Go to search entry in project at the bottom -> type 'overview' -> Right Click on Test_Project -> Remove Images
4. Select Any Image -> Workflow -> Create Script -> File -> Open  ( Select The Script For Tiling ).

[QuPath Script Link](/src/data/data_preprocessing/Tile_Exporter.groovy)
The Script is also Present in the Backup Drive to Run it Directly on the dataset.

Tiling Parameters in the Script That Can be Changed: 
* `downsample`: This parameters helps to select the resolution we want to work with. By default set to 1. Means max resolution of the file that is 20X. The final resoultion of image will be `orginial max resolution / downsample`
* `imageExtension`: '.png' is default . It is a lossless compression format. '.tiff' is good but it uses lot more memory for storage.
* `tileSize`: specify the size of tiles to use. (default: 1000)
* `overlap`: how much do you want tiles to overlap (default: 0)
* `outputPath`: Specify the output path for the tiles to be stored.


The Script Can be Run for Both a Single WSI or the Complete Project:
Right Click on the Script Editor of Qupath: * Choose 'Run' For A single WSI
                                            * Choose 'Run For Project' For Running the Script for the complete project


### Removing Empty Tiles/Patches 

This is done using the file size of the created image patches. The python script will run directly by just using the dataset folder and file size using command line.
Default value for file size for 1000 pixel .png images is : 1 mb.
The File Size can be determined by sorting the patches based on file size using file explorer and then determine the appropriate threshold.

[Remove Empty Tile Script](src/data/data_preprocessing/remove_empty_tiles.py)

Sample Code For running the Script:
```shell
python remove_empty_tiles.py TILED_DATASET_PATH 
```

Optional Arguments:
`--file_size` : File Size Threshold ( Default: 1)

Result of Removing Tiles based on file size:
<img src="/docs/Empty_Tiles_Removal.jpg"  height = '300px' width = '800px' align="center" />

### Clustering of the patches to remove Tiles with artifacts in them
#### 1. Feature Extraction Using Pretrained DenseNet:

```shell
python fast_feature_extraction.py SRC_PATH DST_PATH
```
* `src`: The Path Can be a single folder containing Tiles/Patches or a Directory Containing Multiple folders for Each WSI
* `dst`: The Path Where to Store the Extracted Features.

Running this file will output a json file with the features corresponding to each Patch in a WSI folder.
The Json file will be in the form of a dictionary - Can be read in python using
```python
feature_file = FILE_PATH
with open(feature_file,"r") as file:
        feature_dictionary = json.loads(file.read())
```
[img2vec](https://github.com/christiansafka/img2vec) python library has been used to extract the features from pretrained pytroch models.
The Model Used for feature extraction can also be changed depending on the requirement.

Sample Src And Dst directory structure after running the python script
```bash
SRC_DIRECTORY/
	├── WSI_1
    		├── Patch.png
    		├── Patch_2.png
    		└── ...
DST_DIRECTORY/
   ├──WSI_1_feature_vectors_densenet.json
         
```

#### 2. Clustering of the dataset based on the extracted features

A jupyter notebook has been used for this specific file because visualization and selection of clusters is required to remove the artifacts from the dataset.

For performing PCA and K means Clustering -> META AIs [faiss](https://github.com/facebookresearch/faiss) library has been used. 
It is much more faster and effecient than standard implementations of PCA and K means. As we are dealing with large number of features and images
This is the best to use. Faiss is almost 10X faster and has very low error rate.

[Dataset Clustering Notebook ](/src/data/data_preprocessing/cluster_dataset_densenet.ipynb)

* The Feature Vector Files are loaded. The Feature Space is reduced from 1024 to 500 dimensions using PCA.
* The Images are Clustered Based on those features.
* The Clusters are visualized And Relevant One's are Selected.
* A final csv file containing the selected Patches is saved as output.

Run Each cell in the notebook sequentially. Modify the FILE_PATHS of dataset and feature_vector folders if required.
Using This approach The following artifacts can be easily removed from the dataset:

<img src="/docs/Removing_Unwanted_Tiles.jpg"  height = '300px' width = '800px' align="center" />

### Getting The Image Tiles/Patches to the server
```shell
scp -r 'local dataset folder path' USER@SERVER_IP:'server directory folder path'
```

### Patch/Tile Score Calculation 
Score Can be calculated by just taking into consideration the number of nuclei or by using the histolab's implementation of tissue ratio plus the nuclei ratio.

[Tile Scorer Script](/src/data/data_preprocessing/tile_scorer.py)

Arguments for the script:
* `src` : src dataset folder
* `dst` : destination folder for storing the csv files generated by the script
* `type` : Type of scorer to use (default : nuclei_and_tissue) can be just nuclei as well

Sample Command For Running the Script:
``` shell
python tile_scorer.py DATASET_PATH DESTINATION_PATH --type nuclei_and_tissue
```

### Stain Normalization And Color Augmetation

Stain Normalization Can Either be Done dynamically when loading images to the deep learning model or creating by creating a completely new dataset.
Reinhard, Vahadne, Macenko methods are being used for stain normalization. The implementations of these methods by [HistomicsTk]() and [StainTools]() have been used.

The Patches can be normalized using a standard patch or using mean and sd for the target color space.

```shell
python 
```

* `--type`
* `--standard`

## Running The Model:
Yaml file description
Pytorch Lightning Description
Fine Tuning 
Explaning The Parameters

## Feature Visualization And Clustering

## Trained Model Checkpoints



Python Multiprocessing description

Yaml is data serialization language generally used to create the configuration files for the project.

This file contains all the parameters that are required to train or test the model.

The parameters are:
Descripton of the parameters:

### General Python File Commonalities:
1. Python Multiprocessing Module
2. glob
3. os module

## Handling Github
data directory if empty add to source control
if data added then using .gitignore file -> uncomment the line of /data/ to exclude it from being uploaded to github

commiting:
Pushing:
Staging Commits etc

requirement.txt file : creating this file pipreqs --savepath 
use this command while in project directory

installing using requirements file pip install -r requirements.txt

Enivronment variables to set:
Creating Virtual Environment
Python Src set for the package.
Installing the cuda packages depening on GPU

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── final_dataset       <- Final Selected Patches After Nuclei Ratio/ Tissue Ratio Calculation
    │   ├── stain_norm_dataset  <- Tiles/Patches with Stain Normalization Done
    │   ├── tiled_dataset       <- Conversion of WSI files to Tiles/Patches and Removal Of Empty Tiles
    │   └── raw_wsi             <- The Original .vsi files
    │
    ├── docs               <- Contains Readme.md file and also the images and diagrams associated with it 
    │
    ├── models             <- Trained Model weights. Checkpoints with weights(pytorch lightning)
    │
    ├── requirements.txt   <- Python Package requirements file -  generated with pipreqs
    │                        
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        |
        ├── config
        |    └──bermanlab.yaml  <- configurations for training and testing the model
        |
        ├── data           <- Scripts to transform data , dataloaders , dataset classes
        │   ├── data_preprocessing
        |           ├──cluster_dataset.ipynb
        |           ├──fast_feature_extraction.py
        |          
        |                mean_std_cal.py
        |                remove_empty_tiles.py
        |                select_patches.py
        |                stain_normalization.py
        |                Tile_Exporter.groovy
        |                tile_scorer.py
        |               
        |               
        |
        |
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models    <- Scripts to train models and then use trained models to make predictions
        │   ├── architechture
        |   ├── model architechture
        |   |     └──kimianet_modified.py   <- Pytorch Lightning module for the model
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create Visual Dictionary Using Extracted Features After fine Tuning the Model
               └── visualize.py

            

--------


## Issues:
- All issues reported on the forum of the repository