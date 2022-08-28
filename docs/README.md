Feature Extractor
==============================

Finetuning DenseNet121 architechture using weights of the model provided by KimiaNet for extracting features From Whole Slide Image Patches relevant to Cancer Grades

FineTuning:

DenseNet121 Architechture

pip install -r /path/to/requirements.txt

## Updates:


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


### Getting The Image Tiles/Patches to the server
```shell
scp -r 'local dataset folder path' USER@SERVER_IP:'server directory folder path'
```

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

### Final Selection of Patches 
The Top Patches are finally selected On Basis of the Tile Scores and the results of clustering.
[select_patches](/src/data/data_preprocessing/select_patches.py)

Arguments for the sript:
* `cluster` : Path to File with selected Patches after clustering
* `patch_score`: Path to Directory That contains patch score files
* `dst`: The Path to store the final json file. Path format : directory/file_name.json'

Sample Command For Running The Script:
```shell
python select_patches.py CLUSTERING_RESULT_FILE PATCH_SCORE_FOLDER DST_FILE_PATH
```

### Stain Normalization And Color Augmetation

Stain Normalization Can Either be Done dynamically when loading images to the deep learning model or creating by creating a completely new dataset.
Reinhard, Vahadne, Macenko methods are being used for stain normalization. The implementations of these methods by [HistomicsTk](https://digitalslidearchive.github.io/HistomicsTK/) and [StainTools](https://github.com/Peter554/StainTools) have been used.

The Patches can be normalized using a standard patch or using mean and sd for the target color space.

Sample Command
```shell
python stain_normalization.py SRC_DIR DST_DIR
```
Arguments:
* `src`: Path to the dataset folder with the Patches
* `dst`: Path to store the normalized images to.
* `--type`: Type of normalization to use. using mean_and_std or a reference image
* `--standard`: Path to the image to be used as a reference for normalization.

*If Stain Normalization is to be Done Dynamically while loading images to the model. Check The tiles_dataset.py for instructions.*


## Running The Model

For Running the model on custom dataset
Change the Yaml [config_file](/src/config/bermanlab.yaml) with appropriate parameters before running the script.
The Yaml file can be used to change the mode from train to test.

Pytorch Lightning:

```shell
python main.py --config CONFIG_FILE_PATH 

```
Arguments:
* `config`: configuration file path

## Feature Extraction
Run the [extract_features.py](/src/models/extract_features.py) to extract features using the fine_tuned model for specific Patches.
Sample Command
```shell
python extract_features.py SAVE_ADD MODEL_WEIGTHS CONFIG SELECTED
```
Arguments:
* `save_add`: Path of Directory Where to store the Extracted Features (Add / after the path)
* `model_weights`: Path to weights file to use for feature extraction
* `config`: Path to the model configuration File
* `selected`: Path to CSV file that contains the List of Paths of Patches whose features are required to be extracted.

*The GPU to be used can be changed in the file itself.*




## Extracted Feature Analysis
### Feature Importance Calculation
The Importance of Each Feature is calculated using Support Vector Machines.

Sample Code To Run the [Script](/src/features/feature_importance.py)
```shell

python feature_importance.py DST_PATH FEATURE_FILE_PATH DATA_CSV_PATH
```
Arguments:
* `dst`: Destination Directory to store the feature importance result csv file
* `features`: Path of feature file to analyze
* `labels`: Path to the Data CSV file with sample ids and labels.

## General Commonalities Between The files
Many of the scripts use python multiprocessing module to run multiple processes simultaneously to speed up the calculation.
More Details regarding it's Use can be found at : [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)

For Running the Scripts from command line.
Argparse has been used. 
For every python script if `python script.py --help` will provide the details of all the arguments required with their description.


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
        │    └──data_preprocessing
        |           ├── cluster_dataset.ipynb
        |           ├── fast_feature_extraction.py
        |           ├── mean_std_cal.py
        |           ├── remove_empty_tiles.py 
        |           ├── select_patches.py  
        |           ├── stain_normalization.py
        |           ├── Tile_Exporter.groovy
        |           └── tile_scorer.py 
        ├── features       
        |    ├── feature_analysis.ipynb
        │    └── feature_importance.py
        |           
        └── models    <- Scripts to train models and then use trained models to make predictions
            ├── architechture
            |     └──model_interface.py   <- Pytorch Lightning module for the model
            ├── extract_features.py
            └── main.py
        

        
--------

### Papers to Cite:
1. https://www.sciencedirect.com/science/article/pii/S1361841521000785 KimiNet
2. https://ajp.amjpathol.org/article/S0002-9440(21)00387-4/fulltext kimiaNet Feature Visualization
3. https://www.nature.com/articles/s41598-022-08974-8.pdf?origin=ppub Clustering on basis of Extracted Features


To ADD : Gini Index Calculation for the Clustering Finally Done.

## Issues:
- All issues reported on the forum of the repository
