Feature Extractor
==============================

Finetuning DenseNet121 architechture using weights of the model provided by KimiaNet for extracting features From Whole Slide Image Patches relevant to Cancer Grades

## Updates:

## Pre-requisites:
    * Linux
    * 

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

## Dataset Preparation Steps

## Feature Visualization And Clustering

## Trained Model Checkpoints

## Examples

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── final dataset  <- Tiles Selected as final dataset after Nuclei Ratio calculation and Removal of Artifact Tiles
    │   ├── interim        <- Intermediate data that has been transformed. Empty Tiles Removed and Stain Normalization Done
    │   ├── Tiled_Dataset  <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-pk-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with pipreqs
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    |
    ├── config
    |    └──bermanlab.yaml  <- configurations for training and testing the model
    |
    ├── data           <- Scripts to transform data , dataloaders , dataset classes
    │   └── make_dataset.py
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
