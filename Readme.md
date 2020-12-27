# Project files from Cesare Magnetti

# Summary

scripts and results of my KURF experience

# DATA 

all models have been trained

# MODELS

all trained models are at the following directory:

*/home/cm19/Code/models*

in particular:

resnet18 trained without data augmentation:
*resnet18/*

resnet18 trained with data augmentation:
*resnet18_aug/*

vgg13 trained with data augmentation:
*vgg13_aug/*

vgg13 trained without data augmentation:
*vgg13/*

# RESULTS

all results obtained are contained in this repo inside the *results/* folder

# EXECUTION FROM COMMAND LINE:

only two commad line scripts:

**evaluate.py**:
*python evaluate.py --help*
*python evaluate.py --mode [cam/raw] -m [model name] -e [epochs info (min max step)] --isaug [aug/nonaug] -b [batch_size] -n [num_workers]*

**infer_test.py**:
*python infer_test.py --help*
*python infer_test.py -i [image] -m [model_name] -mp [model_path]*







