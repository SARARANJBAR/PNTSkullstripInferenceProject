# PNTSkullstripInferenceProject

## Introduction 

DeepBrain is SwansonLab's inhouse deep learning model for extracting brain tissue. this model was built using the Tensorflow based platform niftynet, which requires a specific format of data to run. For information about how this model was built, please read:

Ranjbar S, Singleton KW, Curtin L, Rickertsen CR, Paulson LE, Hu LS, Mitchell JR, Swanson KR. Weakly Supervised Skull Stripping of Magnetic Resonance Imaging of Brain Tumor Patients. From Prototype to Clinical Workflow: Moving Machine Learning for Lesion Quantification into Neuroradiological Practice. 2022 Aug 2.

This script is a wrapper for runninng inference on a new test sample (or samples, it works in batch). Given an input folder with images along with image identifier (string saying either T1Gd or FLAIR), and path to an output folder, this script does the following:

1 - fixes all image-related requirements of the trained net, including resampling
    and renaming

2 - generates a csv file, in which image ids and assigned cohort (in this case
    inference) are indicated

3 - generates a configation file for calling niftynet that include paths to input,
   output, model, model, etc.

4 - calls the niftynet using the configuration file in inference mode

5 - renames and resizes predictions to the original image space

6 - deletes all intermediate files

## Where to find the model objects

trainend models are provided in PNTMRSkullstrippingProject repository (due to the size of model objects they were not duplicated here). Please download the models from PNTMRSkullstrippingProject repository to your local machine for running the inference

##  Requirements
1) a folder with FLAIR images in nifti format, with an image identifier tag (e.g. FLAIR) in image names.
2) previously trained models (available in PNTSkullStrippingProject repository).
3) 

##  How to run  
1 - open a terminal window and cd where this file is located

2 - create an conda environment using the niftynet yml file (if you havenot already) by typing 'conda env create -f niftynet.yml'

3 - activate the environment in the terminal window: type 'source activate niftynet'

4 - type: 'python run_deepbrain.py -i <path_to_input_dir> -o <path_to_output_dir> -s FLAIR'
