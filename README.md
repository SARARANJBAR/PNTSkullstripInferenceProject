# PNTSkullstripInferenceProject

## Introduction 
DeepBrain is SwansonLab's inhouse deep learning model for extracting brain tissue. this model was built using the Tensorflow based platform niftynet, which requires a specific format of data to run.

Given an input folder with images along with image identifier (string saying either T1Gd or FLAIR), and an output folder, this script does the following:

1 - fixes all image-related requirements of the trained net, including resampling
    and renaming

2 - generates a csv file, in which image ids and assigned cohort (in this case
    inference) are indicated

3 - generates a configation file for calling niftynet that include paths to input,
   output, model, model, etc.

4 - calls the niftynet using the configuration file in inference mode

5 - renames and resizes predictions to the original image space

6 - deletes all intermediate files

##  Requirements
1) a folder with T2-FLAIR images in nifti format, with an image identifier tag (e.g. FLAIR) in image names.
2) previously trained models (available in PNTSkullStrippingProject repository).
3) 
##  How to run  
1 - open a terminal window and cd where this file is located

2 - create an conda environment using the niftynet yml file (if you havenot already) by typing 'conda env create -f niftynet.yml'

3 - activate the environment in the terminal window: type 'source activate niftynet'

4 - type: 'python run_deepbrain.py -i <path_to_input_dir> -o <path_to_output_dir> -s FLAIR'
