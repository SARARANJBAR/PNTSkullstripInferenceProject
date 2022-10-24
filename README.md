# PNTSkullstripInferenceProject

PNTDeepBrain is SwansonLab's inhouse deep learning model for extracting brain tissue. this model was built using the Tensorflow based platform niftynet, which requires a specific format of data to run. 

## reference
For information about how the model was trained, please read:

Ranjbar S, Singleton KW, Curtin L, Rickertsen CR, Paulson LE, Hu LS, Mitchell JR, Swanson KR. Weakly Supervised Skull Stripping of Magnetic Resonance Imaging of Brain Tumor Patients. From Prototype to Clinical Workflow: Moving Machine Learning for Lesion Quantification into Neuroradiological Practice. 2022 Aug 2.

## Where to find previously trained model objects

Trainend models are provided in PNTDeepBrainModels repository (due to the size of model objects they were not duplicated here). Please download the FLAIR models from this repository to your local machine for running inference.

## run_deepbrain.py 

run_deepbrain.py is a wrapper that takes care of all preprocessing that needs to happen before you can run the model on a new test case. This script does a lot of things: 

1 - finds all input images and resamples and renames data to match what niftynet expects

2 - generates a csv file, in which new image ids are assigned to inference cohort (needed for running niftynet)

3 - generates a reference csv file for keeping track of name change and spacing change.

4 - writes a configation file for calling niftynet that include paths to input,
   output, model, etc.

5 - calls the niftynet platform using said config file in inference mode, this will generate predicted brain masks.

6 - renames and resizes predicted mask back to the original image space and image names

7 - deletes all intermediate files


##  Data  preparation

You need to prepare a folder with your test case FLAIR images in nifti format. Niftynet works in batches, so it can be a folder with all of your test cases or just one case.

Note: Images NEED to have an image type identifier tag (in this case FLAIR) in their names otherwise they won't be recognized.

## how to use

Once you have your data and models ready, clone this repository, open a terminal window and cd where run_deepbrain.py is located.

2 - create an conda environment using the niftynet yml file (if you havenot already) by typing 'conda env create -f niftynet.yml'

3 - activate the environment in the terminal window: type 'source activate niftynet'

4 - type: 'python run_deepbrain.py -i <path_to_input_dir> -o <path_to_output_dir> -s FLAIR'
