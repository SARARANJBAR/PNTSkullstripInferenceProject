#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:22:33 2021
@author: Sara Ranjbar, PhD

"""
import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np
import tempfile
import configparser
import pathlib
import sys
import time
import shutil

# ........... function ...................

def resampleandresize(sitkimage, new_size):
    """
    Resample simpleitk image to target size

    Parameters
    ----------
    sitkimage : sitk image
        input.
    new_size : target size
        list of 3 for image dimension.

    Returns
    -------
    resampled : sitkimage
        resized image.
    """
    #   The spatial definition of the images we want to use in a deep learning framework
    # (smaller than the original).
    reference_image = sitk.Image(new_size, sitkimage.GetPixelIDValue())
    reference_image.SetOrigin(sitkimage.GetOrigin())
    reference_image.SetDirection(sitkimage.GetDirection())
    reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, sitkimage.GetSize(), sitkimage.GetSpacing())])

    # Resample without any smoothing.
    resampled1 = sitk.Resample(sitkimage, reference_image)
    return resampled1

def resize_to_deepbrain_inputsize(input_dir, imtag, out_dir):
    """
    resized images to the input size that DeepBrain expects+ creates datasplitfile

    Parameters
    ----------
    input_dir : path
        path to image directory.
    imtag : string
        imagetype image names are expected to be id_imtag.nii.gz.
    out_dir : path
        path to where resized images and splitfile will be saved

    Returns
    -------
    splitfilepath: path
        path to datasplitfile.
    """
    print('\n--- resizing images & creating datasplitfile.csv ..')
    foldernames = [f for f in os.listdir(input_dir) if '-' in f]
    print('-found %d folders in %s'%(len(foldernames), input_dir))

    imtag = correct_imagetag(imtag)

    # datasplitfile csv
    splitfile_df  = pd.DataFrame(index=np.arange(len(foldernames)), columns=['id', 'set'])
    splitfile_df['set'] = 'inference'

    # masterfile for resizing
    resizinglog_df = pd.DataFrame(index=np.arange(len(foldernames)),
                             columns=['oldname',
                                      'oldsize',
                                      'newname',
                                      'newsize'])

    resizinglog_df['oldname'] = foldernames

    targetsize = [240, 240, 64]  # this is what deepBrain expects
    counter = 1
    num_digits = len(str(len(foldernames)))
    print('resizing to:', targetsize, '\n')
    for ind in resizinglog_df.index.values:

        # read image
        foldername = resizinglog_df.loc[ind, 'oldname']
        impath = os.path.join(input_dir, foldername, imtag + '.nii.gz')
        stkimg = sitk.ReadImage(impath)

        # log details of original image size
        oldsize = 'x'.join([str(v) for v in stkimg.GetSize()])
        newsize = 'x'.join([str(v) for v in targetsize])

        if oldsize != newsize:
            # resize
            resizedimage = resampleandresize(stkimg, targetsize)
        else:
            print('skip resizing.')
            resizedimage = stkimg

        # generate new name so it fits what niftynet expects
        new_id = str(counter).zfill(num_digits)+ '_'
        newname = "{}_{}{}".format(new_id, imtag, '.nii.gz')

        # log details of new name and new size
        resizinglog_df.loc[ind, 'oldsize'] = oldsize
        resizinglog_df.loc[ind, 'newsize'] = newsize
        resizinglog_df.loc[ind, 'newname'] = newname

        # add to datasplitfile
        splitfile_df.loc[ind, 'id'] = new_id
        # write img to file
        sitk.WriteImage(resizedimage, os.path.join(out_dir, newname))

        print('oldname:', foldername, '-oldsize:', oldsize, '-newname:', newname, '-newsize:', newsize)

        counter += 1

    base_dir = os.path.dirname(input_dir)
    resizing_masterfile_path = os.path.join(base_dir, 'resizing_master.csv')
    resizinglog_df.to_csv(os.path.join(base_dir, 'resizing_master.csv'), index=False)
    splitfilepath = os.path.join(base_dir, 'datasplitfile.csv')
    print(splitfile_df)
    splitfile_df.to_csv(splitfilepath, index=False, header=False)
    print('--- done.\n')

    return splitfilepath, resizing_masterfile_path

def create_config_file(input_dir, imtag, model_dir, inference_iter, splitfilepath, out_dir):
    """
    Creates a config file to run niftynet

    Parameters
    ----------
    input_dir : Path
        where to find images.
    imtag : STR
        String associated with image type = image identifiers.
    model_dir : Path
        where trained models are.
    inference_iter : int or str
        which model iteration to use.
    splitfilepath : PATH
        where datasplitfile.csv is.
    out_dir : PATH
        where to save images.

    Returns
    -------
    configfilepath : PATH
        path to saved config file.

    """
    print ('\n---creating deepBrain config file')

    config = configparser.ConfigParser()

    config['image'] = {'path_to_search': input_dir,
                       'filename_contains': '_' + imtag,
                       'spatial_window_size': '(144, 144, 144)',
                       'interp_order': '3',
                       'axcodes': '(R, A, S)'
                       }

    config['SYSTEM'] = {'cuda_devices': '',
                        'num_threads': '6',
                        'num_gpus': '1',
                        'model_dir': model_dir,
                        'queue_length': 36,
                        'dataset_split_file': splitfilepath
                        }

    config['NETWORK'] = {'name': 'dense_vnet',
                        'batch_size': '6',
                        'whitening': 'True',
                        'volume_padding_size': '0',
                        'window_sampling': 'resize'
                        }

    config['INFERENCE'] = {'border': '(0, 0, 0)',
                        'inference_iter': str(inference_iter),
                        'output_interp_order': '0',
                        'spatial_window_size': '(144, 144, 144)',
                        'dataset_to_infer': 'inference',
                        'output_postfix': '_brain_mask',
                        'save_seg_dir': out_dir
                        }

    config['SEGMENTATION'] = {'image': 'image',
                        'label': 'label',
                        'label_normalisation': 'True',
                        'output_prob': 'False',
                        'num_classes': '2'
                        }

    configfilepath = os.path.join(os.path.dirname(out_dir), 'config.ini')
    with open(configfilepath, 'w') as configfile:
        config.write(configfile)
    print(configfilepath)

    print('---done.')
    return configfilepath

def correct_imagetag(imtag):

    if imtag in ['T1', 't1', 't1gd', 't1Gd', 't1gD',
                 't1GD', 'T1GD', 'T1gD', 'T1Gd','T1gd']:
        imtag = 'T1GD'
    elif imtag in ['flair', 'FLAIR']:
        imtag = 'FLAIR'
    else:
        print('image tag %s unknown. Return'%imtag)
        imtag = None
    return imtag

def reverse_resize(resizing_csv, input_dir, output_dir):
    """
    Resize predicted masks to original image dimensions.

    Parameters
    ----------
    resizing_csv : TYPE
        masterfile for dimension of original images.
    input_dir : path
        path to predicted image directory.
    output_dir : path
        where to save the result.

    Returns
    -------
    None.

    """
    imagenames = [f for f in os.listdir(input_dir) if 'nii' in f]

    if len(imagenames) == 0:
        sys.exit()

    print('\n---resizing masks back to the original size..')
    resizinglog_df = pd.read_csv(resizing_csv)



    master_df = pd.read_csv(resizing_csv)

    for ind in master_df.index.values:

        oldname = master_df.loc[ind, 'oldname']
        new_name = master_df.loc[ind, 'newname']
        id_ = new_name.split('_')[0]

        maskname = 'window_seg_' + id_ + '___brain_mask.nii.gz'
        if not os.path.exists(os.path.join(input_dir, maskname)):
            print('%s not found. skipping.'% maskname)
            continue

        stkimg = sitk.ReadImage(os.path.join(input_dir, maskname))
        oldsize = master_df.loc[ind, 'oldsize']
        newsize = master_df.loc[ind, 'newsize']
        oldsize_ls = [int(s) for s in oldsize.split('x')]
        resizedimage = resampleandresize(stkimg, oldsize_ls)

        outimname = oldname.split('.')[0] + '_brain_mask_pred.nii.gz'
        outimgpath = os.path.join(output_dir, outimname)
        sitk.WriteImage(resizedimage, outimgpath)
        print('image:', outimname, '-oldsize:', newsize, '-targetsize:',
              oldsize, '-newname:', outimname)

    print('-done.')

if __name__ == '__main__':

    rootDir = '/Volumes/MLDL_Projects/Sara/postTx_segmentation_untouched'
    modelRootDir = '/Volumes/MLDL_Projects/Sara/DeepBrain-pipelined/DeepBrainModels'
    
    inputDir = os.path.join(rootDir, 'completeAndcorrect/maybeCases')
    outputDir = os.path.join(rootDir, 'completeAndcorrect_brainMasks/round2Masks')
    imagetag = 'FLAIR'

    # anything created in the intermediate stepswill be removed at the end
    # we just need 'A' place
    dirForTempFiles = '/Volumes/MLDL_Projects/Sara'

    start_time = time.time()
    print ('\n----running deepbrain on postTx segmentation data-----')
    print("\nmodelDir: ", modelRootDir)
    print("inputDir: ", inputDir)
    print("outputDir: ", outputDir)
    print("imagetag: ", imagetag)

    # create folders, clean up anything remaining from previous runs
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    if not os.path.exists(modelRootDir):
        print('Error. Model directory not found.')
        sys.exit()

    if imagetag in ['T1GD', 'T1' ,'T1Gd' ,'t1gd']:
        modelDir = os.path.join(modelRootDir, 'only_t1', 'decay0_whitening_models')
        inference_iter = 220 # can be 120, 220, or 300

    elif imagetag in ['FLAIR', 'flair']:
        modelDir = os.path.join(modelRootDir, 'only_flair', 'decay0_whitening_models')
        inference_iter = 380 # can be 220, 380, 420, or 480
    else:
        print('Error - image type %s doesnt match model input (T1GD or FLAIR).'%imagetag)
        sys.exit()

    # the model needs a specific sized input
    # create local input dir for storing resized images
    # this folder will be removed later
    deepBrainInputDir = os.path.join(dirForTempFiles, 'DeepBrainInput')
    if os.path.exists(deepBrainInputDir):
        for f in os.listdir(deepBrainInputDir):
            os.remove(os.path.join(deepBrainInputDir, f))
    else:
        os.makedirs(deepBrainInputDir)

    # create output dir for storing predicted masks
    deepBrainOutputDir = os.path.join(dirForTempFiles, 'DeepBrainOutput')
    if os.path.exists(deepBrainOutputDir):
        for f in os.listdir(deepBrainOutputDir):
            os.remove(os.path.join(deepBrainOutputDir, f))
    else:
        os.makedirs(deepBrainOutputDir)

    # if data splitfile exists, delete it, we will make it again
    splitfilepath = os.path.join(dirForTempFiles, 'datasplitfile.csv')
    if os.path.exists(splitfilepath):
        os.remove(splitfilepath)


    # if an error occurs along the way, success is set to False.
    # this helps with cleaning up afterwards. If it is false, intermediate files
    # are not deleted and therefore can be used for debugging. Otherwise all
    # intermediate files are deleted at the end.
    success = True

    try:

        # Step 0 : activate the niftynet environment
        #command = 'conda activate niftynet'
        #os.system(command)

        # Step 1) resize images to targetsize, save to DeepBrainInputDir
        res = resize_to_deepbrain_inputsize(inputDir, imagetag, deepBrainInputDir)

        splitfilepath,resizing_masterfile_path = res

        # Step 2) create config file for running niftynet
        configfilepath = create_config_file(deepBrainInputDir,
                                            imagetag,
                                            modelDir,
                                            inference_iter,
                                            splitfilepath,
                                            deepBrainOutputDir)

        # Step 3) run inference on niftynet using configfile, resized images, and
        # a previously trained model
        print('\n---running niftynet ...')
        command = 'net_run inference -a niftynet.application.segmentation_application.SegmentationApplication'
        command += ' -c ' + configfilepath
        print(command)
        os.system(command)

        # Step 4) reverse resizing to match original image sizes
        reverse_resize(resizing_masterfile_path, deepBrainOutputDir, outputDir)

    except Exception as e:
        print(e)
        success = False


    print('\n---cleaning up..')
    # if successful delete all intermediate files and folders that were creatd
    for temppath in [deepBrainInputDir,
                    deepBrainOutputDir,
                    splitfilepath,
                    configfilepath,
                    resizing_masterfile_path,
                    resizing_masterfile_path]:

        if os.path.isfile(temppath):
            os.remove(temppath)

        if os.path.isdir(temppath):
            shutil.rmtree(temppath)

    print('\nDone. Output folder :', outputDir)
    print('total deep brain processing time: ', time.time() - start_time, 'sec.')
