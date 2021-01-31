"""
author: Antoine Spahr

date : 05.11.2020

----------

TO DO :
- to be debugged
"""
import os
import shutil
import json
import glob
import numpy as np
import pandas as pd
import skimage.io as io
import skimage.transform
from skimage import img_as_bool, img_as_ubyte
import nibabel as nib
from sklearn.metrics import confusion_matrix

from src.utils.print_utils import print_progessbar
from src.utils.python_utils import AttrDict
from src.postprocessing.analyse_exp import analyse_supervised_exp

def update_pred_folder(pred_folder, save_path, brain_mask_folder, brain_as_nifti, data_path, print_progress=False, rot=True):
    """
    Adjust segmentation predictions on a folder organised as follow : pred folder contains sub-folder for each volume
    prediction as slices in .bmp. The sub-folders are named by the volume id, similar to the dataset and the brain masks.
    ----------
    INPUT
        |---- pred_folder (str) path to the folder containing th predictions to adjusts.
        |---- save_path (str) path where to save the adjusted predictions.
        |---- brain_mask_folder (str) where to find the brain masks for each volumes. Brain masks can be given as folder
        |           of .bmp slices or as a Nifti volume. The folder name or Nifti volume must be named as the volume id
        |           (0 padded to 3 digits).
        |---- brain_as_nifti (bool) specified whether brain mask are passed as Nifti or folder of .bmp.
        |---- data_path (str) path to the data where a 'ct_info.csv' gives ground_truth bmp filenames
        |---- print_progress (bool) whether to print progress with a progress bar.
        |---- rot (bool) whether to rotate the brain mask by 90° counterclockwise.
    OUTPUT
        |---- None
    """
    # load data_info_csv
    data_info = pd.read_csv(os.path.join(data_path, 'ct_info.csv'), index_col=0)
    # iterate over each volumes predictions
    out_info = []
    for vol_folder in glob.glob(os.path.join(pred_folder, '*/')):
        id = int(vol_folder.split('/')[-2])
        # make volume output dir
        os.makedirs(os.path.join(save_path, f'{id}'), exist_ok=True)
        # get brain mask if nifti
        if brain_as_nifti:
            try:
                brain_vol = nib.load(os.path.join(brain_mask_folder, f'{id:03}.nii')).get_fdata()
            except FileNotFoundError:
                brain_vol = nib.load(os.path.join(brain_mask_folder, f'{id:03}.nii.gz')).get_fdata()
        # iterate over Slices
        n_slice = len(glob.glob(os.path.join(vol_folder, '*.bmp')))
        assert n_slice == brain_vol.shape[2], f'The number of slice between the prediction {id} and the brain mask does not match. {n_slice} vs {brain_vol.shape[2]}.'
        for slice in range(1, n_slice+1):
            # load pred
            pred = io.imread(os.path.join(vol_folder, f'{slice}.bmp'))
            # adjust brain mask size to match prediction
            if brain_as_nifti:
                brain_slice = skimage.transform.resize(brain_vol[:,:,slice-1], pred.shape, order=0)
            else:
                brain_slice = io.imread(os.path.join(brain_mask_folder, f'{id:03}/{slice}.bmp'))
                brain_slice = skimage.transform.resize(brain_slice, pred.shape, order=0, preserve_range=True)
            if rot:
                brain_slice = np.rot90(brain_slice, axes=(0,1))
            # keep only prediction inside the brain
            new_pred = img_as_bool(pred) * img_as_bool(brain_slice)
            # load the ground truth and adjust the size
            target_fn = data_info.loc[(data_info.PatientNumber == id) & (data_info.SliceNumber == slice), 'mask_fn'].values[0]
            if target_fn != 'None':
                target = io.imread(os.path.join(data_path, target_fn))
                target = skimage.transform.resize(target, new_pred.shape, order=0, preserve_range=True)
            else:
                target = np.zeros_like(new_pred)
            target = img_as_bool(target)
            # compute the confusion matrix
            tn, fp, fn, tp = confusion_matrix(target.ravel(), new_pred.ravel(), labels=[0,1]).ravel()
            # save new prediction
            new_pred_fn = os.path.join(save_path, f'{id}', f'{slice}.bmp')
            io.imsave(new_pred_fn, img_as_ubyte(new_pred), check_contrast=False)
            # append results
            label = data_info.loc[(data_info.PatientNumber == id) & (data_info.SliceNumber == slice), 'Hemorrhage'].values[0]
            out_info.append({'volID':id, 'slice':slice, 'label':label, 'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn, 'pred_fn': f'{id}/{slice}.bmp'})
            # print progress
            if print_progress:
                print_progessbar(slice-1, n_slice, Name=f'Volume {id:03} ; Slice', Size=40, erase=False)

    # make df of results for folder
    slice_df = pd.DataFrame(out_info)
    volume_df = slice_df[['volID', 'label', 'TP', 'TN', 'FP', 'FN']].groupby('volID').agg({'label':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})
    # compute Dice
    slice_df['Dice'] = (2*slice_df.TP + 1) / (2*slice_df.TP + slice_df.FP + slice_df.FN + 1)
    volume_df['Dice'] = (2*volume_df.TP + 1) / (2*volume_df.TP + volume_df.FP + volume_df.FN + 1)
    # save df
    slice_df.to_csv(os.path.join(save_path, 'slice_prediction_scores.csv'))
    volume_df.to_csv(os.path.join(save_path, 'volume_prediction_scores.csv'))
    # update outputs json
    avg_dice_ICH = volume_df.loc[volume_df.label == 1, 'Dice'].mean(axis=0)
    avg_dice = volume_df.Dice.mean(axis=0)
    with open(os.path.abspath(os.path.join(os.path.dirname(pred_folder), "..", "outputs.json")), 'r') as fn:
        outputs = json.load(fn)
    outputs['eval']['dice'] = {'all':avg_dice, 'positive':avg_dice_ICH}
    with open(os.path.join(os.path.dirname(save_path), "outputs.json"), 'w') as fn:
        json.dump(outputs, fn)

def update_Kfold_folder(exp_folder, save_path, brain_mask_folder, data_path, rot=True, print_progress=False, verbose=False):
    """
    Adjust segmentation predictions on a KFold cross validation experiment whose folder is organised as follow : exp_folder
    contains sub-folder for each Fold as 'Fold_i'. Each fold-folder contains prediction for several volumes stored in
    folder samed after the volume id (similar to the dataset and the brain masks.). Each volume folder contains
    prediction as slices in .bmp.
    ----------
    INPUT
        |---- exp_folder (str) path to the folder containing th Fold and predictions to adjusts.
        |---- save_path (str) path where to save the adjusted folds.
        |---- brain_mask_folder (str) where to find the brain masks for each volumes. Brain masks can be given as folder
        |           of .bmp slices or as a Nifti volume. The folder name or Nifti volume must be named as the volume id.
        |---- brain_as_nifti (bool) specified whether brain mask are passed as Nifti or folder of .bmp.
        |---- data_path (str) path to the data where a 'ct_info.csv' gives ground_truth bmp filenames
        |---- rot (bool) whether to rotate the brain mask by 90° counterclockwise.
        |---- print_progress (bool) whether to print progress with a progress bar.
        |---- verbose (bool) whether to print details of the processing.
    OUTPUT
        |---- None
    """
    # process each fold
    vol_df_list = []
    n_fold = 0
    for i, fold_folder in enumerate(sorted(glob.glob(os.path.join(exp_folder, 'Fold_*/')), key=lambda x: int(x.split('/')[-2].split('_')[-1]))):
        if verbose: print(f">>> Processing Fold {i+1}")
        n_fold += 1
        # check if fold already processed
        if not os.path.exists(os.path.join(save_path, f'Fold_{i+1}/outputs.json')):
            # make dirs
            os.makedirs(os.path.join(save_path, f'Fold_{i+1}/pred'), exist_ok=True)
            # update fold folder predictions
            brain_as_nifti = False if len(glob.glob(os.path.join(brain_mask_folder, '*.nii')) + glob.glob(os.path.join(brain_mask_folder, '*.nii.gz'))) == 0 else True
            update_pred_folder(os.path.join(fold_folder, 'pred/'), os.path.join(save_path, f'Fold_{i+1}/pred'),
                               brain_mask_folder, brain_as_nifti, data_path, rot=rot, print_progress=print_progress)

            # copy log and model.pt in save folder
            shutil.copyfile(os.path.join(fold_folder, 'log.txt'), os.path.join(save_path, f'Fold_{i+1}/log.txt'))
            shutil.copyfile(os.path.join(fold_folder, 'trained_unet.pt'), os.path.join(save_path, f'Fold_{i+1}/trained_unet.pt'))
            if verbose: print(f">>> log.txt and trained_unet.pt copied to {os.path.join(save_path, f'Fold_{i+1}/')}")

        # get volume_df
        vol_df_list.append(pd.read_csv(os.path.join(save_path, f'Fold_{i+1}/pred/volume_prediction_scores.csv'), index_col=0))

    # concatenate all fold volume_df
    all_volume_df = pd.concat(vol_df_list, axis=0).reset_index()
    all_volume_df.to_csv(os.path.join(save_path, 'all_volume_prediction.csv'))
    if verbose: print(f">>> All prediction CSV saved at {os.path.join(save_path, 'all_volume_prediction.csv')}")

    # update avg_score txt file
    scores_list = []
    for k in range(n_fold):
        with open(os.path.join(save_path, f'Fold_{k+1}/outputs.json'), 'r') as f:
            out = json.load(f)
        scores_list.append([out['eval']['dice']['all'], out['eval']['dice']['positive']])
    means = np.array(scores_list).mean(axis=0)
    CI95 = 1.96*np.array(scores_list).std(axis=0)
    with open(os.path.join(save_path, 'average_scores.txt'), 'w') as f:
        f.write(f'Dice = {means[0]} +/- {CI95[0]}\n')
        f.write(f'Dice (Positive) = {means[1]} +/- {CI95[1]}\n')
    if verbose: print(f">>> Average Scores saved at {os.path.join(save_path, 'average_scores.txt')}")

    # add entry to config_file for brain_mask source
    with open(os.path.join(exp_folder, 'config.json'), 'r') as fn:
        config = json.load(fn)
    config['brain_mask'] = True
    config['path']['brain'] = brain_mask_folder
    with open(os.path.join(save_path, 'config.json'), 'w') as fn:
        json.dump(config, fn)
    if verbose: print(">>> Config file updated.")

    # generate summary figure
    analyse_supervised_exp(save_path, data_path, n_fold, save_fn=os.path.join(save_path, 'results_overview.pdf'))
    if verbose: print(f">>> Summary figure saved at {os.path.join(save_path, 'results_overview.pdf')}")

def update_anomaly_pred_folder(pred_folder, save_path, brain_mask_folder, brain_as_nifti, data_path, print_progress=False, rot=True):
    """
    Adjust Anomaly segmentation predictions on a folder organised as follow : pred folder contains sub-folder for each volume
    prediction as slices in XX_anomalies.bmp and anomaly map as XX_map_anomalies.png. The sub-folders are named by the
    volume id, similar to the dataset and the brain masks.
    ----------
    INPUT
        |---- pred_folder (str) path to the folder containing th predictions to adjusts.
        |---- save_path (str) path where to save the adjusted predictions.
        |---- brain_mask_folder (str) where to find the brain masks for each volumes. Brain masks can be given as folder
        |           of .bmp slices or as a Nifti volume. The folder name or Nifti volume must be named as the volume id
        |           (0 padded to 3 digits).
        |---- brain_as_nifti (bool) specified whether brain mask are passed as Nifti or folder of .bmp.
        |---- data_path (str) path to the data where a 'ct_info.csv' gives ground_truth bmp filenames
        |---- print_progress (bool) whether to print progress with a progress bar.
        |---- rot (bool) whether to rotate the brain mask by 90° counterclockwise.
    OUTPUT
        |---- None
    """
    # load data_info_csv
    data_info = pd.read_csv(os.path.join(data_path, 'ct_info.csv'), index_col=0)
    # get image size from config file
    cfg = AttrDict.from_json_path(os.path.join(os.path.dirname(pred_folder.strip('/')), 'config.json'))
    im_size = cfg.data.size

    # iterate over each volumes predictions
    out_info = []
    for vol_folder in glob.glob(os.path.join(pred_folder, '*/')):
        id = int(vol_folder.split('/')[-2])
        # make volume output dir
        os.makedirs(os.path.join(save_path, f'{id}'), exist_ok=True)
        # get brain mask if nifti
        if brain_as_nifti:
            try:
                brain_vol = nib.load(os.path.join(brain_mask_folder, f'{id:03}.nii')).get_fdata()
            except FileNotFoundError:
                brain_vol = nib.load(os.path.join(brain_mask_folder, f'{id:03}.nii.gz')).get_fdata()
        # iterate over Slices
        n_slice = brain_vol.shape[2]#len(glob.glob(os.path.join(vol_folder, '*.bmp')))
        #assert n_slice == brain_vol.shape[2], f'The number of slice between the prediction {id} and the brain mask does not match. {n_slice} vs {brain_vol.shape[2]}.'
        for slice in range(1, n_slice+1):
            # load pred
            if os.path.exists(os.path.join(vol_folder, f'{slice}_anomalies.bmp')): # check if pred is there
                pred = io.imread(os.path.join(vol_folder, f'{slice}_anomalies.bmp'))
                ad_map = io.imread(os.path.join(vol_folder, f'{slice}_map_anomalies.png'))
                save_im = True
            else:
                pred, ad_map = np.zeros([im_size, im_size]), np.zeros([im_size, im_size])
                save_im = False

            # adjust brain mask size to match prediction
            if brain_as_nifti:
                brain_slice = skimage.transform.resize(brain_vol[:,:,slice-1], pred.shape, order=0)
            else:
                brain_slice = io.imread(os.path.join(brain_mask_folder, f'{id:03}/{slice}.bmp'))
                brain_slice = skimage.transform.resize(brain_slice, pred.shape, order=0, preserve_range=True)
            if rot:
                brain_slice = np.rot90(brain_slice, axes=(0,1))

            # keep only prediction inside the brain
            new_pred = img_as_bool(pred) * img_as_bool(brain_slice)
            new_ad_map = ad_map * img_as_bool(brain_slice)

            # load the ground truth and adjust the size
            target_fn = data_info.loc[(data_info.PatientNumber == id) & (data_info.SliceNumber == slice), 'mask_fn'].values[0]
            if target_fn != 'None':
                target = io.imread(os.path.join(data_path, target_fn))
                target = skimage.transform.resize(target, new_pred.shape, order=0, preserve_range=True)
            else:
                target = np.zeros_like(new_pred)
            target = img_as_bool(target)

            # compute the confusion matrix
            tn, fp, fn, tp = confusion_matrix(target.ravel(), new_pred.ravel(), labels=[0,1]).ravel()

            # save new prediction
            if save_im:
                new_pred_fn = os.path.join(save_path, f'{id}', f'{slice}_anomalies.bmp')
                io.imsave(new_pred_fn, img_as_ubyte(new_pred), check_contrast=False)
                new_ad_map_fn = os.path.join(save_path, f'{id}', f'{slice}_map_anomalies.png')
                io.imsave(new_ad_map_fn, img_as_ubyte(new_ad_map), check_contrast=False)

            # append results
            label = data_info.loc[(data_info.PatientNumber == id) & (data_info.SliceNumber == slice), 'Hemorrhage'].values[0]
            out_info.append({'volID':id, 'slice':slice, 'label':label, 'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn, 'pred_fn': f'{id}/{slice}_anomalies.bmp', 'map_fn': f'{id}/{slice}_map_anomalies.png'})

            # print progress
            if print_progress:
                print_progessbar(slice-1, n_slice, Name=f'Volume {id:03} ; Slice', Size=40, erase=False)

    # make df of results for folder
    slice_df = pd.DataFrame(out_info)
    volume_df = slice_df[['volID', 'label', 'TP', 'TN', 'FP', 'FN']].groupby('volID').agg({'label':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})
    # compute Dice
    slice_df['Dice'] = (2*slice_df.TP + 1) / (2*slice_df.TP + slice_df.FP + slice_df.FN + 1)
    volume_df['Dice'] = (2*volume_df.TP + 1) / (2*volume_df.TP + volume_df.FP + volume_df.FN + 1)
    # save df
    slice_df.to_csv(os.path.join(save_path, 'slice_prediction_scores.csv'))
    volume_df.to_csv(os.path.join(save_path, 'volume_prediction_scores.csv'))
    # save outputs json
    avg_dice_ICH = volume_df.loc[volume_df.label == 1, 'Dice'].mean(axis=0)
    avg_dice = volume_df.Dice.mean(axis=0)
    outputs = {'dice all':avg_dice, 'dice positive':avg_dice_ICH}
    with open(os.path.join(os.path.dirname(save_path), "outputs.json"), 'w') as fn:
        json.dump(outputs, fn)

    with  open(os.path.join(os.path.dirname(save_path), "config.json"), 'w') as fn:
        json.dump(cfg, fn)
