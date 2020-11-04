"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import nibabel as nib
import time
import logging
import os
import json
from datetime import timedelta

from src.models.optim.LossFunctions import BinaryDiceLoss
import src.dataset.transforms as tf
from src.utils.tensor_utils import batch_binary_confusion_matrix
from src.utils.ct_utils import window_ct
from src.utils.print_utils import print_progessbar

class UNet2D:
    """
    Class defining a 2.5D U-Net model to train and evaluate for binary segmentation as well as other utilities. The evaluation
    mehtod compute the performances (Dice) for the 3D volume predictions.
    """
    def __init__(self, unet, n_epoch=150, batch_size=16, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
                 lr_scheduler_kwargs=dict(gamma=0.95), loss_fn=BinaryDiceLoss, loss_fn_kwargs=dict(reduction='mean'),
                 weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False):
        """
        Build a UNet2D object.
        ----------
        INPUT
            |---- unet (nn.Module) the network to use in the model. It must take 2D input and genearte a 2D segmentation mask.
            |           Has to output a tensor of shape similar as input (logit for each pixel as outputed by the sigmoid.)
            |---- n_epoch (int) the number of epoch for the training.
            |---- batch_size (int) the batch size to use for loading the data.
            |---- lr (float) the learning rate.
            |---- lr_scheduler (torch.optim.lr_scheduler) the learning rate evolution scheme to use.
            |---- lr_scheduler_kwargs (dict) the keyword arguments to be passed to the lr_scheduler.
            |---- loss_fn (nn.Module) the loss function to use. Should take prediction and mask as forward input.
            |---- loss_fn_kwargs (dict) the keyword arguments to pass to the loss function constructor.
            |---- weight_decay (float) the L2-regularization strenght.
            |---- num_workers (int) the number of workers to use for data loading.
            |---- device (str) the device to use.
            |---- print_progress (bool) whether to print progress bar for batch processing.
        OUTPUT
            |---- UNet () the 2D UNet model.
        """
        self.unet = unet
        # training parameters
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.loss_fn = loss_fn
        self.loss_fn_kwargs = loss_fn_kwargs
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.device = device
        self.print_progress = print_progress

        # Results
        self.outputs = {
            'train':{
                'time': None,
                'evolution': None
            },
            'eval':{
                'time': None,
                'dice': None
            }
        }

    def train(self, dataset, valid_dataset=None, checkpoint_path=None):
        """
        Train the network with the given dataset(s).
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It must return an input image, a
            |           target binary mask, the patientID, and the slice number.
            |---- valid_dataset (torch.utils.data.Dataset) the optional validation dataset. If provided, the model is
            |           validated at each epoch. It must have the same struture as the train dataset.
            |---- checkpoint_path (str) the filename for a possible checkpoint to start the training from.
        OUTPUT
            |---- net (nn.Module) the trained network.
        """
        logger = logging.getLogger()
        # make the dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers)
        # put net to device
        self.unet = self.unet.to(self.device)
        # define optimizer
        optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # define the lr scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        # define the loss function
        loss_fn = self.loss_fn(**self.loss_fn_kwargs)
        # Load checkpoint if present
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            n_epoch_finished = checkpoint['n_epoch_finished']
            self.unet.load_state_dict(checkpoint['net_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['lr_state'])
            epoch_loss_list = checkpoint['loss_evolution']
            logger.info(f'Checkpoint loaded with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            logger.info('No Checkpoint found. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = [] # Placeholder for epoch evolution

        # start training
        logger.info('Start training the U-Net 2.5D.')
        start_time = time.time()
        n_batch = len(train_loader)

        for epoch in range(n_epoch_finished, self.n_epoch):
            self.unet.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get data
                input, target, _, _ = data
                # put data tensors on device
                input = input.to(self.device).float().requires_grad_(True)
                target = target.to(self.device).float().requires_grad_(True)
                # zero the networks' gradients
                optimizer.zero_grad()
                # optimize weights with backpropagation on the batch
                pred = self.unet(input)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # print process
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=40, erase=True)

            # Get validation performance if required
            valid_dice = ''
            if valid_dataset:
                self.evaluate(valid_dataset, print_to_logger=False, save_path=None)
                valid_dice = f"| Valid Dice: {self.outputs['eval']['dice']['all']:.5f} " + \
                             f"| Valid Dice (Positive Slices): {self.outputs['eval']['dice']['positive']:.5f} "

            # log the epoch statistics
            logger.info(f'\t| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train time: {timedelta(seconds=int(time.time() - epoch_start_time))} '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} ' + valid_dice +
                        f'| lr: {scheduler.get_last_lr()[0]:.7f} |')
            # Store epoch loss and epoch dice
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, self.outputs['eval']['dice']['all'], self.outputs['eval']['dice']['positive']])
            # update scheduler
            scheduler.step()
            # Save Checkpoint every 10 epochs
            if (epoch+1) % 10 == 0 and checkpoint_path:
                checkpoint = {'n_epoch_finished': epoch+1,
                              'net_state': self.unet.state_dict(),
                              'optimizer_state': optimizer.state_dict(),
                              'lr_state': scheduler.state_dict(),
                              'loss_evolution': epoch_loss_list}
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

        # End training
        self.outputs['train']['time'] = time.time() - start_time
        self.outputs['train']['evolution'] = epoch_loss_list
        logger.info(f"Finished training U-Net 2D in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def evaluate(self, dataset, print_to_logger=True, save_path=None):
        """
        Evaluate the network with the given dataset. The evaluation score is given for the 3D prediction.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It must return an input image, a
            |           target binary mask, the volume ID and the slice number.
            |---- print_to_logger (bool) whether to print information to the logger.
            |---- save_path (str) the folder path where to save segemntation map (as bitmap for each slice) and the
            |           preformance dataframe. If not provided (i.e. None) nothing is saved.
        OUTPUT
            |---- None
        """
        if print_to_logger:
            logger = logging.getLogger()

        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        # put net on device
        self.unet = self.unet.to(self.device)

        # Evaluation
        if print_to_logger:
            logger.info('Start evaluating the U-Net 2.5D.')
        start_time = time.time()
        id_pred = {'volID':[], 'slice':[], 'label':[], 'TP':[], 'TN':[], 'FP':[], 'FN':[], 'pred_fn':[]} # Placeholder for each 2D prediction scores
        self.unet.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data on device
                input, target, ID, slice_nbr = data
                input = input.to(self.device).float()
                target = target.to(self.device).float()
                # make prediction
                pred = self.unet(input)
                # threshold to binarize
                pred = torch.where(pred >= 0.5, torch.ones_like(pred, device=self.device), torch.zeros_like(pred, device=self.device))
                # get confusion matrix for each slice of each volumes
                tn, fp, fn, tp = batch_binary_confusion_matrix(pred, target)
                # save prediction if required
                if save_path:
                    pred_path = []
                    for id, s_nbr, pred_samp in zip(ID, slice_nbr, pred):
                        # save slice prediction if required
                        os.makedirs(os.path.join(save_path, f'{id}/'), exist_ok=True)
                        io.imsave(os.path.join(save_path, f'{id}/{s_nbr}.bmp'), pred_samp[0,:,:].cpu().numpy().astype(np.uint8)*255, check_contrast=False) # image are binary --> put in uint8 and scale to 255
                        pred_path.append(f'{id}/{s_nbr}.bmp') # file name with volume and slice number
                else:
                    pred_path = ['-']*ID.shape[0]
                # add data to placeholder
                id_pred['volID'] += ID.cpu().tolist()
                id_pred['slice'] += slice_nbr.cpu().tolist()
                id_pred['label'] += target.view(target.shape[0], -1).max(dim=1)[0].cpu().tolist() # 1 if mask has some positive, else 0
                id_pred['TP'] += tp.cpu().tolist()
                id_pred['TN'] += tn.cpu().tolist()
                id_pred['FP'] += fp.cpu().tolist()
                id_pred['FN'] += fn.cpu().tolist()
                id_pred['pred_fn'] += pred_path

                if self.print_progress:
                    print_progessbar(b, len(loader), Name='\t\tEvaluation Batch', Size=40, erase=True)

        # make DataFrame from ID_pred to compute Dice score per image and per volume
        result_df = pd.DataFrame(id_pred)

        # compute Dice per Slice + save DF if required
        result_df['Dice'] = (2*result_df.TP + 1) / (2*result_df.TP + result_df.FP + result_df.FN + 1)
        if save_path:
            result_df.to_csv(os.path.join(save_path, 'slice_prediction_scores.csv'))

        # aggregate by patient TP/TN/FP/FN (sum) + recompute 3D Dice & Jaccard then take mean and return values
        result_3D_df = result_df[['volID', 'label', 'TP', 'TN', 'FP', 'FN']].groupby('volID').agg({'label':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})
        result_3D_df['Dice'] = (2*result_3D_df.TP + 1) / (2*result_3D_df.TP + result_3D_df.FP + result_3D_df.FN + 1)
        if save_path:
            result_3D_df.to_csv(os.path.join(save_path, 'volume_prediction_scores.csv'))

        # take average over positive volumes only and all together
        avg_results_ICH = result_3D_df.loc[result_3D_df.label == 1, 'Dice'].mean(axis=0)
        avg_results = result_3D_df.Dice.mean(axis=0)
        self.outputs['eval']['time'] = time.time() - start_time
        self.outputs['eval']['dice'] = {'all':avg_results, 'positive':avg_results_ICH}

        if print_to_logger:
            logger.info(f"Evaluation time: {timedelta(seconds=int(self.outputs['eval']['time']))}")
            logger.info(f"Evaluation Dice: {self.outputs['eval']['dice']['all']:.5f}.")
            logger.info(f"Evaluation Dice (Positive only): {self.outputs['eval']['dice']['positive']:.5f}.")
            logger.info("Finished evaluating the U-Net 2.5D.")

    def segement_volume(self, vol, save_fn=None, window=None, input_size=(256, 256), return_pred=False):
        """
        Segement each slice of the passed Nifti volumes and save the results as a Nifti volumes.
        ----------
        INPUT
            |---- vol (nibabel.nifti1.Nifti1Pair) the nibabel volumes with metadata to segement.
            |---- save_fn (str) where to save the segmentation.
            |---- window (tuple (center, width)) the winowing to apply to the ct-scan.
            |---- input_size (tuple (h, w)) the input size for the network.
            |---- return_pred (bool) whether to return the volume of prediction.
        OUTPUT
            |---- (mask_vol) (nibabel.nifti1.Nifti1Pair) the prediction volume.
        """
        pred_list = []
        vol_data = np.rot90(vol.get_fdata(), axes=(0,1)) # 90° counterclockwise rotation
        if window:
            vol_data = window_ct(vol_data, win_center=window[0], win_width=window[1], out_range=(0,1))
        transform = tf.Compose(tf.Resize(H=input_size[0], W=input_size[1]), tf.ToTorchTensor())
        self.unet.eval()
        with torch.no_grad():
            for s in range(0, vol_data.shape[2], self.batch_size):
                # get slice in good size and as tensor
                input = transform(vol_data[:,:,s:s+self.batch_size]).to(self.device).float().permute(3,0,1,2)
                # predict
                pred = self.unet(input)
                pred = torch.where(pred >= 0.5, torch.ones_like(pred, device=self.device), torch.zeros_like(pred, device=self.device))
                # store pred (B x H x W)
                pred_list.append(pred.squeeze(dim=1).permute(1,2,0).cpu().numpy().astype(np.uint8)*255)
                if self.print_progress:
                    print_progessbar(s+pred.shape[0]-1, Max=vol_data.shape[2], Name='Slice', Size=20, erase=True)

        # make the prediction volume
        vol_pred = np.concatenate(pred_list, axis=2)
        # resize to input size and rotate 90° clockwise
        vol_pred = np.rot90(skimage.transform.resize(vol_pred, (vol.header['dim'][1], vol.header['dim'][2]), order=0), axes=(1,0))
        # make Nifty and save it
        vol_pred_nii = nib.Nifti1Pair(vol_pred.astype(np.uint8), vol.affine)
        if save_fn:
            nib.save(vol_pred_nii, save_fn)
        # return Nifti prediction
        if return_pred:
            return vol_pred_nii

    def transfer_weights(self, init_state_dict, verbose=False):
        """
        Initialize the network with the weights in the provided state dictionnary. The transfer is performed on the
        matching keys of the provided state_dict and the network state_dict.
        ----------
        INPUT
            |---- init_state_dict (dict (module:params)) the weights to initiliatize the network with. Only the matching
            |           keys of the state_dict dictionnary will be transferred.
            |---- verbose (bool) whether to display some summary of the transfer.
        OUTPUT
            |---- None
        """
        # get U-Net state dict
        unet_dict = self.unet.state_dict()
        # get common keys
        to_transfer_state_dict = {k:w for k, w in init_state_dict.items() if k in unet_dict}
        if verbose:
            logger = logging.getLogger()
            logger.info(f'{len(to_transfer_state_dict)} matching weight keys found on {len(init_state_dict)} to be tranferred to the U-Net ({len(unet_dict)} weight keys).')
        # update U-Net weights
        unet_dict.update(to_transfer_state_dict)
        self.unet.load_state_dict(unet_dict)

    def save_model(self, export_fn):
        """
        Save the model.
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save(self.unet.state_dict(), export_fn)

    def load_model(self, import_fn, map_location='cpu'):
        """
        Load a model from the given path.
        ----------
        INPUT
            |---- import_fn (str) path where to get the model.
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        loaded_state_dict = torch.load(import_fn, map_location=map_location)
        self.unet.load_state_dict(loaded_state_dict)

    def save_outputs(self, export_fn):
        """
        Save the outputs in JSON.
        ----------
        INPUT
            |---- export_fn (str) path where to get the results.
        OUTPUT
            |---- None
        """
        with open(export_fn, 'w') as fn:
            json.dump(self.outputs, fn)
