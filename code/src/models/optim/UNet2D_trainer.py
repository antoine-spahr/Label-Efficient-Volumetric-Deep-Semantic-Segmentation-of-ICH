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
import time
import logging
import os
from datetime import timedelta

#from sklearn.metrics import confusion_matrix
from src.models.optim.LossFunctions import BinaryDiceLoss
from src.utils.tensor_utils import batch_binary_confusion_matrix
from src.utils.print_utils import print_progessbar

class UNet2D_trainer:
    """
    Utility class to train and evaluate a 2D UNet architecture for binary segmentation. The evaluation mehtod compute
    the performances (Dice and IoU) for the 3D volume predictions.
    """
    def __init__(self, n_epoch=150, batch_size=16, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.95), loss_fn=BinaryDiceLoss, loss_fn_kwargs=dict(reduction='mean'),
        weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False):
        """
        Build a UNet2D_trainer object.
        ----------
        INPUT
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
            |---- UNet_trainer () the trainer for the 2D UNet.
        """
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
        self.train_time = None
        self.train_evolution = None
        self.eval_time = None
        self.eval_dice = None # avg Volumetric Dice
        self.eval_IoU = None # avg Volumetric IoU

    def train(self, net, dataset, valid_dataset=None, checkpoint_path=None):
        """
        Train the passed network with the given dataset(s).
        ----------
        INPUT
            |---- net (nn.Module) the network architecture to train. Has to output a tensor of shape similar as input
            |           (logit for each pixel as outputed by the sigmoid.)
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
        net = net.to(self.device)

        # define optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Load checkpoint if present
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            n_epoch_finished = checkpoint['n_epoch_finished']
            net.load_state_dict(checkpoint['net_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            epoch_loss_list = checkpoint['loss_evolution']
            logger.info(f'Checkpoint loaded with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            logger.info('No Checkpoint found. Training from begining.')
            n_epoch_finished = 0
            epoch_loss_list = [] # Placeholder for epoch evolution

        # define the lr scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        # define the loss function
        loss_fn = self.loss_fn(**self.loss_fn_kwargs)

        # start training
        logger.info('Start training the U-Net 2D.')
        start_time = time.time()
        n_batch = len(train_loader)

        for epoch in range(n_epoch_finished, self.n_epoch):
            net.train()
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
                pred = net(input)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # print process
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=40, erase=True)

            # Get validation performance if required
            valid_dice, dice, IoU = '', None, None
            if valid_dataset:
                dice, IoU = self.evaluate(net, valid_dataset, return_score=True, print_to_logger=False, save_path=None)
                valid_dice = f'| Valid ICH Dice: {dice:.5f} | Valid ICH IoU: {IoU:.5f} '

            # log the epoch statistics
            logger.info(f'\t| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train time: {time.time() - epoch_start_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} ' + valid_dice +
                        f'| lr: {scheduler.get_last_lr()[0]:.7f} |')
            # Store epoch loss and epoch dice
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, dice])

            # update scheduler
            scheduler.step()

            # Save Checkpoint every 10 epochs
            if (epoch+1) % 10 == 0 and checkpoint_path:
                checkpoint = {'n_epoch_finished': epoch+1,
                              'net_state': net.state_dict(),
                              'optimizer_state': optimizer.state_dict(),
                              'loss_evolution': epoch_loss_list}
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

        # End training
        self.train_time = time.time() - start_time
        self.train_evolution = epoch_loss_list
        logger.info(f'Finished training U-Net 2D in {str(timedelta(seconds=self.train_time)):0>8}')

        return net

    def evaluate(self, net, dataset, return_score=False, print_to_logger=True, save_path=None):
        """
        Evaluate the network with the given dataset. The evaluation score is given for the 3D prediction.
        ----------
        INPUT
            |---- net (nn.Module) the network architecture to train. Has to output a tensor of shape similar as input
            |           (logit for each pixel as outputed by the sigmoid.)
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It must return an input image, a
            |           target binary mask, the patientID and the slice number.
            |---- return_score (bool) whether to return the mean Dice and mean IoU scores of 3D segmentation (for
            |           the 2D case the Dice is computed on the concatenation of prediction for a patient).
            |---- print_to_logger (bool) whether to print information to the logger.
            |---- save_path (str) the folder path where to save segemntation map (as bitmap for each slice) and the
            |           preformance dataframe. If not provided (i.e. None) nothing is saved.
        OUTPUT
            |---- (Dice) (float) the average Dice coefficient for the 3D segemntation.
            |---- (IoU) (flaot) the average IoU coefficient for the 3D segementation.
        """
        if print_to_logger:
            logger = logging.getLogger()

        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers)
        # put net on device
        net = net.to(self.device)

        # Evaluation
        if print_to_logger:
            logger.info('Start evaluating the UNet2D.')
        start_time = time.time()
        id_pred = {'PatientID':[], 'Slice':[], 'ICH':[], 'TP':[], 'TN':[], 'FP':[], 'FN':[], 'pred_fn':[]} # Placeholder for each 2D prediction scores
        net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data on device
                input, target, pID, slice_nbr = data
                input = input.to(self.device).float()
                target = target.to(self.device).float()
                # make prediction
                pred = net(input)
                # threshold to binarize
                pred = torch.where(pred >= 0.5, torch.ones_like(pred, device=self.device), torch.zeros_like(pred, device=self.device))
                # get confusion matrix for each slice of each patient
                tn, fp, fn, tp = batch_binary_confusion_matrix(pred, target)
                # save prediction if required
                if save_path:
                    pred_path = []
                    for id, s_nbr, pred_samp in zip(pID, slice_nbr, pred):
                        # save slice prediction if required
                        os.makedirs(f'{save_path}/{id}/', exist_ok=True)
                        io.imsave(f'{save_path}/{id}/{s_nbr}.bmp', pred_samp[0,:,:].cpu().numpy().astype(np.uint8)*255, check_contrast=False) # image are binary --> put in uint8 and scale to 255
                        pred_path.append(f'{id}/{s_nbr}.bmp') # file name with patient and slice number
                else:
                    pred_path = ['']*pID.shape[0]
                # add data to placeholder
                id_pred['PatientID'] += pID.cpu().tolist()
                id_pred['Slice'] += slice_nbr.cpu().tolist()
                id_pred['ICH'] += target.view(target.shape[0], -1).max(dim=1) # 1 if mask has some ICH, else 0
                id_pred['TP'] += tp.cpu().tolist()
                id_pred['TN'] += tn.cpu().tolist()
                id_pred['FP'] += fp.cpu().tolist()
                id_pred['FN'] += fn.cpu().tolist()
                id_pred['pred_fn'] += pred_path

                if self.print_progress:
                    print_progessbar(b, len(loader), Name='\t\tEvaluation Batch', Size=40, erase=True)

        # make DataFrame from ID_pred to compute Dice score per image and per volume
        result_df = pd.DataFrame(id_pred)

        # compute Dice & Jaccard (IoU) per Slice + save DF if required
        result_df['Dice'] = (2*result_df.TP + 1) / (2*result_df.TP + result_df.FP + result_df.FN + 1)
        result_df['IoU'] = (result_df.TP + 1) / (result_df.TP + result_df.FP + result_df.FN + 1)
        if save_path:
            result_df.to_csv(f'{save_path}/slice_prediction_scores.csv')

        # aggregate by patient TP/TN/FP/FN (sum) + recompute 3D Dice & Jaccard then take mean and return values
        result_3D_df = result_df[['PatientID', 'ICH', 'TP', 'TN', 'FP', 'FN']].groupby('PatientID').agg({'ICH':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})
        result_3D_df['Dice'] = (2*result_3D_df.TP + 1) / (2*result_3D_df.TP + result_3D_df.FP + result_3D_df.FN + 1)
        result_3D_df['IoU'] = (result_3D_df.TP + 1) / (result_3D_df.TP + result_3D_df.FP + result_3D_df.FN + 1)
        if save_path:
            result_3D_df.to_csv(f'{save_path}/volume_prediction_scores.csv')

        # take average over ICH-positive patient only
        avg_results = result_3D_df[result_3D_df.ICH == 1, ['Dice', 'IoU']].mean(axis=0)
        self.eval_time = time.time() - start_time
        self.eval_dice = avg_results.Dice
        self.eval_IoU = avg_results.IoU

        if print_to_logger:
            logger.info(f'Evaluation time: {str(timedelta(seconds=self.eval_time)):0>8}')
            logger.info(f'Evaluation Dice: {self.eval_dice:.5f}.')
            logger.info(f'Evaluation IoU: {self.eval_IoU:.5f}.')
            logger.info('Finished evaluating the U-Net 2D.')

        if return_score:
            return avg_results.Dice, avg_results.IoU
