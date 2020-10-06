"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :
- check if use of validation set to keep best model ?
- adjust data unpacking depending on dataset we have ! <-- OK
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import PIL.Image
import time
import logging

from sklearn.metrics import confusion_matrix

from src.models.optim.loss_functions.DiceLoss import BinaryDiceLoss
from src.utils.print_utils import print_progessbar

class UNet_trainer:
    """
    Utility class to train and evaluate a 2D or 3D UNet architecture. The evaluation mehtod compute the performances
    (Dice and IoU) for the 3D predictions.
    """
    def __init__(self, use_3D, n_epoch=150, batch_size=16, lr=1e-3, lr_scheduler=optim.lr_scheduler.MultiplicativeLR,
        lr_scheduler_kwargs=dict(lr_lambda=lambda ep: 0.95*ep), loss_fn=BinaryDiceLoss, loss_fn_kwargs=dict(reduction='mean'),
        weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False):
        """
        Build a UNet_trainer object.
        ----------
        INPUT
            |---- use_3D (bool) whether the UNet process images or volumes.
            |---- n_epoch (int) the number of epoch for the training.
            |---- batch_size (int) the batch size to use for loading the data.
            |---- lr (float) the learning rate.
            |---- lr_scheduler (torch.optim.lr_scheduler) the learning rate evolution scheme to use.
            |---- lr_scheduler_kwargs (dict) the keyword arguments to be passed to the lr_scheduler.
            |---- loss_fn (nn.Module) the loss function to use.
            |---- loss_fn_kwargs (dict) the keyword arguments to pass to the loss function constructor.
            |---- weight_decay (float) the L2-regularization strenght.
            |---- num_workers (int) the number of workers to use for data loading.
            |---- device (str) the device to use.
            |---- print_progress (bool) whether to print progress bar for batch processing.
        OUTPUT
            |---- UNet_trainer () the trainer for the UNet (2D or 3D)
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

        self.use_3D = use_3D

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
            |---- net (nn.Module) the network architecture to train.
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It must return an input image, a
            |           target binary mask, the patientID (and the slice number for 2D).
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
            logger.info(f'Checkpoint loaded with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            logger.info('No Checkpoint found. Training from begining.')
            n_epoch_finished = 0

        # define the lr scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        # define the loss function
        loss_fn = self.loss_fn(**self.loss_fn_kwargs)

        # start training
        logger.info('Start training the UNet.')
        start_time = time.time()
        epoch_loss_list = [] # Placeholder for epoch evolution
        n_batch = len(train_loader)

        for epoch in range(n_epoch_finished, self.n_epoch):
            net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get data
                if self.use_3D:
                    input, target, _ = data
                else:
                    input, target, _, _ = data
                # put data tensors on device
                input = input.to(self.device).float().requires_grad_(True)
                target = target.to(self.device)
                # zero the networks' gradients
                optimizer.zero_grad()
                # optimize weights with backpropagation on the batch
                pred = net(input).argmax(dim=1)
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
                valid_dice = f'| Valid Dice: {dice:.3%} | Valid IoU: {IoU:.3%} '

            # log the epoch statistics
            logger.info(f'\t| Epoch: {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train time: {time.time() - epoch_start_time:.3f} [s] '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} ' + valid_dice
                        f'| lr: {scheduler.get_lr()[0]:g} |')
            # Store epoch loss and epoch dice
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, dice])

            # update scheduler
            scheduler.step()

            # Save Checkpoint every 10 epochs
            if (epoch+1) % 10 == 0 and checkpoint_path:
                checkpoint = {'n_epoch_finished': epoch+1,
                              'net_state': net.state_dict(),
                              'optimizer_state': optimizer.state_dict()}
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

            # End training
            self.train_time = time.time() - start_time
            self.train_evolution = epoch_loss_list
            logger.info(f'Finished training UNet in {self.train_time:.3f} [s].')

            return net

        def evaluate(self, net, dataset, return_score=False, print_to_logger=True, save_path=None):
            """
            Evaluate the network with the given dataset. The evaluation score is given for the 3D prediction.
            ----------
            INPUT
                |---- net (nn.Module) the network architecture to train.
                |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It must return an input image, a
                |           target binary mask, the patientID (and the slice number for 2D).
                |---- return_score (bool) whether to return the mean Dice and mean IoU scores of 3D segmentation (for
                |           the 2D case the Dice is computed on the concatenation of prediction for a patient).
                |---- print_to_logger (bool) whether to print information to the logger.
                |---- save_path (str) the folder path where to save segemntation map (as bitmap for each slice) and the
                |           preformance dataframe. If not provided (i.e. None) nothing is saved.
            OUTPUT
                |---- (Dice) (float) the average Dice coefficient for the 3D segemntation.
                |---- (IoU) (flaot) the average IoU coefficient for the 3D segementation.
            """
            # manage to save predictions if save path is given (dataset give patient ID and slice number as well)
            # Need to report Dice for 3D input (even if 2D model) --> need to rearange 2D prediction

            if print_to_logger:
                logger = logging.getLogger()

            # make dataloader
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers)
            # put net on device
            net = net.to(self.device)

            # Evaluation
            if print_to_logger:
                logger.info('Start evaluating the UNet.')
            start_time = time.time()
            id_pred = [] # Placeholder for each 2D prediction scores

            net.eval()
            with torch.no_grad():
                for b, data in enumerate(loader):
                    # get data on device
                    if self.use_3D:
                        input, target, pID = data
                    else:
                        input, target, pID, slice_nbr = data
                    input = input.to(self.device).float()
                    target = target.to(self.device)
                    # make prediction
                    pred = net(input).argmax(dim=1)

                    # get confusion matrix for each slice of each samples
                    if self.use_3D: # <--- to be revised (save in nifty --> no need to iterate slices)
                        # decompose volume in slice and treat process them separately
                        for id, target_samp, pred_samp in zip(pID, target, pred): # iterate over batch
                            for s_nbr in range(target_samp.shape[-1]): # iterate over slices
                                tn, fp, fn, tp = confusion_matrix(target_samp[:,:,:,s_nbr].cpu().data.numpy().ravel(),
                                                                  pred_samp[:,:,:,s_nbr].cpu().data.numpy().ravel()).ravel()
                                # save slice prediction if required
                                if save_path:
                                    im = PIL.Image.fromarray(pred_samp.cpu().numpy().astype(bool))
                                    pred_path = f'{save_path}/{id}/{s_nbr+1}.bmp'
                                    im.save(pred_path)
                                else:
                                    pred_path = 'None

                                # add to list
                                id_pred.append({'PatientID':id.cpu(), 'Slice':i, 'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn, 'pred_fn':None})
                    else:
                        for id, s_nbr, target_samp, pred_samp in zip(pID, slice_nbr, target, pred):
                            tn, fp, fn, tp = confusion_matrix(target_samp.cpu().data.numpy().ravel(),
                                                              pred_samp.cpu().data.numpy().ravel()).ravel()
                            # save slice prediction if required
                            if save_path:
                                im = PIL.Image.fromarray(pred_samp.cpu().numpy().astype(bool))
                                pred_path = f'{save_path}/{id}/{s_nbr}.bmp'
                                im.save(pred_path)
                            else:
                                pred_path = 'None'
                            # add to list
                            id_pred.append({'PatientID':id.cpu(), 'Slice':s_nbr.cpu(), 'TP':tp, 'TN':tn, 'FP':fp, 'FN':fn, 'pred_fn':None})

                    if self.print_progress:
                        print_progessbar(b, len(loader), Name='\t\tEvaluation Batch', Size=40, erase=True)

            # make DataFrame from ID_pred to compute Dice score per image and per volume
            result_df = pd.DataFrame(id_pred)

            # compute Dice & Jaccard (IoU) per Slice + save DF if required
            result_df['Dice'] = (2*result_df.TP + 1e-9) / (2*result_df.TP + result_df.FP + result_df.FN + 1e-9)
            result_df['IoU'] = (result_df.TP + 1e-9) / (result_df.TP + result_df.FP + result_df.FN + 1e-9)
            if save_path:
                result_df.to_csv(f'{save_path}/prediction_scores.csv')

            # aggregate by patient TP/TN/FP/FN (sum) + recompute 3D Dice & Jaccard then take mean and return values
            result_3D_df = result_df[['PatientID', 'TP', 'TN', 'FP', 'FN']].groupby('PatientID').sum()
            result_3D_df['Dice'] = (2*result_3D_df.TP + 1e-9) / (2*result_3D_df.TP + result_3D_df.FP + result_3D_df.FN + 1e-9)
            result_3D_df['IoU'] = (result_3D_df.TP + 1e-9) / (result_3D_df.TP + result_3D_df.FP + result_3D_df.FN + 1e-9)
            avg_results = result_3D_df[['Dice', 'IoU']].mean(axis=0)

            self.eval_time = time.time() - start_time()
            self.eval_dice = avg_results.Dice
            self.eval_IoU = avg_result.IoU

            if print_to_logger:
                logger.info(f'Evaluation time: {self.eval_time} [s].')
                logger.info(f'Evaluation Dice: {self.eval_dice:.3%}.')
                logger.info(f'Evaluation IoU: {self.eval_IoU:.3%}.')
                logger.info('Finished evaluating the UNet.')

            if return_score:
                return avg_results.Dice, avg_results.IoU
