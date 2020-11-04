"""
author: Antoine Spahr

date : 27.10.2020

----------

TO DO :
- check if need to normalize output feature_maps in local contrastive task.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
import os
import warnings
import json
from datetime import timedelta

from sklearn.manifold import TSNE
from src.models.optim.LossFunctions import InfoNCELoss, LocalInfoNCELoss
from src.utils.print_utils import print_progessbar
from src.utils.python_utils import rgetattr

class Contrastive:
    """
    Class defining model to train on the global or local contrastive task. In the globale case, the batch is used as
    comparison set (similarly as Chen et al. 2019). In Local, the comparison is made between region of the feature map
    of the decoder (negatives are other region of the feature map) (similar as in Chaitanya et al. 2020).
    """
    def __init__(self, net, n_epoch=150, batch_size=32, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
                 lr_scheduler_kwargs=dict(gamma=0.95), loss_fn=InfoNCELoss, loss_fn_kwargs=dict(set_size=32, tau=0.1, device='cuda'),
                 weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False, is_global=True):
        """
        Build a Contrastive model for the global or local contrastive task .
        ----------
        INPUT
            |---- net (nn.Module) the network to train. It should be a Encoder architecture outputing a N-dimensional
            |           representation of the input image for a Global contrastive training, or an Encoder followed by a
            |           partial decoder for a Local contrastive training.
            |---- n_epoch (int) the number of epoch for the training.
            |---- batch_size (int) the batch size to use for loading the data.
            |---- lr (float) the learning rate.
            |---- lr_scheduler (torch.optim.lr_scheduler) the learning rate evolution scheme to use.
            |---- lr_scheduler_kwargs (dict) the keyword arguments to be passed to the lr_scheduler.
            |---- loss_fn (nn.Module) the loss function to use. It should take two representation as input of the forward
            |           pass. In Global case, the loss takes 2 representation B x N. In the Local case, the loss takes 2
            |           feature map representation (B x H x W x C)
            |---- loss_fn_kwargs (dict) the keyword arguments to pass to the loss function constructor.
            |---- weight_decay (float) the L2-regularization strenght.
            |---- num_workers (int) the number of workers to use for data loading.
            |---- device (str) the device to use.
            |---- print_progress (bool) whether to print progress bar for batch processing.
            |---- is_global (bool) whether the task is global contrastive or local contrastive.
        OUTPUT
            |---- ContrastiveGlobal () the model for the Contrastive learning task of encoder.
        """
        self.net = net
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
        self.is_global = is_global
        # put network to device
        self.net = self.net.to(self.device)

        self.outputs = {
            "train": {
                "time": None,
                "evolution": None
            },
            "eval":{
                "time": None,
                "repr": None
            }
        }

    def train(self, dataset, checkpoint_path=None):
        """
        Train the network on the given dataset for the contrastive task (global or local).
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It should output two augmented
            |           version of the same image as well as the image index.
            |---- checkpoint_path (str) the filename for a possible checkpoint to start the training from. If None, the
            |           network's weights are not saved regularily during training.
        OUTPUT
            |---- None.
        """
        logger = logging.getLogger()
        # initialize dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        # initialize loss function
        loss_fn = self.loss_fn(**self.loss_fn_kwargs)
        # initialize otpitimizer
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # initialize scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        # load checkpoint if any
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            n_epoch_finished = checkpoint['n_epoch_finished']
            self.net.load_state_dict(checkpoint['net_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['lr_state'])
            epoch_loss_list = checkpoint['loss_evolution']
            logger.info(f'Checkpoint loaded with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            logger.info('No Checkpoint found. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = []
        # Train Loop
        logger.info(f"Start trianing the network on the {'global' if self.is_global else 'local'} contrastive task.")
        start_time = time.time()
        n_batch = len(loader)
        for epoch in range(n_epoch_finished, self.n_epoch):
            self.net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(loader):
                # get data
                im1, im2, _ = data
                im1 = im1.to(self.device).float().requires_grad_(True)
                im2 = im2.to(self.device).float().requires_grad_(True)
                # zeros gradient
                optimizer.zero_grad()
                # get image representations
                z1 = self.net(im1)
                z2 = self.net(im2)
                # normalize representations
                if self.is_global:
                    z1 = nn.functional.normalize(z1, dim=1)
                    z2 = nn.functional.normalize(z2, dim=1)
                # compute loss and backpropagate
                loss = loss_fn(z1, z2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # print progress
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=40, erase=True)

            # print epoch statistics
            logger.info(f'\t| Epoch : {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train time: {timedelta(seconds=int(time.time() - epoch_start_time))} '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} '
                        f'| lr: {scheduler.get_last_lr()[0]:.7f} |')
            # store epoch loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch])
            # update lr
            scheduler.step()
            # save checkpoint if needed
            if (epoch+1)%1 == 0 and checkpoint_path:
                checkpoint = {'n_epoch_finished': epoch+1,
                              'net_state': self.net.state_dict(),
                              'optimizer_state': optimizer.state_dict(),
                              'lr_state': scheduler.state_dict(),
                              'loss_evolution': epoch_loss_list}
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

        # End training
        self.outputs['train']['time'] = time.time() - start_time
        self.outputs['train']['evolution'] = epoch_loss_list
        logger.info(f"Finished training on the network on the {'global' if self.is_global else 'local'} contrastive task in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def evaluate(self, dataset):
        """
        Evaluate the network on the given dataset for the Contrastive task (get t-SNE representation of samples). Only if global task.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for evaluation. It should output the original image,
            |           and the sample index.
        OUTPUT
            |---- None
        """
        if self.is_global:
            logger = logging.getLogger()
            # initiliatize Dataloader
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            # Evaluate
            logger.info("Start Evaluating the network on global contrastive task.")
            start_time = time.time()
            idx_repr = [] # placeholder for bottleneck representation
            n_batch = len(loader)
            self.net.eval()
            self.net.return_bottleneck = True
            with torch.no_grad():
                for b, data in enumerate(loader):
                    im, idx = data
                    im = im.to(self.device).float()
                    idx = idx.to(self.device)
                    # get representations
                    _, z = self.net(im)
                    # keep representations (bottleneck)
                    idx_repr += list(zip(idx.cpu().data.tolist(), z.squeeze().cpu().data.tolist()))
                    # print_progress
                    if self.print_progress:
                        print_progessbar(b, n_batch, Name='\t\tEvaluation Batch', Size=40, erase=True)
                # reset the network attriubtes
                self.net.return_bottleneck = False
            # compute tSNE for representation
            idx, repr = zip(*idx_repr)
            repr = np.array(repr)
            logger.info('Computing the t-SNE representation.')
            repr_2D = TSNE(n_components=2).fit_transform(repr)
            self.outputs['eval']['repr'] = list(zip(idx, repr_2D.tolist()))
            logger.info('Succesfully computed the t-SNE representation.')
            # finish evluation
            self.outputs['eval']['time'] = time.time() - start_time
            logger.info(f"Finished evaluating of encoder on the global contrastive task in {timedelta(seconds=int(self.outputs['eval']['time']))}")
        else:
            warnings.warn("Evaluation is only possible with a global contrastive task.")

    def transfer_weights(self, init_state_dict, verbose=False, freeze=False):
        """
        Initialize the network with the weights in the provided state dictionnary. The transfer is performed on the
        matching keys of the provided state_dict and the network state_dict.
        ----------
        INPUT
            |---- init_state_dict (dict (module:params)) the weights to initiliatize the network with. Only the matching
            |           keys of the state_dict dictionnary will be transferred.
            |---- verbose (bool) whether to display some summary of the transfer.
            |---- freeze (bool) whether to freeze the transferred weigths.
        OUTPUT
            |---- None
        """
        # get Net state dict
        net_dict = self.net.state_dict()
        # get common keys
        to_transfer_state_dict = {k:w for k, w in init_state_dict.items() if k in net_dict}
        if verbose:
            logger = logging.getLogger()
            logger.info(f'{len(to_transfer_state_dict)} matching weight keys found on {len(init_state_dict)} to be tranferred to the net ({len(net_dict)} weight keys).')
        # update U-Net weights
        net_dict.update(to_transfer_state_dict)
        self.net.load_state_dict(net_dict)
        # freeze the transferred weights on the network
        if freeze:
            for module_key in to_transfer_state_dict.keys():
                rgetattr(self.net, module_key).requires_grad = False

    def get_state_dict(self):
        """
        Return the model weights as State dictionnary.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- stat_dict (dict) the model's weights.
        """
        return self.net.state_dict()

    def save_model(self, export_fn):
        """
        Save the model.
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save(self.net.state_dict(), export_fn)

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
        self.net.load_state_dict(loaded_state_dict)

    def save_outputs(self, export_fn):
        """
        Save the training stats in JSON.
        ----------
        INPUT
            |---- export_fn (str) path where to get the results.
        OUTPUT
            |---- None
        """
        with open(export_fn, 'w') as fn:
            json.dump(self.outputs, fn)
