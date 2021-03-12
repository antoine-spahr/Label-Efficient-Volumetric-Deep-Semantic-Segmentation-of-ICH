"""
author: Antoine Spahr

date : 03.12.2020

----------

TO DO :
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import pandas as pd
import skimage
import time
import logging
import os
import json
from datetime import timedelta
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.manifold import TSNE
from src.utils.print_utils import print_progessbar

class BinaryClassifier:
    """
    Define a Binary Classifier model able to return the tSNE representation of feature map prior to the MLP.
    """
    def __init__(self, net, n_epoch=150, batch_size=32, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
                 lr_scheduler_kwargs=dict(gamma=0.95), loss_fn=nn.CrossEntropyLoss, loss_fn_kwargs=dict(reduction='mean'),
                 weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False):
        """
        Build a BinaryClassifer model.
        ----------
        INPUT
            |---- net (nn.Module) the network to train. It should be a Encoder architecture returning a 2 neurons outputs.
            |           To get the tSNE representation, the network should have a boolean attribute 'return_bottleneck'
            |           that can be set to True to recover the bottleneck feature map (after average pooling before the MLP).
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
            |---- ContextRestoration () the model for the Context restoration task.
        """
        self.net = net
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
                'auc': None,
                'acc': None,
                'recall': None,
                'precision': None,
                'f1': None,
                'pred': None,
                'repr': None
            }
        }

    def train(self, dataset, valid_dataset=None, checkpoint_path=None):
        """
        Train the passed network on the given dataset for the Context retoration task.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It should output the image,
            |           the binary label, and the sample index.
            |---- valid_dataset (torch.utils.data.Dataset) the dataset to use for validation at each epoch. It should
            |           output the image, the binary label, and the sample index.
            |---- checkpoint_path (str) the filename for a possible checkpoint to start the training from. If None, the
            |           network's weights are not saved regularily during training.
        OUTPUT
            |---- None.
        """
        logger = logging.getLogger()
        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers, pin_memory=False,
                                                   worker_init_fn=lambda _: np.random.seed())
        # put net on device
        self.net = self.net.to(self.device)
        # define optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # define lr scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        # define the loss function
        loss_fn = self.loss_fn(**self.loss_fn_kwargs)
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

        # Start Training
        logger.info('Start training the Binary Classification task.')
        start_time = time.time()
        n_batch = len(train_loader)

        for epoch in range(n_epoch_finished, self.n_epoch):
            self.net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get data : target is original image, input is the corrupted image
                input, label, _ = data
                # put data on device
                input = input.to(self.device).float().requires_grad_(True)
                label = label.to(self.device).long()#.requires_grad_(True)
                # zero the networks' gradients
                optimizer.zero_grad()
                # optimize the weights with backpropagation on the batch
                pred = self.net(input)
                pred = nn.functional.softmax(pred, dim=1)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # print progress
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=50, erase=True)

            auc = None
            if valid_dataset:
                auc, acc, recall, precision, f1 = self.evaluate(valid_dataset, save_tsne=False, return_scores=True)
                valid_summary = f'| AUC {auc:.3%} | Accuracy {acc:.3%} | Recall {recall:.3%} | Precision {precision:.3%} | F1 {f1:.3%} '
            else:
                valid_summary = ''

            # Print epoch statistics
            logger.info(f'\t| Epoch : {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train time: {timedelta(seconds=int(time.time() - epoch_start_time))} '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} '
                        f'{valid_summary}'
                        f'| lr: {scheduler.get_last_lr()[0]:.7f} |')
            # Store epoch loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, auc])
            # update lr
            scheduler.step()
            # Save Checkpoint every epochs
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
        self.outputs['train']['evolution'] = {'col_name': ['Epoch', 'loss', 'Valid_AUC'],
                                              'data': epoch_loss_list}
        logger.info(f"Finished training inpainter Binary Classifier in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def evaluate(self, dataset, save_tsne=False, return_scores=False):
        """
        Evaluate the passed network on the given dataset for the Context retoration task.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for evaluation. It should output the original image,
            |           and the sample index.
            |---- save_tsne (bool) whether to compute and store in self.outputs the tsne representation of the feature map
            |           after the average pooling layer and before the MLP
            |---- return_scores (bool) whether to return the measured ROC AUC, accuracy, recall, precision and f1-score.
        OUTPUT
            |---- (auc) (float) the ROC AUC on the dataset.
            |---- (acc) (float) the accuracy on the dataset.
            |---- (recall) (float) the recall on the dataset.
            |---- (precision) (float) the precision on the dataset.
            |---- (f1) (float) the f1-score on the dataset.
        """
        logger = logging.getLogger()
        # make loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers, worker_init_fn=lambda _: np.random.seed())
        # put net on device
        self.net = self.net.to(self.device)

        # Evaluate
        start_time = time.time()
        idx_repr = [] # placeholder for bottleneck representation
        idx_label_pred = [] # placeholder for label & prediction
        n_batch = len(loader)
        self.net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data : load in standard way (no patch swaped image)
                input, label, idx = data
                input = input.to(self.device).float()
                label = label.to(self.device).float()
                idx = idx.to(self.device)
                if save_tsne:
                    # get representation
                    self.net.return_bottleneck = True
                    pred, repr = self.net(input)
                    pred = nn.functional.softmax(pred, dim=1)
                    # down sample representation for reduced memory impact
                    repr = nn.AdaptiveAvgPool2d((4,4))(repr)
                    # add ravelled representations to placeholder
                    idx_repr += list(zip(idx.cpu().data.tolist(), repr.view(repr.shape[0], -1).cpu().data.tolist()))
                    idx_label_pred += list(zip(idx.cpu().data.tolist(), label.cpu().data.tolist(), pred.argmax(dim=1).cpu().data.tolist(), pred[:,1].cpu().data.tolist())) # pred score is softmax activation of class 1
                else:
                    pred = self.net(input)
                    pred = nn.functional.softmax(pred, dim=1)
                    idx_label_pred += list(zip(idx.cpu().data.tolist(), label.cpu().data.tolist(), pred.argmax(dim=1).cpu().data.tolist(), pred[:,1].cpu().data.tolist()))
                # print_progress
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tEvaluation Batch', Size=50, erase=True)
            # reset the network attriubtes
            if save_tsne:
                self.net.return_bottleneck = False

        # compute tSNE for representation
        if save_tsne:
            idx, repr = zip(*idx_repr)
            repr = np.array(repr)
            logger.info('Computing the t-SNE representation.')
            repr_2D = TSNE(n_components=2).fit_transform(repr)
            self.outputs['eval']['repr'] = list(zip(idx, repr_2D.tolist()))
            logger.info('Succesfully computed the t-SNE representation.')

        # Compute Accuracy
        _, label, pred, pred_score = zip(*idx_label_pred)
        label, pred, pred_score = np.array(label), np.array(pred), np.array(pred_score)
        auc = roc_auc_score(label, pred_score)
        acc = accuracy_score(label, pred)
        recall = recall_score(label, pred)
        precision = precision_score(label, pred)
        f1 = f1_score(label, pred)
        self.outputs['eval']['auc'] = auc
        self.outputs['eval']['acc'] = acc
        self.outputs['eval']['recall'] = recall
        self.outputs['eval']['precision'] = precision
        self.outputs['eval']['f1'] = f1
        self.outputs['eval']['pred'] = idx_label_pred

        # finish evluation
        self.outputs['eval']['time'] = time.time() - start_time

        if return_scores:
            return auc, acc, recall, precision, f1

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
        torch.save(self.net, export_fn)

    def save_model_state_dict(self, export_fn):
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

class MultiClassifier:
    """
    Define a Multi-Label Classifier model able to return the tSNE representation of feature map prior to the MLP.
    """
    def __init__(self, net, n_epoch=150, batch_size=32, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
                 lr_scheduler_kwargs=dict(gamma=0.95), loss_fn=nn.BCEWithLogitsLoss, loss_fn_kwargs=dict(reduction='mean'),
                 weight_decay=1e-6, score_average='macro', num_workers=0, device='cuda', print_progress=False):
        """
        Build a MultiClassifer model.
        ----------
        INPUT
            |---- net (nn.Module) the network to train. It should be a Encoder architecture returning a 2 neurons outputs.
            |           To get the tSNE representation, the network should have a boolean attribute 'return_bottleneck'
            |           that can be set to True to recover the bottleneck feature map (after average pooling before the MLP).
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
            |---- MultiClassifer () the model for the Context restoration task.
        """
        self.net = net
        # training parameters
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.loss_fn = loss_fn
        self.loss_fn_kwargs = loss_fn_kwargs
        self.weight_decay = weight_decay
        self.score_average = score_average
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
                'auc': None,
                'acc': None,
                'subset_acc': None,
                'recall': None,
                'precision': None,
                'f1': None,
                'pred': None,
                'repr': None
            }
        }

    def train(self, dataset, valid_dataset=None, checkpoint_path=None):
        """
        Train the passed network on the given dataset for the Multi classification task.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for training. It should output the image,
            |           the multi label, and the sample index.
            |---- valid_dataset (torch.utils.data.Dataset) the dataset to use for validation at each epoch. It should
            |           output the image, the binary label, and the sample index.
            |---- checkpoint_path (str) the filename for a possible checkpoint to start the training from. If None, the
            |           network's weights are not saved regularily during training.
        OUTPUT
            |---- None.
        """
        logger = logging.getLogger()
        # make dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers, pin_memory=False,
                                                   worker_init_fn=lambda _: np.random.seed())
        # put net on device
        self.net = self.net.to(self.device)
        # define optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # define lr scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        # define the loss function
        loss_fn = self.loss_fn(**self.loss_fn_kwargs)
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

        # Start Training
        logger.info('Start training the Multi Classification task.')
        start_time = time.time()
        n_batch = len(train_loader)

        for epoch in range(n_epoch_finished, self.n_epoch):
            self.net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(train_loader):
                # get data : target is original image, input is the corrupted image
                input, label, _ = data
                # put data on device
                input = input.to(self.device).float().requires_grad_(True)
                label = label.to(self.device).float()#long()#.requires_grad_(True)
                # zero the networks' gradients
                optimizer.zero_grad()
                # optimize the weights with backpropagation on the batch
                pred = self.net(input)
                pred = torch.sigmoid(input)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # print progress
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tTrain Batch', Size=50, erase=True)

            if valid_dataset:
                auc, acc, sub_acc, recall, precision, f1 = self.evaluate(valid_dataset, save_tsne=False, return_scores=True)
                valid_summary = f'| AUC {auc:.3%} | Accuracy {acc:.3%} | Subset-Accuracy {sub_acc:.3%} | Recall {recall:.3%} | Precision {precision:.3%} | F1 {f1:.3%} '
            else:
                valid_summary = ''

            # Print epoch statistics
            logger.info(f'\t| Epoch : {epoch + 1:03}/{self.n_epoch:03} '
                        f'| Train time: {timedelta(seconds=int(time.time() - epoch_start_time))} '
                        f'| Train Loss: {epoch_loss / n_batch:.6f} '
                        f'{valid_summary}'
                        f'| lr: {scheduler.get_last_lr()[0]:.7f} |')
            # Store epoch loss
            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, auc])
            # update lr
            scheduler.step()
            # Save Checkpoint every epochs
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
        self.outputs['train']['evolution'] = {'col_name': ['Epoch', 'loss', 'Valid_AUC'],
                                              'data': epoch_loss_list}
        logger.info(f"Finished training inpainter Binary Classifier in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def evaluate(self, dataset, save_tsne=False, return_scores=False):
        """
        Evaluate the passed network on the given dataset for the Context retoration task.
        ----------
        INPUT
            |---- dataset (torch.utils.data.Dataset) the dataset to use for evaluation. It should output the original image,
            |           and the sample index.
            |---- save_tsne (bool) whether to compute and store in self.outputs the tsne representation of the feature map
            |           after the average pooling layer and before the MLP
            |---- return_scores (bool) whether to return the measured ROC AUC, accuracy, recall, precision and f1-score.
        OUTPUT
            |---- (auc) (float) the ROC AUC on the dataset.
            |---- (acc) (float) the accuracy on the dataset.
            |---- (recall) (float) the recall on the dataset.
            |---- (precision) (float) the precision on the dataset.
            |---- (f1) (float) the f1-score on the dataset.
        """
        logger = logging.getLogger()
        # make loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                             num_workers=self.num_workers, worker_init_fn=lambda _: np.random.seed())
        # put net on device
        self.net = self.net.to(self.device)

        # Evaluate
        start_time = time.time()
        idx_repr = [] # placeholder for bottleneck representation
        idx_label_pred = [] # placeholder for label & prediction
        n_batch = len(loader)
        self.net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                # get data : load in standard way (no patch swaped image)
                input, label, idx = data
                input = input.to(self.device).float()
                label = label.to(self.device).float()
                idx = idx.to(self.device)
                if save_tsne:
                    # get representation
                    self.net.return_bottleneck = True
                    pred_score, repr = self.net(input)
                    pred_score = torch.sigmoid(pred_score)
                    pred = torch.where(pred_score > 0.5, torch.ones_like(pred_score, device=self.device), torch.zeros_like(pred_score, device=self.device))
                    # down sample representation for reduced memory impact
                    #repr = nn.AdaptiveAvgPool2d((4,4))(repr)
                    # add ravelled representations to placeholder
                    idx_repr += list(zip(idx.cpu().data.tolist(), repr.view(repr.shape[0], -1).cpu().data.tolist()))
                    idx_label_pred += list(zip(idx.cpu().data.tolist(), label.cpu().data.tolist(), pred.cpu().data.tolist(), pred_score.cpu().data.tolist())) # pred score is softmax activation of class 1
                else:
                    pred_score = self.net(input)
                    pred_score = torch.sigmoid(pred_score) # B x N_class
                    pred = torch.where(pred_score > 0.5, torch.ones_like(pred_score, device=self.device), torch.zeros_like(pred_score, device=self.device))
                    idx_label_pred += list(zip(idx.cpu().data.tolist(), label.cpu().data.tolist(), pred.cpu().data.tolist(), pred_score.cpu().data.tolist()))
                # print_progress
                if self.print_progress:
                    print_progessbar(b, n_batch, Name='\t\tEvaluation Batch', Size=50, erase=True)
            # reset the network attriubtes
            if save_tsne:
                self.net.return_bottleneck = False

        # compute tSNE for representation
        if save_tsne:
            idx, repr = zip(*idx_repr)
            repr = np.array(repr)
            logger.info('Computing the t-SNE representation.')
            repr_2D = TSNE(n_components=2).fit_transform(repr)
            self.outputs['eval']['repr'] = list(zip(idx, repr_2D.tolist()))
            logger.info('Succesfully computed the t-SNE representation.')

        # Compute Accuracy
        _, label, pred, pred_score = zip(*idx_label_pred)
        label, pred, pred_score = np.array(label), np.array(pred), np.array(pred_score)
        auc = roc_auc_score(label, pred_score, average=self.score_average)
        acc = accuracy_score(label.ravel(), pred.ravel())
        sub_acc = accuracy_score(label, pred)
        recall = recall_score(label, pred, average=self.score_average)
        precision = precision_score(label, pred, average=self.score_average)
        f1 = f1_score(label, pred, average=self.score_average)
        self.outputs['eval']['auc'] = auc
        self.outputs['eval']['acc'] = acc
        self.outputs['eval']['subset_acc'] = sub_acc
        self.outputs['eval']['recall'] = recall
        self.outputs['eval']['precision'] = precision
        self.outputs['eval']['f1'] = f1
        self.outputs['eval']['pred'] = idx_label_pred

        # finish evluation
        self.outputs['eval']['time'] = time.time() - start_time

        if return_scores:
            return auc, acc, sub_acc, recall, precision, f1

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
        torch.save(self.net, export_fn)

    def save_model_state_dict(self, export_fn):
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
