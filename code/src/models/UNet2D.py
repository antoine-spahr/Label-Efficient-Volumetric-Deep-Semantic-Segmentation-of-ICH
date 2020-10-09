"""
author: Antoine Spahr

date : 01.10.2020

----------

TO DO :
"""
import torch
import json

from src.models.optim.UNet2D_trainer import UNet2D_trainer
from src.models.optim.LossFunctions import BinaryDiceLoss

class UNet2D:
    """
    Define a 2D Unet model with basic utility functions.
    """
    def __init__(self, unet):
        """
        Build a 2D Unet model for a given architecture.
        ----------
        INPUT
            |---- unet (nn.Module) the network to use in the model. It must take 2D input and genearte a 2D segmentation mask.
        OUTPUT
            |---- a UNet2D model.
        """
        self.unet = unet
        self.trainer = None

        self.train_stat = {
            'train':{
                'time': None,
                'evolution': None
            },
            'eval':{
                'time': None,
                'dice': None,
                'IoU': None
            }
        }

    def train(self, dataset, valid_dataset=None, checkpoint_path=None, n_epoch=150, batch_size=16,
        lr=1e-3, lr_scheduler=torch.optim.lr_scheduler.ExponentialLR, lr_scheduler_kwargs=dict(gamma=0.95),
        loss_fn=BinaryDiceLoss, loss_fn_kwargs=dict(reduction='mean'), weight_decay=1e-6, num_workers=0, device='cuda',
        print_progress=False):
        """
        Train the Unet for the 2D case on the passed settings.
        ----------
        INPUT
            |---- dataset (torch.Dataset) the dataset on which to train. Must return the 2D CT, the segmentation mask,
            |           the patient id and the slice number.
            |---- valid_dataset (torch.Dataset) the optional validation dataset. If provided, the model is
            |           validated at each epoch. It must have the same struture as the train dataset.
            |---- checkpoint_path (str) the filename for a possible checkpoint to start the training from.
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
            |---- None
        """
        self.trainer = UNet2D_trainer(n_epoch=n_epoch, batch_size=batch_size, lr=lr, lr_scheduler=lr_scheduler,
                                      lr_scheduler_kwargs=lr_scheduler_kwargs, loss_fn=loss_fn, loss_fn_kwargs=loss_fn_kwargs,
                                      weight_decay=weight_decay, num_workers=num_workers, device=device,
                                      print_progress=print_progress)
        # Train Unet
        self.unet = self.trainer.train(self.unet, dataset, valid_dataset=valid_dataset, checkpoint_path=checkpoint_path)
        # recover train stat
        self.train_stat['train']['time'] = self.trainer.train_time
        self.train_stat['train']['evolution'] = self.trainer.train_evolution

    def evaluate(self, dataset, save_path=None, batch_size=16, num_workers=0, device='cuda', print_progress=False):
        """
        Evaluate the unet on the passed dataset.
        ----------
        INPUT
            |---- dataset (torch.Dataset) the dataset on which to evaluate. Must return the 2D CT, the segmentation mask,
            |           the patient id and the slice number.
            |---- save_path (str) the folder path where to save segemntation map (as bitmap for each slice) and the
            |           preformance dataframe. If not provided (i.e. None) nothing is saved.
            |---- batch_size (int) the batch size to use for loading the data.
            |---- num_workers (int) the number of workers to use for data loading.
            |---- device (str) the device to use.
            |---- print_progress (bool) whether to print progress bar for batch processing.
        OUTPUT
            |---- None.
        """
        if self.trainer is None:
            self.trainer = UNet2D_trainer(batch_size=batch_size, num_workers=num_workers, device=device, print_progress=print_progress)

        # Evaluate network
        self.trainer.evaluate(self.unet, dataset, return_score=False, print_to_logger=True, save_path=save_path)
        # recover performances
        self.train_stat['eval']['time'] = self.trainer.eval_time
        self.train_stat['eval']['dice'] = self.trainer.eval_dice
        self.train_stat['eval']['IoU'] = self.trainer.eval_IoU

    def save_model(self, export_fn):
        """
        Save the model.
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save({'unet_state_dict': self.unet.state_dict()}, export_fn)

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
        model = torch.load(import_fn, map_location=map_location)
        self.unet.load_state_dict(model['unet_state_dict'])

    def save_train_stat(self, export_fn):
        """
        Save the training stats in JSON.
        ----------
        INPUT
            |---- export_fn (str) path where to get the results.
        OUTPUT
            |---- None
        """
        with open(export_fn, 'w') as fn:
            json.dump(self.train_stat, fn)
