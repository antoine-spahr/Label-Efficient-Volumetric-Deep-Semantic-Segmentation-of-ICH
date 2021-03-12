"""
author: Antoine Spahr

date : 01.03.2021

----------
To Do:
    -
"""
import os
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import skimage.io as io
from skimage import img_as_ubyte
from datetime import timedelta

from src.models.optim.LossFunctions import GDL
from src.utils.print_utils import print_progessbar

class AE:
    """ """
    def __init__(self, net, n_epoch=100, batch_size=16, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
                 lr_scheduler_kwargs=dict(gamma=0.95), lambda_GDL={"0": 0.0, "25": 1.0}, weight_decay=1e-6, num_workers=0,
                 device='cuda', print_progress=False, checkpoint_freq=3):
        """

        """
        self.ae = net.to(device)
        # data param
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        # optimization  param
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        # loss param
        self.ep_GDL = lambda_GDL
        self.lambda_GDL = 0.0#self.ep_GDL["1"]
        # other
        self.device = device
        self.print_progress = print_progress
        self.checkpoint_freq = checkpoint_freq
        # outputs
        self.outputs = {
            "train" : {
                "time": None,
                "evolution": None
            },
            "eval": None
        }

    def train(self, dataset, checkpoint_path=None, valid_dataset=None, valid_path=None, valid_freq=5):
        """

        """
        logger = logging.getLogger()
        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=True, worker_init_fn=lambda _: np.random.seed())
        # make optimizers
        optimizer = optim.Adam(self.ae.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # make scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        # make loss functions
        gdl_fn = GDL(reduction='mean', device=self.device)
        mae_fn = nn.L1Loss(reduction='mean')
        mse_fn = nn.MSELoss(reduction='mean')
        # Load checkpoint if present
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            n_epoch_finished = checkpoint['n_epoch_finished']
            self.ae.load_state_dict(checkpoint['net_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['lr_state'])
            epoch_loss_list = checkpoint['loss_evolution']
            logger.info(f'Checkpoint loaded with {n_epoch_finished} epoch finished.')
        except FileNotFoundError:
            logger.info('No Checkpoint found. Training from beginning.')
            n_epoch_finished = 0
            epoch_loss_list = [] # Placeholder for epoch evolution

        logger.info('Start training the inpainting AE.')
        start_time = time.time()
        n_batch = len(loader)
        # train loop
        for epoch in range(n_epoch_finished, self.n_epoch):
            self.ae.train()
            epoch_loss, epoch_loss_l1, epoch_loss_l2, epoch_loss_gdl = 0.0, 0.0, 0.0, 0.0
            epoch_start_time = time.time()

            # update Lambda GDL
            if str(epoch) in self.ep_GDL.keys():
                self.lambda_GDL = self.ep_GDL[str(epoch)]
                logger.info(f"Lambda GLD set to {self.lambda_GDL}.")

            for b, data in enumerate(loader):
                im, _ = data
                im = im.to(self.device).float().requires_grad_(True)

                optimizer.zero_grad()
                rec = self.ae(im)

                loss_l1 = mae_fn(rec, im)
                loss_l2 = mse_fn(rec, im)
                #loss_gdl = self.lambda_GDL*gdl_fn(im, rec) if epoch+1 >= self.ep_GDL else 0.0*gdl_fn(im, rec) # consider gdl loss only after some epoch
                loss_gdl = self.lambda_GDL*gdl_fn(im, rec)
                loss = loss_l1 + loss_l2 + loss_gdl

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_l1 += loss_l1.item()
                epoch_loss_l2 += loss_l2.item()
                epoch_loss_gdl += loss_gdl.item()

                if self.print_progress:
                    print_progessbar(b, n_batch, Name='Train Batch', Size=100, erase=True)

            # validate
            valid_loss = 0.0
            if valid_dataset:
                save_path = valid_path if (epoch+1)%valid_freq == 0 else None
                valid_loss = self.validate(valid_dataset, save_path=save_path, prefix=f'_ep{epoch+1}')

            # print epoch summary
            logger.info(f"| Epoch {epoch+1:03}/{self.n_epoch:03} "
                        f"| Time {timedelta(seconds=int(time.time() - epoch_start_time))} "
                        f"| Loss (L1 + L2 + GDL) {epoch_loss/n_batch:.5f} = {epoch_loss_l1/n_batch:.5f} + {epoch_loss_l2/n_batch:.5f} + {epoch_loss_gdl/n_batch:.5f} "
                        f"| Valid Loss {valid_loss:.5f} | lr {scheduler.get_last_lr()[0]:.6f} |")

            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, epoch_loss_l1/n_batch, epoch_loss_l2/n_batch, epoch_loss_gdl/n_batch])

            # update lr
            scheduler.step()

            # save checkpoint
            if (epoch+1)%self.checkpoint_freq == 0 and checkpoint_path is not None:
                checkpoint ={
                    'n_epoch_finished': epoch+1,
                    'net_state': self.ae.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'lr_state': scheduler.state_dict(),
                    'loss_evolution': epoch_loss_list
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

        self.outputs['train']['time'] = time.time() - start_time
        self.outputs['train']['evolution'] = {'col_name': ['Epoch', 'Loss_total', 'L1_loss', 'L2_loss', 'GDL_loss'],
                                              'data': epoch_loss_list}
        logger.info(f"Finished training inpainter AE in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def validate(self, dataset, save_path=None, prefix=''):
        """

        """
        with torch.no_grad():
            # make loader
            valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                       shuffle=False, worker_init_fn=lambda _: np.random.seed())
            n_batch = len(valid_loader)
            l1_fn, l2_fn, gdl_fn = nn.L1Loss(reduction='mean'), nn.MSELoss(reduction='mean'), GDL(reduction='mean', device=self.device)

            self.ae.eval()
            # validate data by batch
            valid_loss = 0.0
            for b, data in enumerate(valid_loader):
                im, idx = data
                im = im.to(self.device).float()
                idx = idx.cpu().numpy()
                # reconstruct
                im_rec = self.ae(im)
                # compute L1 loss
                valid_loss += l1_fn(im_rec, im).item() + l2_fn(im_rec, im).item() + self.lambda_GDL*gdl_fn(im, im_rec).item()
                # save results
                if save_path:
                    for i in range(im.shape[0]):
                        arr = np.concatenate([im[i].permute(1,2,0).squeeze().cpu().numpy(), im_rec[i].permute(1,2,0).squeeze().cpu().numpy()], axis=1)
                        io.imsave(os.path.join(save_path, f'valid_im{idx[i]}{prefix}.png'), img_as_ubyte(arr), check_contrast=False)

                print_progessbar(b, n_batch, Name='Valid Batch', Size=100, erase=True)

        return valid_loss / n_batch

    def save_model(self, export_fn):
        """
        Save the model.
        ----------
        INPUT
            |---- export_fn (str) the export path.
        OUTPUT
            |---- None
        """
        torch.save(self.ae.state_dict(), export_fn)

    def load_model(self, import_fn, map_location='cuda'):
        """
        Load an AE model from the given path.
        ----------
        INPUT
            |---- import_fn (str) path where to get the model.
            |---- map_location (str) device on which to load the model.
        OUTPUT
            |---- None
        """
        loaded_state_dict = torch.load(import_fn, map_location=map_location)
        self.ae.load_state_dict(loaded_state_dict)

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
