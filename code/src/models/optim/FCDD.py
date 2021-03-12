"""
author: Antoine Spahr

date : 04.03.2021

----------

TO DO :
    - check best way to bring heat map into [0,1]
"""
import os
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
import skimage.io as io
from skimage import img_as_ubyte
from sklearn.metrics import roc_auc_score

from src.models.optim.LossFunctions import HSCLoss
from src.utils.print_utils import print_progessbar

class FCDD:
    """ """
    def __init__(self, net, n_epoch=100, batch_size=64, lr=1e-3, lr_scheduler=optim.lr_scheduler.ExponentialLR,
                 lr_scheduler_kwargs=dict(gamma=0.95), weight_decay=1e-6, num_workers=0, device='cuda', print_progress=False,
                 checkpoint_freq=10):
        """

        """
        self.net = net.to(device)
        # data param
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        # optimization  param
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
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


    def train(self, dataset, checkpoint_path=None, valid_dataset=None):
        """

        """
        logger = logging.getLogger()
        # make dataloader
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=True, worker_init_fn=lambda _: np.random.seed())
        # make optimizers
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # make scheduler
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        # make loss functions
        loss_fn = HSCLoss(reduction='mean')
        # Load checkpoint if present
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
            epoch_loss_list = [] # Placeholder for epoch evolution

        logger.info('Start training FCDD.')
        start_time = time.time()
        n_batch = len(loader)
        # train loop
        for epoch in range(n_epoch_finished, self.n_epoch):
            self.net.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for b, data in enumerate(loader):
                im, label, _ = data
                im = im.to(self.device).float().requires_grad_(True)
                label = label.to(self.device)

                optimizer.zero_grad()
                feat_map = self.net(im)

                loss = loss_fn(feat_map, label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if self.print_progress:
                    print_progessbar(b, n_batch, Name='Train Batch', Size=100, erase=True)

            # validate
            valid_loss, valid_auc = 0.0, 0.0
            if valid_dataset:
                valid_loss, valid_auc = self.validate(valid_dataset)

            # print epoch summary
            logger.info(f"| Epoch {epoch+1:03}/{self.n_epoch:03} "
                        f"| Time {timedelta(seconds=int(time.time() - epoch_start_time))} "
                        f"| Loss {epoch_loss/n_batch:.5f} "
                        f"| Valid Loss {valid_loss:.5f} | Valid AUC {valid_auc:.2%} "
                        f"| lr {scheduler.get_last_lr()[0]:.6f} |")

            epoch_loss_list.append([epoch+1, epoch_loss/n_batch, valid_loss, valid_auc])

            # update lr
            scheduler.step()

            # save checkpoint
            if (epoch+1)%self.checkpoint_freq == 0 and checkpoint_path is not None:
                checkpoint ={
                    'n_epoch_finished': epoch+1,
                    'net_state': self.net.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'lr_state': scheduler.state_dict(),
                    'loss_evolution': epoch_loss_list
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info('\tCheckpoint saved.')

        self.outputs['train']['time'] = time.time() - start_time
        self.outputs['train']['evolution'] = {'col_name': ['Epoch', 'Train_Loss', 'Valid_Loss', 'Valid_AUC'],
                                              'data': epoch_loss_list}
        logger.info(f"Finished training FCDD in {timedelta(seconds=int(self.outputs['train']['time']))}")

    def validate(self, dataset):
        """
        validate with dataset and retrun loss and AUC computed on anomaly score (i.e. sum of output feature map).
        """
        with torch.no_grad():
            # make loader
            valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                       shuffle=False, worker_init_fn=lambda _: np.random.seed())
            n_batch = len(valid_loader)
            loss_fn = HSCLoss(reduction='mean')

            self.net.eval()
            # validate data by batch
            valid_loss = 0.0
            label_score = []
            for b, data in enumerate(valid_loader):
                im, label, _ = data
                im = im.to(self.device).float()
                label = label.to(self.device)
                # reconstruct
                feat_map = self.net(im)
                # compute loss
                valid_loss += loss_fn(feat_map, label).item()
                # compute score
                ad_score = (torch.sqrt(feat_map**2 + 1) - 1).reshape(feat_map.shape[0], -1).sum(-1)
                # save label and scores
                label_score += list(zip(label.cpu().data.tolist(), ad_score.cpu().data.tolist()))

                if self.print_progress:
                    print_progessbar(b, n_batch, Name='Valid Batch', Size=100, erase=True)

            # compute AUC
            label, score = zip(*label_score)
            auc = roc_auc_score(np.array(label), np.array(score))

        return valid_loss / n_batch, auc

    def get_min_max(self, loader, reception=True, std=None, cpu=True, q_min=0.025, q_max=0.975):
        """
        Compute the Min and Max values of the heat maps on the dataset. For each batch a possible new min or max values
        is selected as the q_min or q_max quantile of the heatmaps entries.
        """
        self.net.eval()
        # get scaling parameters with one forward pass
        min_val, max_val = np.inf, -np.inf
        for b, data in enumerate(loader):
            #im, _, _ = data
            im = data[0]
            im = im.to(self.device).float()
            #label = label.to(self.device)
            heatmap = self.generate_heatmap(im, reception=reception, std=std, cpu=cpu)
            qmax = torch.kthvalue(heatmap.reshape(-1), int(q_max * heatmap.reshape(-1).size(0)))[0] if q_max < 1.0 else heatmap.max()
            if qmax > max_val:
                max_val = qmax
            qmin = torch.kthvalue(heatmap.reshape(-1), int(q_min * heatmap.reshape(-1).size(0)))[0] if q_min > 0 else heatmap.min()
            if qmin < min_val:
                min_val = qmin

            if self.print_progress:
                print_progessbar(b, len(loader), Name='Getting Scaling Factor', Size=100, erase=True)

        return min_val, max_val

    def localize_anomalies(self, dataset, save_path=None, reception=True, std=None, cpu=True, q_min=0.025, q_max=0.975):
        """
        Generate heat map for image in dataset and save them.
        """
        with torch.no_grad():
            # make loader
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                 shuffle=False, worker_init_fn=lambda _: np.random.seed())
            self.net.eval()
            
            min_val, max_val = self.get_min_max(loader, reception=reception, std=std, cpu=cpu, q_min=q_min, q_max=q_max)

            # computing and saving heatmaps
            for b, data in enumerate(loader):
                im, label, idx = data
                im = im.to(self.device).float()
                label = label.to(self.device)

                heatmap = self.generate_heatmap(im, reception=reception, std=std, cpu=cpu)#, qu=qu)
                # scaling
                heatmap = ((heatmap - min_val) / (max_val - min_val)).clamp(0,1)

                if save_path:
                    for i in range(im.shape[0]):
                        arr = np.concatenate([im[i].squeeze().cpu().numpy(),
                                              heatmap.repeat(1,im.shape[1],1,1)[i].squeeze().cpu().numpy()], axis=1)
                        io.imsave(os.path.join(save_path, f'heatmap_{idx[i]}_{label[i].item()}.png'), img_as_ubyte(arr), check_contrast=False)

                if self.print_progress:
                    print_progessbar(b, len(loader), Name='Heatmap Generation Batch', Size=100, erase=True)

    def generate_heatmap(self, input, reception=True, std=None, cpu=True):
        """
        Generate heat map for image in dataset and save them.
        """
        assert input.ndim == 4, f"Input must be 4-dimensional [B x C x H x W]. Given {input.shape}"
        with torch.no_grad():
            self.net.eval()
            input = input.to(self.device).float()
            feat_map = torch.sqrt(self.net(input) ** 2 + 1) - 1
            heatmap = self.net.receptive_upsample(feat_map, reception=reception, std=std, cpu=cpu)

        return heatmap

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

    def load_model(self, import_fn, map_location='cuda'):
        """
        Load a networkmodel from the given path.
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
        Save the outputs in JSON.
        ----------
        INPUT
            |---- export_fn (str) path where to get the results.
        OUTPUT
            |---- None
        """
        with open(export_fn, 'w') as fn:
            json.dump(self.outputs, fn)
