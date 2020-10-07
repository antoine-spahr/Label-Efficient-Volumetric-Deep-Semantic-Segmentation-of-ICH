"""
author: Antoine Spahr

date : 29.09.2020

----------

TO DO :
"""
import pandas as pd
import numpy as np
import skimage.io as io
import skimage
import torch
from torch.utils import data
import nibabel as nib

import src.dataset.transforms as tf
from src.utils.ct_utils import window_ct, resample_ct

class public_SegICH_Dataset2D(data.Dataset):
    """
    Define a torch dataset enabling to load 2D CT and ICH mask.
    """
    def __init__(self, data_df, data_path, data_augmentation=True, window=None, output_size=256):
        """
        Build a dataset for the 2D annotated segmentation of ICH.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a patient number, a slice
            |           number, an image filename and a mask filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- data_augmentation (bool) whether to apply data augmentation.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
        OUTPUT
            |---- ICH_Dataset2D (torch.Dataset) the 2D dataset.
        """
        super(public_SegICH_Dataset2D, self).__init__()
        self.data_df = data_df
        self.data_path = data_path
        self.window = window

        if data_augmentation:
            self.transform = tf.Compose(tf.Translate(low=-0.1, high=0.1),
                                        tf.Rotate(low=-10, high=10),
                                        tf.Scale(low=0.9, high=1.1),
                                        tf.Resize(H=output_size, W=output_size),
                                        tf.HFlip(p=0.5),
                                        tf.ToTorchTensor())
        else:
            self.transform = tf.Compose(tf.Resize(H=output_size, W=output_size),
                                        tf.ToTorchTensor())

    def __len__(self):
        """
        Return the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Extract the CT and corresponding mask sepcified by idx.
        ----------
        INPUT
            |---- idx (int) the sample index in self.data_df.
        OUTPUT
            |---- slice (torch.tensor) the CT image with dimension (1 x H x W).
            |---- mask (torch.tensor) the segmentation mask with dimension (1 x H x W).
            |---- patient_nbr (torch.tensor) the patient id as a single value.
            |---- slice_nbr (torch.tensor) the slice number as a single value.
        """
        # load image
        slice = io.imread(self.data_path + self.data_df.loc[idx, 'CT_fn'])
        if self.window:
            slice = window_ct(slice, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # load mask if one, else make a blank array
        if self.data_df.loc[idx, 'mask_fn'] == 'None':
            mask = np.zeros_like(slice)
        else:
            mask = io.imread(self.data_path + self.data_df.loc[idx, 'mask_fn'])
        # get the patient id
        patient_nbr = torch.tensor(self.data_df.loc[idx, 'PatientNumber'])
        # get slice number
        slice_nbr = torch.tensor(self.data_df.loc[idx, 'SliceNumber'])

        # Apply the transform : Data Augmentation + image formating
        slice, mask = self.transform(slice, mask)

        return slice, mask, patient_nbr, slice_nbr

class public_SegICH_Dataset3D(data.Dataset):
    """
    Define a torch dataset enabling to load 3D CT and ICH mask from NIfTI.
    """
    def __init__(self, data_df, data_path, data_augmentation=True, win_center=40, win_width=120, out_range=(0,1),
                 resampling_dim=(-1, -1, 2.5), resampling_order=1, vol_thickness=64):
        """
        Build a dataset for the 3D annotated segmentation of ICH from NIfTI images.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a patient number, an
            |           image filename and a mask filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- data_augmentation (bool) whether to apply data augmentation.
            |---- win_center (int) the window center for the CT-scan windowing.
            |---- win_width (int) the window width for the CT-scan windowing.
            |---- out_range (tuple (low, high)) the output range for the windowing.
            |---- resampling_dim (tuple (x, y, z)) the output pixel dimension for volume reampling. If value is set to
            |           -1, the input pixel dimension is used.
            |---- resampling_order (int) define the interpolation strategy for the resampling. Must be between 0 and 5.
            |           See scipy.ndimage.zoom().
            |---- vol_thickness (int) the number of slice to take from the resampled volume.
        OUTPUT
            |---- ICH_Dataset3D (torch.Dataset) the 3D dataset.
        """
        super(public_SegICH_Dataset3D, self).__init__()
        self.data_df = data_df
        self.data_path = data_path

        self.win_center = win_center
        self.win_width = win_width
        self.out_range = out_range
        self.resampling_dim = resampling_dim
        self.resampling_order = resampling_order

        if data_augmentation:
            self.transform = tf.Compose(tf.RandomZCrop(Z=vol_thickness),
                                        tf.Translate(low=-0.1, high=0.1),
                                        tf.Rotate(low=-10, high=10),
                                        tf.Scale(low=0.9, high=1.1),
                                        tf.Resize(H=256, W=256),
                                        tf.HFlip(p=0.5),
                                        tf.ToTorchTensor())
        else:
            self.transform = tf.Compose(tf.RandomZCrop(Z=vol_thickness),
                                        tf.Resize(H=256, W=256),
                                        tf.ToTorchTensor())

    def __len__(self):
        """
        Return the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Get the CT-volumes of the given patient idx.
        ----------
        INPUT
            |---- idx (int) the patient index in self.PatientID_list to extract.
        OUTPUT
            |---- volume (torch.Tensor) the CT-volume in a tensor (H x W x Slice)
        """
        # load data
        ct_nii = nib.load(self.data_path + self.data_df.loc[idx, 'CT_fn'])
        mask_nii = nib.load(self.data_path + self.data_df.loc[idx, 'mask_fn'])
        pID = torch.tensor(self.data_df.loc[idx, 'PatientNumber'])
        # get volumes and pixel dimension
        ct_vol = np.rot90(ct_nii.get_fdata(), axes=(0,1))
        mask = np.rot90(mask_nii.get_fdata(), axes=(0,1))
        pix_dim = ct_nii.header['pixdim'][1:4] # recover pixel physical dimension

        # window CT-scan for soft tissus
        ct_vol = window_ct(ct_vol, win_center=self.win_center, win_width=self.win_width, out_range=self.out_range)
        # resample vol and mask
        ct_vol = resample_ct(ct_vol, pix_dim, out_pixel_dim=self.resampling_dim, preserve_range=True,
                             order=self.resampling_order)
        mask = resample_ct(mask, pix_dim, out_pixel_dim=self.resampling_dim, preserve_range=True,
                           order=0)#self.resampling_order)

        ct_vol, mask = self.transform(ct_vol, mask)

        return ct_vol, mask.bool(), pID