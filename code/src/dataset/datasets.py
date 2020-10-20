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
import pydicom

import src.dataset.transforms as tf
from src.utils.ct_utils import window_ct, resample_ct

class public_SegICH_Dataset2D(data.Dataset):
    """
    Define a torch dataset enabling to load 2D CT and ICH mask.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256):
        """
        Build a dataset for the 2D annotated segmentation of ICH.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a patient number, a slice
            |           number, an image filename and a mask filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
        OUTPUT
            |---- ICH_Dataset2D (torch.Dataset) the 2D dataset.
        """
        super(public_SegICH_Dataset2D, self).__init__()
        self.data_df = data_df
        self.data_path = data_path
        self.window = window

        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size),
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
        slice = io.imread(self.data_path + self.data_df.iloc[idx].CT_fn)
        if self.window:
            slice = window_ct(slice, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # load mask if one, else make a blank array
        if self.data_df.iloc[idx].mask_fn == 'None':
            mask = np.zeros_like(slice)
        else:
            mask = io.imread(self.data_path + self.data_df.iloc[idx].mask_fn)
        # get the patient id
        patient_nbr = torch.tensor(self.data_df.iloc[idx].PatientNumber)
        # get slice number
        slice_nbr = torch.tensor(self.data_df.iloc[idx].SliceNumber)

        # Apply the transform : Data Augmentation + image formating
        slice, mask = self.transform(slice, mask)

        return slice, mask, patient_nbr, slice_nbr

class public_SegICH_Dataset3D(data.Dataset):
    """
    Define a torch dataset enabling to load 3D CT and ICH mask from NIfTI.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.RandomZCrop(Z=64), tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, resampling_dim=(-1, -1, 2.5),
                 resampling_order=1):
        """
        Build a dataset for the 3D annotated segmentation of ICH from NIfTI images.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a patient number, an
            |           image filename and a mask filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- resampling_dim (tuple (x, y, z)) the output pixel dimension for volume reampling. If value is set to
            |           -1, the input pixel dimension is used.
            |---- resampling_order (int) define the interpolation strategy for the resampling. Must be between 0 and 5.
            |           See scipy.ndimage.zoom().
        OUTPUT
            |---- ICH_Dataset3D (torch.Dataset) the 3D dataset.
        """
        super(public_SegICH_Dataset3D, self).__init__()
        self.data_df = data_df
        self.data_path = data_path

        self.window = window
        self.resampling_dim = resampling_dim
        self.resampling_order = resampling_order

        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size),
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
        ct_vol = window_ct(ct_vol, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # resample vol and mask
        ct_vol = resample_ct(ct_vol, pix_dim, out_pixel_dim=self.resampling_dim, preserve_range=True,
                             order=self.resampling_order)
        mask = resample_ct(mask, pix_dim, out_pixel_dim=self.resampling_dim, preserve_range=True,
                           order=0)#self.resampling_order)

        ct_vol, mask = self.transform(ct_vol, mask)

        return ct_vol, mask.bool(), pID


class RSNA_dataset(data.Dataset):
    """
    Dataset object to load the RSNA data.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256,
                 mode='standard', n_swap=10, swap_w=15, swap_h=15, contrastive_augmentation=None):
        """
        Build a dataset for the RSNA dataset of ICH CT slice.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a filename and a columns
            |           Hemorrhage specifying if slice has or not an hemorrhage.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
            |---- mode (str) define how to load the RSNA dataset. 'standard': return an image with its label.
            |           'context_restoration': return the image and the corruped image. 'contrastive': return two heavilly
            |           augmented version of the input image.
            |---- n_swap (int) the number of swap to use in the context_restoration mode.
            |---- swap_h (int) the height of the swapped patch in the context_restoration mode.
            |---- swap_w (int) the width of the swapped patch in the context_restoration mode.
            |---- contrastive_augmentation (list of transformation) the list of augmentation to apply in the contrastive
            |           mode. They must be composable by tf.Compose.
        OUTPUT
            |---- RSNA_dataset (torch.Dataset) the RSNA dataset.
        """
        super(RSNA_dataset, self).__init__()
        self.data_df = data_df
        self.data_path = data_path
        self.window = window
        assert mode in ['standard', 'context_restoration', 'contrastive'], f"Invalid mode. Must be one of 'standard', 'context_restoration', 'contrastive'. Given : {mode}"
        self.mode = mode

        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size))#,
                                    #tf.ToTorchTensor())
        self.toTensor = tf.ToTorchTensor()
        if mode == 'context_restoration':
            self.swap_tranform = tf.RandomPatchSwap(n=n_swap, w=swap_w, h=swap_h)
        elif mode == 'contrastive':
            self.contrastive_transform = tf.Compose()
            raise NotImplementedError

    def __len__(self):
        """
        eturn the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Extract the CT sepcified by idx.
        ----------
        INPUT
            |---- idx (int) the sample index in self.data_df.
        OUTPUT
            |---- im (torch.tensor) the CT image with dimension (1 x H x W).
            |---- lab (torch.tensor) the label for hemorrhage presence (0 or 1).
            |---- idx (torch.tensor) the sample idx.
        """
        # load dicom and recover the CT pixel values
        dcm_im = pydicom.dcmread(self.data_path + self.data_df.iloc[idx].filename)
        im = (dcm_im.pixel_array * float(dcm_im.RescaleSlope) + float(dcm_im.RescaleIntercept))
        # Window the CT-scan
        if self.window:
            im = window_ct(im, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # transform image
        im = self.transform(im)
        if self.mode == 'standard':
            # load label
            #lab = self.data_df.iloc[idx].Hemorrhage
            return self.toTensor(im), torch.tensor(idx) #torch.tensor(lab), torch.tensor(idx)
        elif self.mode == 'context_restoration':
            # generate corrupeted version
            swapped_im = self.swap_tranform(im)
            return self.toTensor(im), self.toTensor(swapped_im), torch.tensor(idx)
        elif self.mode == 'contrastive':
            # augmente image twice
            im1 = self.contrastive_transform(im)
            im2 = self.contrastive_transform(im)
            return self.toTensor(im1), self.toTensor(im2), torch.tensor(idx)
