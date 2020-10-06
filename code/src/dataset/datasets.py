"""
author: Antoine Spahr

date : 29.09.2020

----------

TO DO :
- check if mask should be loaded as ubyte (boolean)
- adapte to load NIfTI images in 2D and 3D --> in 3D need to expand Z-direction to have the same physical spacing (say 5mm).
"""
import pandas as pd
import numpy as np
import skimage.io as io
import skimage
import torch
from torch.utils import data

import src.dataset.transforms as tf

class public_SegICH_Dataset2D(data.Dataset):
    """
    Define a torch dataset enabling to load 2D CT and ICH mask.
    """
    def __init__(self, data_df, data_path, data_augmentation=True):
        """
        Build a dataset for the 2D annotated segmentation of ICH.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a patient number, a slice
            |           number, an image filename and a mask filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- data_augmentation (bool) whether to apply data augmentation.
        OUTPUT
            |---- ICH_Dataset2D (torch.Dataset) the 2D dataset.
        """
        super(public_SegICH_Dataset2D, self).__init__()
        self.data_df = data_df
        self.data_path = data_path

        if data_augmentation:
            self.transform = tf.Compose(tf.Translate(low=-0.1, high=0.1),
                                        tf.Rotate(low=-10, high=10),
                                        tf.Scale(low=0.9, high=1.1),
                                        tf.Resize(H=256, W=256),
                                        tf.HFlip(p=0.5),
                                        tf.ToTorchTensor())
        else:
            self.transform = tf.Compose(tf.Resize(H=256, W=256),
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
        # load mask if one, else make a blank array
        if self.data_df.loc[idx, 'mask_fn'] == 'None':
            mask = skimage.img_as_bool(np.zeros_like(slice))
        else:
            mask = skimage.img_as_bool(io.imread(self.data_path + self.data_df.loc[idx, 'mask_fn']))
        # get the patient id
        patient_nbr = torch.tensor(self.data_df.loc[idx, 'PatientNumber'])
        # get slice number
        slice_nbr = torch.tensor(self.data_df.loc[idx, 'SliceNumber'])

        # Apply the transform : Data Augmentation + image formating
        slice, mask = self.transform(slice, mask)

        return slice, mask, patient_nbr, slice_nbr

class public_SegICH_Dataset3D(data.Dataset):
    """

    """
    def __init__(self, data_df, data_path, data_augmentation=True):
        """

        """
        super(public_SegICH_Dataset3D, self).__init__()
        self.data_df = data_df
        self.data_path = data_path

        # get the list of Patient --> list of volumes (i.e. the sample list)
        self.PatientID_list = self.data_df.PatientNumber.unique()

        if data_augmentation:
            # self.transform = ...
            pass
        else:
            # self.transform = ...
            pass

    def __len__(self):
        """
        Return the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return len(self.PatientID_list)

    def __getitem__(self, idx):
        """
        Get the CT-volumes of the given patient idx.
        ----------
        INPUT
            |---- idx (int) the patient index in self.PatientID_list to extract.
        OUTPUT
            |---- volume (torch.Tensor) the CT-volume in a tensor (H x W x Slice)
        """
        # get the dataframe of the patient at idx
        df_tmp = self.data_df[self.data_df.PatientNumber == self.PatientID_list[idx]]

        # load all the slices & mask into a 3D array
        slice_list, mask_list = [], []
        for slice_path, mask_path in zip(df_tmp.img_fn, df_tmp.mask_fn):
            slice_list.append(io.imread(self.data_path + slice_path))
            if mask_path == 'None':
                mask = np.zeros_like(slice_list[-1])#skimage.img_as_bool(np.zeros_like(slice_list[-1]))
            else:
                mask = io.imread(self.data_path + mask_path)#skimage.img_as_bool(io.imread(self.data_path + mask_path))
            mask_list.append(mask)

        volume = np.stack(slice_list, axis=2)
        volume_mask = np.stack(mask_list, axis=2)


        # get the patient id
        pID = torch.tensor(self.PatientID_list[idx])

        # Apply the transform : Data Augmentation + image formating
        #volume, volume_mask = self.transform(volume, volume_mask)

        return volume, volume_mask, pID
