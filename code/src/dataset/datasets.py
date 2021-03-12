"""
author: Antoine Spahr

date : 29.09.2020

----------

TO DO :
"""
import os
import pandas as pd
import numpy as np
import skimage.io as io
import skimage.transform
import skimage.draw
import skimage
import cv2
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

class public_SegICH_AttentionDataset2D(data.Dataset):
    """

    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256):
        """
        Build a dataset for the 2D annotated segmentation of ICH with attention map.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a patient number, a slice
            |           number, an image filename, a mask filename, and an attention map filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
        OUTPUT
            |---- ICH_Dataset2D (torch.Dataset) the 2D dataset.
        """
        super(public_SegICH_AttentionDataset2D, self).__init__()
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
        Extract the stacked CT and attention map, the corresponding ground truth mask, volume id, and slice number sepcified by idx.
        ----------
        INPUT
            |---- idx (int) the sample index in self.data_df.
        OUTPUT
            |---- input (torch.tensor) the CT image stacked with the attention map with dimension (2 x H x W).
            |---- mask (torch.tensor) the segmentation mask with dimension (1 x H x W).
            |---- patient_nbr (torch.tensor) the patient id as a single value.
            |---- slice_nbr (torch.tensor) the slice number as a single value.
        """
        # load image
        slice = io.imread(self.data_path + self.data_df.iloc[idx].ct_fn)
        if self.window:
            slice = window_ct(slice, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # load attention map and stack it with the slice
        if self.data_df.iloc[idx].attention_fn == 'None':
            attention_map = np.zeros_like(slice)
        else:
            attention_map = skimage.img_as_float(io.imread(self.data_path + self.data_df.iloc[idx].attention_fn))
            attention_map = skimage.transform.resize(attention_map, slice.shape[:2], order=1, preserve_range=True)
        input = np.stack([slice, attention_map], axis=2)

        # load mask if one, else make a blank array
        if self.data_df.iloc[idx].mask_fn == 'None':
            mask = np.zeros_like(slice)
        else:
            mask = io.imread(self.data_path + self.data_df.iloc[idx].mask_fn)
        # get the patient id
        patient_nbr = torch.tensor(self.data_df.iloc[idx].id)
        # get slice number
        slice_nbr = torch.tensor(self.data_df.iloc[idx].slice)

        # Apply the transform : Data Augmentation + image formating
        input, mask = self.transform(input, mask)

        return input, mask, patient_nbr, slice_nbr

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

class brain_extract_Dataset2D(data.Dataset):
    """
    Define a torch dataset enabling to load 2D CT and brain mask.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256):
        """
        Build a dataset for the 2D annotated segmentation of brain.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a volume number, a slice
            |           number, an image filename and a mask filename.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
        OUTPUT
            |---- brain_Dataset2D (torch.Dataset) the 2D dataset.
        """
        super(brain_extract_Dataset2D, self).__init__()
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
        slice = io.imread(os.path.join(self.data_path, self.data_df.iloc[idx].ct_fn))
        if self.window:
            slice = window_ct(slice, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # load mask if one, else make a blank array
        if self.data_df.iloc[idx].mask_fn == 'None':
            mask = np.zeros_like(slice)
        else:
            mask = io.imread(os.path.join(self.data_path, self.data_df.iloc[idx].mask_fn))
        # get the patient id
        vol_id = torch.tensor(self.data_df.iloc[idx].volume)
        # get slice number
        slice_nbr = torch.tensor(self.data_df.iloc[idx].slice)

        # Apply the transform : Data Augmentation + image formating
        slice, mask = self.transform(slice, mask)

        return slice, mask, vol_id, slice_nbr

class RSNA_dataset(data.Dataset):
    """
    Dataset object to load the RSNA data.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256,
                 mode='standard', n_swap=10, swap_w=15, swap_h=15, swap_rot=False, contrastive_augmentation=None):
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
            |           augmented version of the input image. 'binary_classification': return image and binary label (ICH vs No-ICH).
            |---- n_swap (int) the number of swap to use in the context_restoration mode.
            |---- swap_h (int) the height of the swapped patch in the context_restoration mode.
            |---- swap_w (int) the width of the swapped patch in the context_restoration mode.
            |---- swap_rot (bool) whether to rotate patches. If true, swap_h must be None.
            |---- contrastive_augmentation (list of transformation) the list of augmentation to apply in the contrastive
            |           mode. They must be composable by tf.Compose.
        OUTPUT
            |---- RSNA_dataset (torch.Dataset) the RSNA dataset.
        """
        super(RSNA_dataset, self).__init__()
        self.data_df = data_df.copy()
        self.n_sample = len(data_df)
        self.data_path = data_path
        self.window = window
        assert mode in ['standard', 'context_restoration', 'contrastive', 'binary_classification', 'multi_classification'], f"Invalid mode. Must be one of 'standard', 'context_restoration', 'contrastive', 'binary_classification'. Given : {mode}"
        self.mode = mode

        self.transform = tf.Compose(*augmentation_transform, tf.Resize(H=output_size, W=output_size))#,
                                    #tf.ToTorchTensor())
        self.toTensor = tf.ToTorchTensor()
        if mode == 'context_restoration':
            self.swap_tranform = tf.RandomPatchSwap(n=n_swap, w=swap_w, h=swap_h, rotate=swap_rot)
        elif mode == 'contrastive':
            self.contrastive_transform = tf.Compose(*contrastive_augmentation)
        elif mode == 'multi_classification':
            # add a columns 'no_Hemorrage'
            self.data_df['no_Hemorrhage'] = 1 - self.data_df.Hemorrhage
            # name of the classes
            self.class_name = ['no_Hemorrhage', 'Hemorrhage', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

    def __len__(self):
        """
        eturn the number of samples in the dataset.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- N (int) the number of samples in the dataset.
        """
        return self.n_sample

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

        if self.mode == 'standard':
            # transform image
            im = self.transform(im)
            return self.toTensor(im), torch.tensor(idx) #torch.tensor(lab), torch.tensor(idx)
        elif self.mode == 'context_restoration':
            # generate corrupeted version
            # transform image
            im = self.transform(im)
            swapped_im = self.swap_tranform(im)
            return self.toTensor(im), self.toTensor(swapped_im), torch.tensor(idx)
        elif self.mode == 'contrastive':
            # augmente image twice
            im1 = self.contrastive_transform(self.transform(im))
            im2 = self.contrastive_transform(self.transform(im))
            return self.toTensor(im1), self.toTensor(im2), torch.tensor(idx)
        elif self.mode == 'binary_classification':
            im = self.transform(im)
            label = self.data_df.iloc[idx].Hemorrhage
            return self.toTensor(im), torch.tensor(label), torch.tensor(idx)
        elif self.mode == 'multi_classification':
            im = self.transform(im)
            samp = self.data_df.iloc[idx]
            label = [samp[name] for name in self.class_name]
            return self.toTensor(im), torch.tensor(label), torch.tensor(idx)

class RSNA_Inpaint_dataset(data.Dataset):
    """
    Dataset object to load the RSNA data for the inpainting task.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256, n_draw=(1,4), vertex=(5,15),
                 brush_width=(10,30), angle=(0.0,6.28), length=(10,30), n_salt_pepper=(0,10), salt_peper_radius=(1,3)):
        """
        Build a dataset for the RSNA dataset CT slice.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a filename and a columns
            |           Hemorrhage specifying if slice has or not an hemorrhage.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
            |---- n_draw (tuple (low, high)) range of number of inpaint element to draw.
            |---- vertex (tuple (low, high)) range of number of vertex for each inpaint element.
            |---- brush_width (tuple (low, high)) range of brush size to draw each inpaint element.
            |---- angle (tuple (low, high)) the range of angle between each vertex of an inpaint element. Note that every
            |               two segment, Pi is added to the angle to keep the drawing in the vicinity. Angle in radian.
            |---- length (tuple (low, high)) range of length for each segment.
            |---- n_salt_pepper (tuple (low, high)) range of number of salt and pepper disk element to draw. Set to (0,1)
            |               for no salt and pepper elements.
            |---- salt_peper_radius (tuple (low, high)) range of radius for the salt and pepper disk element.
        OUTPUT
            |---- RSNA_Inpaint_dataset (torch.Dataset) the RSNA dataset for inpainting.
        """
        super(RSNA_Inpaint_dataset, self).__init__()
        self.data_df = data_df
        self.data_path = data_path
        self.window = window

        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size),
                                    tf.ToTorchTensor())
        self.n_draw = n_draw
        self.vertex = vertex
        self.brush_width = brush_width
        self.angle = angle
        self.length = length
        self.n_salt_pepper = n_salt_pepper
        self.salt_peper_radius = salt_peper_radius

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
            |---- mask (torch.tensor) the inpaining mask with dimension (1 x H x W).
        """
        # load dicom and recover the CT pixel values
        dcm_im = pydicom.dcmread(self.data_path + self.data_df.iloc[idx].filename)
        im = (dcm_im.pixel_array * float(dcm_im.RescaleSlope) + float(dcm_im.RescaleIntercept))
        # Window the CT-scan
        if self.window:
            im = window_ct(im, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # transform image
        im = self.transform(im)
        # get a mask
        mask = self.random_ff_mask((im.shape[1], im.shape[2]))

        return im, tf.ToTorchTensor()(mask)

    def random_ff_mask(self, shape):
        """
        Generate a random inpainting mask with given shape.
        ----------
        INPUT
            |---- shape (tuple (h,w)) the size of the inpainting mask.
        OUTPUT
            |---- mask (np.array) the inpainting mask with value 1 on region to inpaint and zero otherwise.
        """
        h, w = shape
        mask = np.zeros(shape)
        # draw random number of patches
        for _ in range(np.random.randint(low=self.n_draw[0], high=self.n_draw[1])):
            n_vertex = np.random.randint(low=self.vertex[0], high=self.vertex[1])
            brush_width = np.random.randint(low=self.brush_width[0], high=self.brush_width[1])
            start_x, start_y = int(np.random.normal(w/2, w/8)), int(np.random.normal(h/2, h/8))
            #start_x, start_y = np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)

            beta = np.random.uniform(low=0, high=6.28)
            for i in range(n_vertex):
                angle = beta + np.random.uniform(low=self.angle[0], high=self.angle[1])
                length = np.random.randint(low=self.length[0], high=self.length[1])
                if i % 2 == 0:
                    angle = np.pi + angle #2 * np.pi - angle # reverse mode
                # draw line
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
                # set new start point
                start_x, start_y = end_x, end_y

        # salt and pepper
        for _ in range(np.random.randint(low=self.n_salt_pepper[0], high=self.n_salt_pepper[1])):
            start_x, start_y = np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)
            r = np.random.randint(low=self.salt_peper_radius[0], high=self.salt_peper_radius[1])
            cv2.circle(mask, (start_x, start_y), r, 1.0, -1)

        return mask

class ImgMaskDataset(data.Dataset):
    """
    Dataset object to load an image and mask together.
    """
    def __init__(self, data_df, data_path, augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                 tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)], window=None, output_size=256):
        """
        Build a dataset for loading image and mask.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a columns 'im_fn' with
            |               image filepath and a column 'mask_fn' with mask filepath.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for image intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
        OUTPUT
            |---- RSNA_Inpaint_dataset (torch.Dataset) the RSNA dataset for inpainting.
        """
        super(ImgMaskDataset, self).__init__()
        self.data_df = data_df
        self.data_path = data_path
        self.window = window

        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size),
                                    tf.ToTorchTensor())

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
        Extract the image and mask sepcified by idx.
        ----------
        INPUT
            |---- idx (int) the sample index in self.data_df.
        OUTPUT
            |---- im (torch.tensor) the image with dimension (1 x H x W).
            |---- mask (torch.tensor) the mask with dimension (1 x H x W).
            |---- idx (torch.tensor) the data index in the data_df.
        """
        # load dicom and recover the CT pixel values
        im = io.imread(os.path.join(self.data_path, self.data_df.iloc[idx].im_fn))
        mask = io.imread(os.path.join(self.data_path, self.data_df.iloc[idx].mask_fn))
        # Window the CT-scan
        if self.window:
            im = window_ct(im, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # transform image
        im, mask = self.transform(im, mask)

        return im, mask, torch.tensor(idx)

class RSNA_FCDD_dataset(data.Dataset):
    """  """
    def __init__(self, data_df, data_path, artificial_anomaly=True, anomaly_proba=0.5,
                 augmentation_transform=[tf.Translate(low=-0.1, high=0.1), tf.Rotate(low=-10, high=10),
                                         tf.Scale(low=0.9, high=1.1), tf.HFlip(p=0.5)],
                 window=None, output_size=256,
                 drawing_params=dict(n_ellipse=(1,10), major_axis=(1,25), minor_axis=(1,25),
                                     rotation=(0,2*np.pi), intensity=(0.1, 1), noise=None)):
        """
        Build a dataset for the RSNA dataset for FCDD training.
        ----------
        INPUT
            |---- data_df (pd.DataFrame) the input dataframe of samples. Each row must contains a filename and a columns
            |           Hemorrhage specifying if slice has or not an hemorrhage.
            |---- data_path (str) path to the root of the dataset folder (until where the samples' filnames begins).
            |---- artificial_anomaly (bool) whether to generate anomalies with drawing of ellipse on top of image. If False,
            |           the dataset will return labled hemorrhagous as anomalies.
            |---- anomaly_proba (float in [0.0,1.0]) the probability to geenrate an artificial anomaly. Ignored if
            |           artificial_anomaly is False.
            |---- augmentation_transform (list of transofrom) data augmentation transformation to apply.
            |---- window (tuple (center, width)) the window for CT intensity rescaling. If None, no windowing is performed.
            |---- output_size (int) the dimension of the output (H = W).
            |---- drawing_params (dict) the parameters to be passed to the ellipse drawing method.
        OUTPUT
            |---- RSNA_FCDD_dataset (data.Dataset)
        """
        super().__init__()
        assert 0.0 <= anomaly_proba <= 1.0, f"Probability of anomaly must be in [0.0 , 1.0]. Given {anomaly_proba}."
        self.data_df = data_df
        self.data_path = data_path
        self.artificial_anomaly = artificial_anomaly
        self.anomaly_proba = anomaly_proba
        self.window = window

        self.transform = tf.Compose(*augmentation_transform,
                                    tf.Resize(H=output_size, W=output_size),
                                    tf.ToTorchTensor())

        self.drawing_params = drawing_params

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
            |---- mask (torch.tensor) the inpaining mask with dimension (1 x H x W).
        """
        # load dicom and recover the CT pixel values
        dcm_im = pydicom.dcmread(self.data_path + self.data_df.iloc[idx].filename)
        im = (dcm_im.pixel_array * float(dcm_im.RescaleSlope) + float(dcm_im.RescaleIntercept))
        # Window the CT-scan
        if self.window:
            im = window_ct(im, win_center=self.window[0], win_width=self.window[1], out_range=(0,1))
        # transform image
        im = self.transform(im)
        label = self.data_df.iloc[idx].Hemorrhage

        if self.artificial_anomaly and (np.random.rand() < self.anomaly_proba) and (label == 0):
            # get a mask
            anomalies = tf.ToTorchTensor()(self.draw_ellipses((im.shape[1], im.shape[2]), **self.drawing_params))
            im = torch.where(anomalies > 0, anomalies, im)
            label = 1

        return im, torch.tensor(label), torch.tensor(idx)

    @staticmethod
    def draw_ellipses(im_shape, n_ellipse=(1,10), major_axis=(1,25), minor_axis=(1,25), rotation=(0,2*np.pi),
                      intensity=(0.1, 1), noise=None):
        """
        Draw mltiple ellipses with randomly samples parametrers and intensity.
        ----------
        INPUT
            |---- im_shape (tuple) shape of the image to generate.
            |---- n_ellipse (2-tuple low,high) the range to sample the number of ellipse from.
            |---- major_axis (2-tuple low,high) the range to sample the major_axis of ellipse from.
            |---- minor_axis (2-tuple low,high) the range to sample the minor axis of ellipse from.
            |---- rotation (2-tuple low,high) the range to sample the angle of ellipse from (in radian).
            |---- intensity (2-tuple low,high) the range to sample the pixel intensity of ellipse from.
            |---- noise (float) whether to add gaussian noise to the elipse value. If None, no noise is added.
        OUTPUT
            |---- out (np.array) the grayscale image with a random set of ellipse drawn.
        """
        h, w = im_shape
        out = np.zeros(im_shape)
        # generate a random number of ellipses
        for _ in range(np.random.randint(low=n_ellipse[0], high=n_ellipse[1])):
            # get center
            r, c = int(np.random.normal(w/2, w/6)), int(np.random.normal(h/2, h/6))
            # get major/minor axis
            max_ax = np.random.uniform(low=major_axis[0], high=major_axis[1])
            min_ax = np.random.uniform(low=minor_axis[0], high=min(minor_axis[1], max_ax))
            # get angle
            rot = np.random.uniform(low=rotation[0], high=rotation[1])
            # compute ellipse coordinates
            rr, cc = skimage.draw.ellipse(r, c, min_ax, max_ax, shape=im_shape, rotation=rot)
            # draw ellipse with or without noise
            if noise:
                gs_val = np.random.uniform(low=intensity[0], high=intensity[1])
                out[rr, cc] = np.clip(np.random.normal(gs_val, noise, size=len(rr)), 0.0, 1.0)
            else:
                out[rr, cc] = np.random.uniform(low=intensity[0], high=intensity[1])

        return out


#%%
# def random_ff_mask(shape):
#     """
#
#     """
#     h, w = shape
#     mask = np.zeros(shape)
#     # get drawing params
#     for _ in range(np.random.randint(low=1, high=4)):
#         n_vertex = np.random.randint(low=5, high=15+1)
#         brush_width = np.random.randint(low=10, high=35+1)
#         start_x, start_y = int(np.random.normal(w/2, w/8)), int(np.random.normal(h/2, h/8))
#         #start_x, start_y = np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)
#
#         for i in range(n_vertex):
#             angle = np.random.uniform(low=0.0, high=np.pi*2)
#             length = np.random.randint(low=10, high=30)
#             if i % 2 == 0:
#                 angle = np.pi + angle #2 * np.pi - angle # reverse mode
#             # draw line
#             end_x = (start_x + length * np.sin(angle)).astype(np.int32)
#             end_y = (start_y + length * np.cos(angle)).astype(np.int32)
#             cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
#             # set new start point
#             start_x, start_y = end_x, end_y
#
#     # salt and pepper
#     for _ in range(np.random.randint(low=0, high=5)):
#         start_x, start_y = np.random.randint(low=0, high=w), np.random.randint(low=0, high=h)
#         r = np.random.randint(low=1, high=3)
#         cv2.circle(mask, (start_x, start_y), r, 1.0, -1)
#
#     if np.random.random() > 0.5:
#         mask = np.flip(mask, axis=0)
#
#     if np.random.random() > 0.5:
#         mask = np.flip(mask, axis=1)
#
#     return mask
#
#
# #%%
# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(4,4,figsize=(10,10))
# for ax in axs.reshape(-1):
#     ax.imshow(random_ff_mask((256,256)), cmap='gray')
# plt.show()

#%%













#
