"""
author: Antoine Spahr

date : 29.09.2020

----------

TO DO :
- check interpolation order for resize (1 or 3) + for other transform (need order 3 ?)
- check if need a depth padding for 3D nifti for evaluation (so that whole volume can be passed and still be cut in half several times)
"""
import torchvision.transforms
import torch
import numpy as np
import skimage.transform
import scipy.ndimage

class Compose:
    """
    Compose, in a sequential fashion, multiple transforms than can handle an
    image and a mask in their __call__.
    """
    def __init__(self, *transformations, pass_mask=True):
        """
        Build a sequential composition of transforms.
        ----------
        INPUT
            |---- *transformations (args) list of transforms to compose.
        OUTPUT
            |---- Transformation
        """
        self.transformations = transformations

    def __call__(self, image, mask=None):
        """
        Passes the image (and mask) through all the transforms.
        ----------
        INPUT
            |---- image () the input image.
            |---- mask () the input mask.
        OUTPUT
            |---- image () the transformed image.
            |---- mask () the transformed mask.
        """
        if mask is not None:
            for f in self.transformations:
                image, mask = f(image, mask)
            return image, mask
        else:
            for f in self.transformations:
                image = f(image)
            return image

    def __str__(self):
        """
        Transform printing format
        """
        tf_names = [str(t) for t in self.transformations]
        max_size = max(len(x) for x in tf_names)
        link = '\n' + '|'.center(max_size) + '\n' + 'V'.center(max_size) + '\n'
        return link.join([t.center(max_size) for t in tf_names])

class RandomZCrop:
    """
    Randomly crop the 3D image and mask along the z-dimension.
    """
    def __init__(self, Z=64):
        """
        Build a to random Z-crop transform.
        ----------
        INPUT
            |---- Z (int) the number of slice to take along z.
        OUTPUT
            |---- RandomZCrop transform.
        """
        self.Z = Z

    def __call__(self, image, mask=None):
        """
        Randomly crop the np.array image and mask along Z.
        ----------
        INPUT
            |---- image (3D np.array) the image to crop.
            |---- mask (3D np.array) the mask to crop.
        OUTPUT
            |---- image (3D np.array) the cropped image.
            |---- mask (3D np.array) the cropped mask.
        """
        assert image.ndim == 3 and mask.ndim == 3, 'Input must be 3D numpy arrays!'
        assert image.shape[2] > self.Z, f'Input image z-dimension ({image.shape[2]}) must be larger that the crop size {self.Z}.'
        if mask:
            assert mask.shape[2] > self.Z, f'Input mask z-dimension ({mask.shape[2]}) must be larger that the crop size {self.Z}.'

        z_idx = np.random.randint(0, image.shape[2] - self.Z)
        image = image[:, :, z_idx:z_idx+self.Z]
        if mask is not None:
            mask = mask[:, :, z_idx:z_idx+self.Z]
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomZCrop(Z={self.Z})"

class Resize:
    """
    Resize the image and mask to the given height and width.
    """
    def __init__(self, H=256, W=256, pass_mask=True):
        """
        Build a to resize transform.
        ----------
        INPUT
            |---- H (int) the new Height.
            |---- W (int) the new Width.
        OUTPUT
            |---- Rotate transform.
        """
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        """
        Resize the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        image = skimage.transform.resize(image, (self.H, self.W, *image.shape[2:]), order=1)#, cval=image.min())
        if mask is not None:
            mask = skimage.transform.resize(mask, (self.H, self.W, *mask.shape[2:]), order=0, preserve_range=True)
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Resize(H={self.H}, W={self.W})"

class Translate:
    """
    Randomly translate the image and mask in a range of fraction.
    """
    def __init__(self, low=-0.1, high=0.1):
        """
        Build a to translate transform.
        ----------
        INPUT
            |---- low (float) the lower bound of the translate fraction.
            |---- high (float) the upper bound of the translate fraction.
        OUTPUT
            |---- Translate transform.
        """
        self.low = low
        self.high = high

    def __call__(self, image, mask=None):
        """
        Randomly translate the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- (mask) (np.array) the converted mask.
        """
        # randomly sample a shift in x and y. No shift in z
        shift = [np.random.uniform(low=image.shape[0]*self.low, high=image.shape[0]*self.high),
                 np.random.uniform(low=image.shape[1]*self.low, high=image.shape[1]*self.high)]
        shift += [0] * (image.ndim - 2)

        image = scipy.ndimage.shift(image, shift, order=3)#, cval=image.min())
        if mask is not None:
            mask = scipy.ndimage.shift(mask, shift, order=0)
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Translate(low={self.low}, high={self.high})"

class Scale:
    """
    Randomly scale the image and mask in a range of factor.
    """
    def __init__(self, low=0.9, high=1.1):
        """
        Build a to scale transform.
        ----------
        INPUT
            |---- low (float) the lower bound of the fraction.
            |---- high (float) the upper bound of the fraction.
        OUTPUT
            |---- Scale transform.
        """
        self.low = low
        self.high = high

    def __call__(self, image, mask=None):
        """
        Randomly scale the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        # randomly sample a scale in x and y. No scale in z
        scale_factor = np.random.uniform(low=self.low, high=self.high)
        scales = [scale_factor] * 2
        scales += [1] * (image.ndim - 2)


        adjust_h = abs((np.round(scale_factor*image.shape[0]) - image.shape[0]) / 2)
        adjust_w = abs((np.round(scale_factor*image.shape[1]) - image.shape[1]) / 2)

        adjust_list = [(int(np.floor(adjust_h)), int(np.ceil(adjust_h))),
                       (int(np.floor(adjust_w)), int(np.ceil(adjust_w)))] + [(0,0)] * (image.ndim - 2)

        if scale_factor >= 1:
            image = skimage.util.crop(scipy.ndimage.zoom(image, scales, order=3), adjust_list)
            if mask is not None:
                mask = skimage.util.crop(scipy.ndimage.zoom(mask, scales, order=0), adjust_list)
        else:
            image = skimage.util.pad(scipy.ndimage.zoom(image, scales, order=3), adjust_list)#, constant_values=image.min())
            if mask is not None:
                mask = skimage.util.pad(scipy.ndimage.zoom(mask, scales, order=0), adjust_list)

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Scale(low={self.low}, high={self.high})"

class Rotate:
    """
    Randomly rotate the image and mask in a range of angles.
    """
    def __init__(self, low=-10, high=10):
        """
        Build a to rotate transform.
        ----------
        INPUT
            |---- low (float) the lower bound of the angle in degree.
            |---- high (float) the upper bound of the angle in degree.
        OUTPUT
            |---- Rotate transform.
        """
        self.low = low
        self.high = high

    def __call__(self, image, mask=None):
        """
        Randomly rotate the np.array image and mask (in x, y).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        # randomly sample an angle
        angle = np.random.uniform(low=self.low, high=self.high)

        image = scipy.ndimage.rotate(image, angle, axes=(1,0), order=3, reshape=False)#, cval=image.min())
        if mask is not None:
            mask = scipy.ndimage.rotate(mask, angle, axes=(1,0), order=0, reshape=False)
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"Rotate(low={self.low}, high={self.high})"

class HFlip:
    """
    Randomly flip the image and mask.
    """
    def __init__(self, p=0.5):
        """
        Build a horizontal-flip transform.
        ----------
        INPUT
            |---- p (float) probability of flipping. Must be in [0,1]
        OUTPUT
            |---- H-Flip tranform.
        """
        assert p >= 0 or p <= 1, f'Probability must be between 0 and 1. Given : {p}'
        self.p = p

    def __call__(self, image, mask=None):
        """
        Randomly flip horizontally the np.array image and mask.
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        if np.random.random() < self.p:
            image = np.array(np.flip(image, axis=1)) # reconvert in array to make it contiguous
            if mask is not None:
                mask = np.array(np.flip(mask, axis=1))

        if mask is not None:
            return image, mask
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return f"HFlip(p={self.p})"

class ToTorchTensor:
    """
    Convert the image (and mask) to a torch.Tensor.
    """
    def __init__(self):
        """
        Build a to tensor transform.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- ToTensor transform.
        """

    def __call__(self, image, mask=None):
        """
        Convert the image and mask (np.array) to a torch.Tensor
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (torch.Tensor) the converted image.
            |---- mask (torch.Tensor) the converted mask.
        """
        image = torch.from_numpy(image).unsqueeze(dim=0)#torchvision.transforms.ToTensor()(image)
        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(dim=0)#torchvision.transforms.ToTensor()(mask)
            return image, mask.bool()
        else:
            return image

    def __str__(self):
        """
        Transform printing format
        """
        return "ToTorchTensor()"

class RandomPatchSwap:
    """
    Corrupt an input image by swapping random crops within itself. (as in Chen 2019).
    """
    def __init__(self, n=10, w=15, h=15):
        """
        Build a RandomPatchSwap transform.
        ----------
        INPUT
            |---- n (int) the number of swap to perform.
            |---- w (int) the width of the patch to swap.
            |---- h (int) the height of the patch to swap.
        OUTPUT
            |---- a RandomPatchSwap transform.
        """
        self.n = n
        self.h = h
        self.w = w

    def __call__(self, image, mask=None):
        """
        Swap randomly several crop within the image (and mask).
        ----------
        INPUT
            |---- image (np.array) the image to convert.
            |---- mask (np.array) the mask to convert.
        OUTPUT
            |---- image (np.array) the converted image.
            |---- mask (np.array) the converted mask.
        """
        image = image.copy()
        mask = mask.copy() if mask is not None else mask
        for _ in range(self.n):
            # get patches to swap
            p1, p2 = None, None
            while self.has_overlap(p1, p2):
                p1 = (np.random.randint(low=0, high=image.shape[0]-self.h), np.random.randint(low=0, high=image.shape[1]-self.w))
                p2 = (np.random.randint(low=0, high=image.shape[0]-self.h), np.random.randint(low=0, high=image.shape[1]-self.w))

            # swap patches content
            patch1 = image[p1[0]:p1[0]+self.h, p1[1]:p1[1]+self.w].copy()
            image[p1[0]:p1[0]+self.h, p1[1]:p1[1]+self.w] = image[p2[0]:p2[0]+self.h, p2[1]:p2[1]+self.w]
            image[p2[0]:p2[0]+self.h, p2[1]:p2[1]+self.w] = patch1

            # crop mask as well if passed
            if mask is not None:
                patch1 = mask[p1[0]:p1[0]+self.h, p1[1]:p1[1]+self.w].copy()
                mask[p1[0]:p1[0]+self.h, p1[1]:p1[1]+self.w] = mask[p2[0]:p2[0]+self.h, p2[1]:p2[1]+self.w]
                mask[p2[0]:p2[0]+self.h, p2[1]:p2[1]+self.w] = patch1

        if mask is not None:
            return image, mask
        else:
            return image

    def has_overlap(self, p1, p2):
        """
        Check if there's any overlap between p1 and p2.
        ----------
        INPUT
            |---- p1 (tuple) top left corner of first rectangle (row, col).
            |---- p2 (tuple) top left corner of second rectangle (row, col).
        OUTPUT
            |---- overlap (bool) True if the two rectangles overlap.
        """
        if p1 is None or p2 is None:
            return True
        else:
            return (abs(p1[0] - p2[0]) <= self.h) and (abs(p1[1] - p2[1]) <= self.w)

    def __str__(self):
        """
        Transform printing format
        """
        return f"RandomPatchSwap(n={self.n}, w={self.w}, h={self.h})"






#
