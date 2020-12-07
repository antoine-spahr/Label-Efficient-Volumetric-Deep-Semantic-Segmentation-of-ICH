"""
author: Antoine Spahr

date : 01.12.2020

----------

To Do:
    -
"""

import os
import logging
import warnings
import torch
import torch.utils.data as data
import numpy as np
import skimage
import skimage.morphology
import skimage.filters
import skimage.io as io
import scipy.stats

from src.utils.print_utils import print_progessbar

class InpaintAnomalyDetector:
    """
    Define an anomaly detector based on multiple inpainting of each pixels.
    """
    def __init__(self, inpaint_net, grid_hole=(32,32), grid_step=1, dilation_radius=3, n_iter=10,
                 alpha01=0.0, alpha02=1.0, alpha1=1.0, alpha2=1.5, use_wasserstein=False,
                 grid_anomaly_inpaint=(128,128), shuffle_AD_mask_loader=True, device='cuda', batch_size=8):
        """
        Build a anomaly detector based on inpainting.
        ----------
        INPUT
            |---- inpaint_net (torch.nn.Module) an inpainting network to use to detect anomalies. It should be a pytorch
            |               nn.Module that takes the image (uncorrupted) and inpaint mask as input, and return the inpainted
            |               image. All input and outputs are 4-Dimensional [B, C, H, W].
            |---- grid_hole (2-tuple of int (h, w)) the dimension of the hole of each grid element for inpainting.
            |---- grid_step (int) the step for sliding the grid over the image to obtained different mask. A larger step
            |               will reduce the number of grid mask generated and thus reduce the computation time.
            |---- dilation_radius (int) the radius of the disk structuring element for dilation of the anomaly mask prior
            |               of correction inpainting.
            |---- n_iter (int) the number of iteration of mask cleaning to perform.
            |---- alpha01 (float) fraction of IQR to take to define the low hysteresis threshold in the initialization
            |               phase. The threshold is defined as t_low = q75(D0) + alpha01*IQR(D0).
            |---- alpha01 (float) fraction of IQR to take to define the high hysteresis threshold in the initialization
            |               phase. The threshold is defined as t_high = q75(D0) + alpha02*IQR(D0).
            |---- alpha01 (float) fraction of IQR to take to define the low hysteresis threshold in the iterative phase.
            |               The threshold is defined as t_low = q75(Di) + alpha1*IQR(Di).
            |---- alpha01 (float) fraction of IQR to take to define the high hysteresis threshold in the iterative phase.
            |               The threshold is defined as t_high= q75(Di) + alpha2*IQR(Di).
            |---- use_wasserstein (bool) whether to use the wasserstein-1 distance instead of the normal approcimation of DKL.
            |---- grid_anomaly_inpaint (2-tuple of int) the dimension of patches to use for inpainting the anomalies (to
            |               correct the image). Inpainting by patch enables a more stable inpainting when there are many
            |               pixels missing.
            |---- shuffle_AD_mask_loader (bool) whether to shuffle the different inpainting patch when inpaining anomalies.
            |               If True, it removed the bias of starting the inpainting from the upper left corner to the lower
            |               right one.
            |---- device (str) the device on which to process.
            |---- batch_size (int) the batch_size to use for the error sampling through grid inpainting.
        OUTPUT
            |---- InpaintAnomalyDetector
        """
        self.inpaint_net = inpaint_net
        self.grid_hole = grid_hole
        self.grid_step = grid_step
        self.dilation_radius = dilation_radius
        self.n_iter = n_iter
        self.use_wasserstein = use_wasserstein

        assert alpha01 <= alpha02, f"alpha01 must be smaller or equal to alpha02. Given alpha01 = {alpha01} and alpha02 = {alpha02}."
        assert alpha1 <= alpha2, f"alpha1 must be smaller or equal to alpha2. Given alpha1 = {alpha1} and alpha2 = {alpha2}."
        self.alpha01 = alpha01
        self.alpha02 = alpha02
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.grid_anomaly_inpaint = grid_anomaly_inpaint
        self.shuffle_AD_mask_loader = shuffle_AD_mask_loader

        self.device = device
        self.batch_size = batch_size

        self.inpaint_net = self.inpaint_net.to(self.device)

    def detect(self, image, save_dir=None, verbose=False):
        """
        Detect and localize anomalies on the passed image using grid inpainting.
        ----------
        INPUT
            |---- image (torch.tensor or array like) the image to process with dimension [C, H, W] or [H, W].
            |---- save_dir (str) specified the directory path where to save intermediary results. If None, nothing is saved.
            |---- verbose (bool) whether to print in consol or log the processing evolution.
        OUTPUT
            |---- mA (np.array) the anomaly mask where a value of 1 highlight anomalies. It has dimension [H, W].
        """
        # Check input image is torch.tensor or can be converted to it
        if not isinstance(image, torch.Tensor):
            try:
                image = torch.Tensor(image)
            except TypeError:
                raise TypeError(f"Cannot convert input image of type {type(image)} to torch.Tensor.")
        # add channel dim if none
        if image.ndim == 2:
            image = image.unsqueeze(0)
        # check that image is [C, H, W]
        assert image.ndim == 3, f"The input image must be 3D dimensional [C, H, W] but image with shape {image.shape} was given."

        # if verbose define where info are printed (print of log)
        if verbose:
            if logging.getLogger().hasHandlers():
                logger = logging.getLogger()
                verbose_fn = logger.info
            else:
                verbose_fn = print

        # make save_directory if necessary
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        #--------------------------------------
        #      PHASE 1 : Detect Anomalies
        #--------------------------------------
        # STEP 1 Make Grid
        if verbose: verbose_fn(f"Generating inpainting grids with holes {self.grid_hole} and step {self.grid_step}.")
        all_grid_masks = self._get_grid_mask(image.shape[1:], hole_size=self.grid_hole, step=self.grid_step)
        # sanity check
        if verbose: verbose_fn(f"All pixel covered equally by the grid masks : {np.all(all_grid_masks.sum(axis=0) == all_grid_masks.sum(axis=0)[0,0])}")

        # STEP 2 Compute pixelwise inpainting error over
        if verbose: verbose_fn("Computing the pixelwise inpaining error sample.")
        errors = self._pixelwise_error(image, all_grid_masks, verbose=verbose)
        errors = errors.mean(axis=1) # reduction on Channels

        if self.use_wasserstein:
              # STEP 3 First Estimatation of error distribution of each pixel
              p0 = np.random.normal(loc=np.zeros(image.shape[1:]), scale=np.ones(image.shape[1:])*np.quantile(errors.std(axis=0), 0.25), size=errors.shape)
              pA = np.random.normal(loc=errors.mean(axis=0), scale=errors.std(axis=0), size=errors.shape)
              # STEP 4 Compute W1 distance map between pA and p0
              D0 = self.pixelwise_wassertein_1(p0, pA)
        else:
              # STEP 3 First Estimatation of error distribution of each pixel
              if verbose: verbose_fn("Estimating prior and posterior distribution parameters.")
              p0 = (np.zeros(image.shape[1:]), np.ones(image.shape[1:])*np.quantile(errors.std(axis=0), 0.25))
              pA = (errors.mean(axis=0), errors.std(axis=0))
              # STEP 4 Compute KL distance map between pA and p0
              if verbose: verbose_fn("Computing the distance map between prior and posterior distribution.")
              D0 = self.kl_divergence_normal(p0, pA)

        # STEP 5 Hysteresis Threshold D0
        t_low = np.quantile(D0, 0.75) + (np.quantile(D0, 0.75) - np.quantile(D0, 0.25)) * self.alpha01
        t_high = np.quantile(D0, 0.75) + (np.quantile(D0, 0.75) - np.quantile(D0, 0.25)) * self.alpha02
        print(f"Thresholding distance map with t_low : {t_low:.5f} and t_high {t_high:.5f}")
        mA0 = skimage.filters.apply_hysteresis_threshold(D0, t_low, t_high)

        # STEP 6 Remove detected anomaly with normal element using the inpainter
        if verbose: verbose_fn(f"Inpaint anomalies on original image.")
        mA_dilated = skimage.morphology.binary_dilation(mA0, selem=skimage.morphology.disk(self.dilation_radius))
        im_corrected_0 = self._inpaint_anomaly(image, torch.tensor(mA_dilated).unsqueeze(0))

        if save_dir:
            io.imsave(os.path.join(save_dir, 'sqrtD0.png'), skimage.img_as_ubyte(skimage.exposure.rescale_intensity(np.sqrt(D0+1e-12), out_range=(0.0,1.0))), check_contrast=False)
            io.imsave(os.path.join(save_dir, 'mA0.png'), skimage.img_as_ubyte(mA0), check_contrast=False)
            io.imsave(os.path.join(save_dir, 'im_corrected_0.png'), skimage.img_as_ubyte(skimage.exposure.rescale_intensity(im_corrected_0.cpu().numpy()[0], out_range=(0.0,1.0))), check_contrast=False)

        #--------------------------------------------------
        #      PHASE 2 : Clean Iteratively Anomaly Mask
        #--------------------------------------------------
        mAi = mA0
        im_corrected = im_corrected_0
        if verbose: verbose_fn("Iterative Ajustment of the anomaly mask.")

        for i in range(self.n_iter):
            # STEP 2.1 Compute pixelwise inpainting error over
            errors = self._pixelwise_error(im_corrected, all_grid_masks, verbose=verbose)
            errors = errors.mean(axis=1) # reduction along channels

            if self.use_wasserstein:
                  # STEP 2.2 Estimatation of error distribution of each pixel of corrected image
                  p0i = np.random.normal(loc=np.zeros(image.shape[1:]), scale=np.ones(image.shape[1:])*np.quantile(errors.std(axis=0), 0.25), size=errors.shape)
                  pAi = np.random.normal(loc=errors.mean(axis=0), scale=errors.std(axis=0), size=errors.shape)
                  # STEP 2.3 Compute W1 distance map between pAi and p0i
                  Di = self.pixelwise_wassertein_1(p0i, pAi)
            else:
                  # STEP 2.2 Estimatation of error distribution of each pixel of corrected image
                  p0i = (np.zeros(image.shape[1:]), np.ones(image.shape[1:])*np.quantile(errors.std(axis=0), 0.25))
                  pAi = (errors.mean(axis=0), errors.std(axis=0))
                  # STEP 2.3 Compute KL distance map between pAi and p0i
                  Di = self.kl_divergence_normal(p0i, pAi)

            # STEP 2.4 Threshold Di
            t_low = np.quantile(Di, 0.75) + (np.quantile(Di, 0.75) - np.quantile(Di, 0.25)) * self.alpha1
            t_high = np.quantile(Di, 0.75) + (np.quantile(Di, 0.75) - np.quantile(Di, 0.25)) * self.alpha2
            mAi_normal = skimage.filters.apply_hysteresis_threshold(Di, t_low, t_high)

            # STEP 2.5 Get new anomaly mask by removing region that appeared abnormal on corrected images
            mAi = (mAi == 1) & (mAi_normal == 0)

            # STEP 2.6 reinpaint the original image with the new anomaly mask to obtain a better corrected image
            mA_dilated = skimage.morphology.binary_dilation(mAi, selem=skimage.morphology.disk(self.dilation_radius))
            im_corrected = self._inpaint_anomaly(image, torch.tensor(mA_dilated).unsqueeze(0))

            if verbose:
                verbose_fn(f"| Step {i+1:03}/{self.n_iter:03} | Threshold Low {t_low:.5f} High {t_high:.5f} "
                           f"| Pixel difference : {int(mAi.sum() - mA0.sum())} |")

            if save_dir:
                io.imsave(os.path.join(save_dir, f'sqrtD{i+1}.png'), skimage.img_as_ubyte(skimage.exposure.rescale_intensity(np.sqrt(Di+1e-12), out_range=(0.0,1.0))), check_contrast=False)
                io.imsave(os.path.join(save_dir, f'mA{i+1}.png'), skimage.img_as_ubyte(mAi), check_contrast=False)
                io.imsave(os.path.join(save_dir, f'im_corrected_{i+1}.png'), skimage.img_as_ubyte(skimage.exposure.rescale_intensity(im_corrected.cpu().numpy()[0], out_range=(0.0,1.0))), check_contrast=False)

        return mAi

    @staticmethod
    def _get_grid_mask(shape, hole_size=(32, 32), step=4):
        """
        Generate list of mask to produce a gridded image in a way that each pixels is masked the same amount of time
        throughout different grid shifts. The grid_masks are then accessible throught the class attribute `grid_masks`.
        ----------
        INPUT
            |---- shape (tuple of int (h, w)) the dimension of the mask to generate.
            |---- hole_size (tuple of int (hole_h, hole_w)) the shape of each grid hole.
            |---- step (int) the pixel shift of the hole between two grid masks.
        OUTPUT
            |---- grid_list (np.array) the set of grid_masks with dimension [N_grid, h, w].
        """
        h, w = shape
        hole_h, hole_w = hole_size
        # make grid at resolution of hole size and make it bigger than image by two hole size
        a = np.zeros(h // hole_h + 2)
        a[np.arange(0,a.shape[0],2)] = 1
        b = np.zeros(w // hole_w + 2)
        b[np.arange(0,b.shape[0],2)] = 1
        grid = np.expand_dims(a, 1) * np.expand_dims(b, 0)
        # increase to have each pixels to the size of all
        grid = np.repeat(grid, hole_h, axis=0)
        grid = np.repeat(grid, hole_w, axis=1)
        # crop image size for different offset of the grid
        grid_list = []
        for i in range(0, 2 * hole_h, step):
            for j in range(0, 2 * hole_w, step):
                grid_list.append(grid[i:i+h, j:j+w])
        # stack all the
        return np.stack(grid_list, axis=2).transpose(2,0,1) # fisrt dim to select grid

    def _inpaint(self, im, mask):
        """
        Inpaint the input on mask using the inpainter.
        ----------
        INPUT
            |---- im (torch.tensor) image to inpaint with dimension (Batch x C x H x W) or (C x H x W).
            |---- mask (torch.tensor) mask defining region to inpaint (where mask = 1). It should have dimension
            |               (Batch x 1 x H x W) or (1 x H x W).
        OUTPUT
            |---- inpainted_im (torch.tensor) the image inpainted on the region defined by the mask with dimension
            |               (B x C x H x W) or (1 x C x H x W). Note that the input image is kept unchanged outside the
            |               inpainted region (where mask = 0).
        """
        # check input
        assert im.ndim == mask.ndim, f"Image and mask tensors must have the same number of dimension. Given {im.ndim} and {mask.ndim}."
        if im.ndim == 3 and mask.ndim == 3:
            im = im.unsqueeze(0)
            mask = mask.unsqueeze(0)
        assert (im.shape[0], im.shape[2:]) == (mask.shape[0], mask.shape[2:]), f"Image and mask's 1st, 3rd and/or 4th dimensions do not match. Given {im.shape} and {mask.shape}."

        with torch.no_grad():
            # make input float and put them to device
            im, mask = im.to(self.device).float().requires_grad_(False), mask.to(self.device).float().requires_grad_(False)
            # inpaint
            inpainted_im = self.inpaint_net(im * (1 - mask), mask)
            # keep only region to inpaint
            inpainted_im = im * (1 - mask) + inpainted_im * mask

        return inpainted_im

    def _pixelwise_error(self, input, grid_masks, verbose=False):
        """
        Generate a the pixelwise error sample of input when masked with grid.
        ----------
        INPUT
            |---- input (torch.tensor) the image on which the error sample is computed with the grids. It should have
            |               dimension [C, H, W] or [1, C, H, W].
            |---- grid_masks (np.array) the set of grid mask to use for inpainting. It should have dimension [N_grid, H, W].
            |---- verbose (bool) whether to print a progress bar of the inpainting process.
        OUTPUT
            |---- err (np.array) the pixelwise sample of inpainting errors with dimension [N_grid, C, H, W].
        """
        assert input.ndim == 3, f"Input must be 3 dimensional (C x H x W). Got {input.shape}"

        grid_dataset = data.TensorDataset(torch.tensor(grid_masks).unsqueeze(1))
        grid_loader = data.DataLoader(grid_dataset, batch_size=self.batch_size)

        input = input.unsqueeze(0) # add a batch dimension

        error_list = []
        for b, grid_mask_batch in enumerate(grid_loader):
            grid_mask_batch = grid_mask_batch[0]
            # repeat input to along batch dimension
            input_rep = input.repeat(grid_mask_batch.shape[0], 1, 1, 1).to(self.device)
            # inpaint im
            inpaint_im = self._inpaint(input_rep, grid_mask_batch)
            # compute difference to input
            error_list.append(inpaint_im - input_rep)

            if verbose:
                print_progessbar(b, len(grid_loader), Name='Grid Inpainting', Size=50, erase=True)

        # keep only inpainting error where the grid mask were present
        measures = torch.cat(error_list, dim=0) # [N_measure x C x H x W]
        grid_sample = torch.tensor(grid_masks).unsqueeze(1).repeat(1, input.shape[1], 1, 1) # use the whole grid array and repeat the channel dimension
        err = measures.permute(1,2,3,0)[grid_sample.permute(1,2,3,0) == 1]
        c, h, w = input.shape[1:]
        err = err.view(c, h, w, -1).permute(3,0,1,2) # [N_err, C, H, W]

        return err.cpu().numpy()

    @staticmethod
    def kl_divergence_normal(p1, p2):
        """
        Compute the normal approximation of the Kullbach-Leibler divergence.
        ----------
        INPUT
            |---- p1 (tuple of np.array (mu, sigma)) the first normal distriution as a tuple defined by an array of mean
            |               and an array of standard deviation.
            |---- p2 (tuple of np.array (mu, sigma)) the second normal distriution as a tuple defined by an array of mean
            |               and an array of standard deviation.
        OUTPUT
            |---- kl (np.array) kullback_leibler approximation between p1 and p2 for each of their element.
        """
        eps = 1e-12
        return np.log(p1[1] / (p2[1] + eps) + eps) + (p2[1]**2 + (p2[0]-p1[0])**2) / (2*p1[1]**2 + eps) - 1/2

    @staticmethod
    def pixelwise_wassertein_1(p1, p2):
        """
        Compute the Wasserstein-1 distance between two image-like sample
        ----------
        INPUT
            |---- p1 (np.array 3D) the first sample with dimension [Samp, H, W].
            |---- p2 (np.array 3D) the second sample with dimension [Samp, H, W].
        OUTPUT
            |---- W1 (np.array 2D) Wassersetin-1 distance between p1 and p2 for each of their on H,W.
        """
        assert p1.ndim == 3, f"P1 must be a 3D array. Given {p1.ndim}D."
        assert p2.ndim == 3, f"P2 must be a 3D array. Given {p2.ndim}D."
        assert p1.shape[1:] == p2.shape[1:], f"P1 and P2 must have the same size on dimension 1 and 2. Given P1 : {p1.shape}, P2 : {p2.shape}"
        # compute W1 for each pixel
        W1 = np.empty(p1.shape[1:])
        for i in range(p1.shape[1]):
          for j in range(p1.shape[2]):
            W1[i,j] = scipy.stats.wasserstein_distance(p1[:,i,j], p2[:,i,j])
        return W1

    def _inpaint_anomaly(self, im, anomaly_mask):
        """
        Inpaint image on mask by inpainting sequentially part of the image for more stable results.
        ----------
        INPUT
            |---- im (torch.tensor) the image to correct with dimension [C, H, W].
            |---- anomaly_mask (torch.tensor) the mask defining region to inpaint with dimension [1, H, W].
        OUTPUT
            |---- im_corr (torch.tensor) the coroected image by patch with dimension [C, H, W].
        """
        c, h, w = im.shape
        grid_h, grid_w = self.grid_anomaly_inpaint

        # generate list of masks
        n_grids = (h // grid_h, w // grid_w)
        grids = torch.zeros(n_grids[0]*n_grids[1], 1, h, w) # [N, 1, H, W]
        k = 0
        for i in range(n_grids[0]):
            for j in range(n_grids[1]):
                h_i,  w_j = i*grid_h, j*grid_w
                grids[k, :, h_i:h_i+grid_h, w_j:w_j+grid_w] = 1
                k += 1

        # repeat anomaly mask along batch_dimension and then multiply it with list of mask
        anomaly_mask = anomaly_mask.repeat(n_grids[0]*n_grids[1], 1, 1, 1)
        inpaint_mask = grids * anomaly_mask

        # make a loader
        loader = data.DataLoader(data.TensorDataset(inpaint_mask), batch_size=1, shuffle=self.shuffle_AD_mask_loader)
        # inpaint in each patch series
        im_corr = im.unsqueeze(0)
        for mask in loader:
            im_corr = self._inpaint(im_corr, mask[0])

        return im_corr.squeeze(0).cpu()
