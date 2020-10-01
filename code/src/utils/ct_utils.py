"""
author: Antoine Spahr

date : 30.09.2020

----------

TO DO :
"""
import scipy.ndimage
import numpy as np

def window_ct(ct_scan, win_center=40, win_width=120, out_range=(0,1)):
    """
    Window the raw CT-scan with the given center and width in Hounsfeild unit (HU).
    ----------
    INPUT
        |---- ct_scan (np.array) the input ct_scan array in HU.
        |---- win_center (float) the center HU value of the window to use.
        |---- win_width (float) the width of the window in HU.
        |---- out_range (tuple (l, h)) the output range of the windowed ct_scan. Values resacled below l are clipped to
        |           l and values rescaled over h are clipped to h.
    OUTPUT
        |---- ct_scan (np.array) the windowed ct-scan.
    """
    # get window boundary in Hounsfeild unit
    win_min = win_center - win_width / 2
    win_max = win_center + win_width / 2
    # rescale to have pixel value zero at win_min and 255 at win_max
    out_delta = out_range[1] - out_range[0]
    ct_scan = (out_delta * (ct_scan - win_min) / (win_max - win_min)) + out_range[0]
    # clip value to the output range
    ct_scan[ct_scan < out_range[0]] = out_range[0]
    ct_scan[ct_scan > out_range[1]] = out_range[1]

    return ct_scan

def resample_ct(ct_scan, in_pixel_dim, out_pixel_dim=[1,1,1], preserve_range=True, order=3):
    """
    Resample the given CT-scan (volume) to a specified physical dimension.
    ----------
    INPUT
        |---- ct_scan (np.array) the ct-volume to resample with dimension.
        |---- in_pixel_dim (list) list of input pixel dimesnion. The number of elemnt must equal the number of
        |           dimension of ct_scan.
        |---- out_pixel_dim (list) list of output pixel dimensions. The number of elemnt must equal the number of
        |           dimension of ct_scan. Setting an element of the list to -1 will use the input dimension and there
        |           will be no resampling in that dimension.
        |---- preserve_range (bool) whether to rescale the output to the input's range.
        |---- order (int) the interpolation startegy used by scipy. Must be between 0 and 5.
    OUTPUT
        |---- resampled_scan (np.array) the resampled ct_scan with pixel dimension equal to out_pixel_dim.
    """
    # compute the resizing factors
    in_pixel_dim, out_pixel_dim = np.array(in_pixel_dim).astype(float), np.array(out_pixel_dim).astype(float)
    # keep input dim where output is -1
    out_pixel_dim[out_pixel_dim == -1] = in_pixel_dim[out_pixel_dim == -1]

    new_shape = np.round(ct_scan.shape * in_pixel_dim / out_pixel_dim)
    resize_factor = new_shape / ct_scan.shape
    # resample scan
    resampled_scan = scipy.ndimage.zoom(ct_scan, resize_factor, order=order)

    if preserve_range:
        in_range = (ct_scan.min(), ct_scan.max())
        resampled_scan = (in_range[1] - in_range[0])*(resampled_scan - resampled_scan.min()) / resampled_scan.ptp() + in_range[0]

    return resampled_scan
