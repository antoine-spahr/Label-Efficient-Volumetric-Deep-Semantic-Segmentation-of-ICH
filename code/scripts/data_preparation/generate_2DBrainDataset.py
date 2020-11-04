"""
author: Antoine Spahr

date : 04.11.2020

----------

TO DO :
"""
import sys
sys.path.append('../../')
import click
import os
import pandas as pd
import numpy as np
import nibabel as nib
import skimage
import skimage.io

from src.utils.ct_utils import window_ct
from src.utils.print_utils import print_progessbar

@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.option('--output_data_path', type=click.Path(exists=False), default=None, help='Where to save the 2D data.')
@click.option('--window', type=tuple, default=None, help='The windowing center and width as a tuple for CT intensity rescaling. Default: None. (Windowing not applied)')
def main(input_data_path, output_data_path, window):
    """
    Convert the Volumetric CT data and mask (in NIfTI format) to a dataset of 2D images in tif and masks in bitmap for the brain extraction.
    """
    # open data info dataframe
    info_df = pd.read_csv(os.path.join(input_data_path, 'info.csv'), index_col=0)
    # make patient directory
    if not os.path.exists(output_data_path): os.mkdir(output_data_path)
    # iterate over volume to extract data
    output_info = []
    for n, id in enumerate(info_df.id.values):
        # read nii volume
        ct_nii = nib.load(os.path.join(input_data_path, f'ct_scans/{id}.nii'))
        mask_nii = nib.load(os.path.join(input_data_path, f'masks/{id}.nii.gz'))
        # get np.array
        ct_vol = ct_nii.get_fdata()
        mask_vol = skimage.img_as_bool(mask_nii.get_fdata())
        # rotate 90° counter clockwise for head pointing upward
        ct_vol = np.rot90(ct_vol, axes=(0,1))
        mask_vol = np.rot90(mask_vol, axes=(0,1))
        # window the ct volume to get better contrast of soft tissues
        if window is not None:
            ct_vol = window_ct(ct_vol, win_center=window[0], win_width=window[1], out_range=(0,1))

        if mask_vol.shape != ct_vol.shape:
            print(f'>>> Warning! The ct volume of patient {id} does not have '
                  f'the same dimension as the ground truth. CT ({ct_vol.shape}) vs Mask ({mask_vol.shape})')
        # make patient directory
        if not os.path.exists(os.path.join(output_data_path, f'{id:03}/ct/')): os.makedirs(os.path.join(output_data_path, f'{id:03}/ct/'))
        if not os.path.exists(os.path.join(output_data_path, f'{id:03}/mask/')): os.makedirs(os.path.join(output_data_path, f'{id:03}/mask/'))
        # iterate over slices to save slices
        for i, slice in enumerate(range(ct_vol.shape[2])):
            ct_slice_fn =f'{id:03}/ct/{slice+1}.tif'
            # save CT slice
            skimage.io.imsave(os.path.join(output_data_path, ct_slice_fn), ct_vol[:,:,slice], check_contrast=False)
            is_low = True if skimage.exposure.is_low_contrast(ct_vol[:,:,slice]) else False
            # save mask if some brain on slice
            if np.any(mask_vol[:,:,slice]):
                mask_slice_fn = f'{id:03}/mask/{slice+1}_Seg.bmp'
                skimage.io.imsave(os.path.join(output_data_path, mask_slice_fn), skimage.img_as_ubyte(mask_vol[:,:,slice]), check_contrast=False)
            else:
                mask_slice_fn = 'None'
            # add info to output list
            output_info.append({'volume':id, 'slice':slice+1, 'ct_fn':ct_slice_fn, 'mask_fn':mask_slice_fn, 'low_contrast_ct':is_low})

            print_progessbar(i, ct_vol.shape[2], Name=f'Volume {id:03} {n+1:03}/{len(info_df.id):03}',
                             Size=20, erase=False)

    # Make dataframe of outputs
    output_info_df = pd.DataFrame(output_info)
    # save df
    output_info_df.to_csv(os.path.join(output_data_path, 'slice_info.csv'))
    print('>>> Slice informations saved at ' + os.path.join(output_data_path, 'slice_info.csv'))
    # save patient df
    info_df.to_csv(os.path.join(output_data_path, 'volume_info.csv'))
    print('>>> Volume informations saved at ' + os.path.join(output_data_path, 'volume_info.csv'))

if __name__ == '__main__':
    main()
