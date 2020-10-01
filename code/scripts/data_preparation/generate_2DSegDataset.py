"""
author: Antoine Spahr

date : 30.09.2020

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
def main(input_data_path, output_data_path):
    """
    Convert the Volumetric CT data and mask (in NIfTI format) to a dataset of 2D images and mask in png.
    """
    # open data info dataframe
    info_df = pd.read_csv(input_data_path + 'hemorrhage_diagnosis_raw_ct.csv')
    # replace No-Hemorrhage to hemorrange
    info_df['Hemorrhage'] = 1 - info_df.No_Hemorrhage
    info_df.drop(columns='No_Hemorrhage', inplace=True)
    # open patient info dataframe
    patient_df = pd.read_csv(input_data_path + 'Patient_demographics.csv', header=1, skipfooter=2, engine='python') \
                   .rename(columns={'Unnamed: 0':'PatientNumber', 'Unnamed: 1':'Age',
                                    'Unnamed: 2':'Gender', 'Unnamed: 8':'Fracture', 'Unnamed: 9':'Note'})
    patient_df[patient_df.columns[3:9]] = patient_df[patient_df.columns[3:9]].fillna(0).astype(int)
    # add columns Hemorrgae (any ICH)
    patient_df['Hemorrage'] = patient_df[patient_df.columns[3:8]].max(axis=1)

    # make patient directory
    if not os.path.exists(output_data_path): os.mkdir(output_data_path)
    if not os.path.exists(output_data_path + 'Patient_CT/'): os.mkdir(output_data_path + 'Patient_CT/')
    # iterate over volume to extract data
    output_info = []
    for n, id in enumerate(info_df.PatientNumber.unique()):#id, slice in zip(info_df.PatientNumber.values, info_df.SliceNumber.values):
        # read nii volume
        ct_nii = nib.load(input_data_path + f'ct_scans/{id:03}.nii')
        mask_nii = nib.load(input_data_path + f'masks/{id:03}.nii')
        # get np.array
        ct_vol = ct_nii.get_fdata()
        mask_vol = skimage.img_as_bool(mask_nii.get_fdata())
        # rotate 90° counter clockwise
        ct_vol = np.rot90(ct_vol, axes=(0,1))
        mask_vol = np.rot90(mask_vol, axes=(0,1))
        # window the ct volume to get better contrast of soft tissues
        ct_vol = window_ct(ct_vol, win_center=40, win_width=120, out_range=(0,1))

        if mask_vol.shape != ct_vol.shape:
            print(f'>>> Warning! The ct volume of patient {id} does not have '
                  f'the same dimension as the ground truth. CT ({ct_vol.shape}) vs Mask ({mask_vol.shape})')
        # make patient directory
        if not os.path.exists(output_data_path + f'Patient_CT/{id:03}/'): os.mkdir(output_data_path + f'Patient_CT/{id:03}/')
        # iterate over slices to save slices
        for i, slice in enumerate(range(ct_vol.shape[2])):
            ct_slice_fn =f'Patient_CT/{id:03}/{slice+1}.png'
            # save CT slice
            skimage.io.imsave(output_data_path + ct_slice_fn, skimage.img_as_uint(ct_vol[:,:,slice]), check_contrast=False)
            # save mask if some positive ICH
            if np.any(mask_vol[:,:,slice]):
                mask_slice_fn = f'Patient_CT/{id:03}/{slice+1}_HGE_Seg.png'
                skimage.io.imsave(output_data_path + mask_slice_fn, skimage.img_as_uint(mask_vol[:,:,slice]), check_contrast=False)
            else:
                mask_slice_fn = 'None'
            # add info to output list
            output_info.append({'PatientNumber':id, 'SliceNumber':slice+1, 'CT_fn':ct_slice_fn, 'mask_fn':mask_slice_fn})

            print_progessbar(i, ct_vol.shape[2], Name=f'Patient {id:03} {n+1:03}/{len(info_df.PatientNumber.unique()):03}',
                             Size=20, erase=False)

    # Make dataframe of outputs
    output_info_df = pd.DataFrame(output_info)
    # Merge with input info
    info_df = pd.merge(info_df, output_info_df, how='inner', on=['PatientNumber', 'SliceNumber'])
    # save df
    info_df.to_csv(output_data_path + 'ct_info.csv')
    print('>>> Data informations saved at ' + output_data_path + 'ct_info.csv')
    # save patient df
    patient_df.to_csv(output_data_path + 'patient_info.csv')
    print('>>> Patient informations saved at ' + output_data_path + 'patient_info.csv')


if __name__ == '__main__':
    main()
