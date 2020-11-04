"""
author: Antoine Spahr

date : 05.10.2020

----------

TO DO :
"""
import sys
sys.path.append('../../')
import click
import os
import glob
import pandas as pd
from dicom2nifti.convert_dicom import dicom_array_to_nifti
import dicom2nifti.settings
import pydicom

from src.utils.print_utils import print_progessbar

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--out_folder', type=click.Path(exists=False), default='', help='The directory where to save the nifti data. Default: where scripts is run.')
def main(input_path, out_folder):
    """
    Convert the qureAI CQ500 dataset (at input_path) from series of dicom to 3D NIfTI volumes.
    """
    print(f'>>> Start converting dicom series from {input_path} to NIfTI volumes saved at {out_folder}.')
    # adjust dicom conversion settings to not check for orthogonality
    dicom2nifti.settings.disable_validate_orthogonal()
    # Place holder for nifti file information
    out_info_list = []
    # iterate over subfolder
    dir_list = glob.glob(input_path + '*/')
    for n, patient_dir in enumerate(dir_list):
        # get patient CT ID
        ID = patient_dir.split('/')[-2]
        # iterate over patient's series
        dcm_list = []
        for dcm_fn in glob.glob(patient_dir + '*.dcm'):
            # read dicom and decompress it
            ds = pydicom.dcmread(dcm_fn)
            ds.decompress()
            dcm_list.append(ds)
        # convert the dicom serie into a nifti file
        out_path = out_folder + ID + '.nii'
        _ = dicom2nifti.convert_dicom.dicom_array_to_nifti(dcm_list, out_path)
        out_info_list.append({'id': int(ID), 'filename': ID + '.nii',
                              'n_slice': len(dcm_list)})

        print_progessbar(n, len(dir_list), '\tCT Scan', Size=40, erase=False)

    print(f'>>> {len(out_info_list)} NIfTI volumes successfully saved at {out_folder}.')
    # read info csv and merge with filepath
    in_df = pd.read_csv(input_path + 'ICH_probabilities.csv', index_col=0)
    fn_df = pd.DataFrame(out_info_list)
    df = pd.merge(fn_df, in_df, left_on='id', right_index=True, how='outer')
    # save data info
    df.to_csv(out_folder + 'info.csv')
    print(f">>> NIfTI file informations saved at {out_folder + 'info.csv'}.")

if __name__ == '__main__':
    main()
