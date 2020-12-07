"""
author: Antoine Spahr

date : 15.10.2020

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

@click.command()
@click.argument('csv_info_fn', type=click.Path(exists=True))
@click.option('--output_fn', type=click.Path(exists=False), default=None,
              help='Filename to save the csv file. Default is data location + slice_info.csv')
def main(csv_info_fn, output_fn):
    """
    Rearrange the data info csv file (found at csv_info_fn) of the RSNA ICH dataset.
    """
    print(f'>>> Loading RSNA data info from {csv_info_fn}')
    info_df = pd.read_csv(csv_info_fn)
    print('>>> Rearranging the RSNA data info.')
    # extract id and hemorrhage type from ID
    df = info_df.join(info_df['ID'].str.split('_', 2, expand=True).rename(columns={0:'ID1', 1:'ID2', 2:'Type'})) \
                .drop(columns=['ID', 'ID1']) \
                .rename(columns={'ID2':'ID'})
    # get each type as a column
    df = df.groupby(['ID', 'Type']).max().unstack(level=-1)
    df.columns = df.columns.droplevel()
    df.columns.name = None
    df.reset_index(inplace=True)
    # Add filename columns
    df['filename'] = 'ID_' + df.ID + '.dcm'
    # rename any into Hemorrhage
    df.rename(columns={'any':'Hemorrhage'}, inplace=True)
    # remove corrupted file ID_6431af929.dcm
    df.drop(df[df.filename == 'stage_2_train/ID_6431af929.dcm'].index, inplace=True)
    # save csv
    save_path = output_fn if output_fn else '/'.join(csv_info_fn.split('/')[:-1]) + '/slice_info.csv'
    df.to_csv(save_path)
    print(f'>>> Rearranged data info saved at {save_path}')

if __name__ == "__main__":
    main()
