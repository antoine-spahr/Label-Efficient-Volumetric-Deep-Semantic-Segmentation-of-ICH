"""
author: Antoine Spahr

date : 07.12.2020

----------

To Do:
    -
"""
import os
import shutil
import glob
import sys
sys.path.append('../../')
import click
import pandas as pd

from src.utils.print_utils import print_progessbar

@click.command()
@click.argument("src_data_path", type=click.Path(exists=True))
@click.argument("src_data_info", type=click.Path(exists=True))
@click.argument("dst_data_path", type=click.Path(exists=True))
@click.argument("dst_data_info", type=click.Path(exists=True))
@click.option("--which", type=click.Choice(['mask', 'map']), default='map', help="what data to transfer : either 'mask' or 'map'.")
def main(src_data_path, src_data_info, dst_data_path, dst_data_info, which):
    """

    """
    # load src csv
    src_df = pd.read_csv(src_data_info, index_col=0)
    src_df = src_df[['id', 'slice', f'ad_{which}_fn']]

    # load dst csv
    dst_df = pd.read_csv(dst_data_info, index_col=0)

    # merge df
    df = pd.merge(dst_df, src_df, on=['id', 'slice'])
    df["attention_fn"] = df.apply(lambda row: f"{row['id']:03}" + os.sep + 'anomaly' + os.sep + os.path.basename(row[f'ad_{which}_fn']) if row[f'ad_{which}_fn'] != 'None' else 'None', axis=1)#df['id'] + os.sep + 'anomaly' + os.sep + os.path.basename(df[f'ad_{which}_fn'])

    # remove old folder and make new empty ones
    for i, id_i in enumerate(df.id.unique()):#(_, row) in enumerate(df.iterrows()):
        dir_i = os.path.join(dst_data_path, f'{id_i:03}/anomaly/')
        if os.path.isdir(dir_i):
            for fn in glob.glob(os.path.join(dir_i, '*.png')):
                os.remove(fn)
        else:
            os.makedirs(dir_i)
        print_progessbar(i, len(df.id.unique()), Name='Folder cleaning', Size=50)

    # for each sample : transfer file
    for i, (_, row) in enumerate(df.iterrows()):
        if os.path.basename(row[f'ad_{which}_fn']) != 'None':
            _ = shutil.copy2(os.path.join(src_data_path, row[f'ad_{which}_fn']), os.path.join(dst_data_path, row['attention_fn']))
        print_progessbar(i, len(df), Name='Sample', Size=50)

    # remove src fn and save df
    df = df.drop(columns=[f'ad_{which}_fn'])
    df.to_csv(os.path.join(dst_data_path, 'info.csv'))
    print(f">>> new info csv saved at {os.path.join(dst_data_path, 'info.csv')}")

if __name__ == '__main__':
    main()
