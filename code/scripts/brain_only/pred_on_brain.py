"""
author: Antoine Spahr

date : 05.11.2020

----------

TO DO :
"""
import os
import sys
sys.path.append('../../')
import click

from src.postprocessing.update_pred import update_Kfold_folder, update_anomaly_pred_folder

@click.command()
@click.argument("exp_folder", type=click.Path(exists=True))
@click.argument("brain_mask_folder", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--save_path", type=click.Path(exists=False), default='None', help="Where to save the adjusted experiment. Default: exp_folder_BrainOnly")
@click.option("--brain_nii/--brain_bmp", default=True, help="Whether the brain mask are provided as nifti or bmp. Defualt : as nifti.")
@click.option("--rot/--no_rot", default=True, help="Whether brain mask are rotated 90° counterclockwise. Default True.")
@click.option("--kfold/--anomaly", default=True, help="Whether to process a KFold experiment or a Single prediction. Default : Kfold.")
def main(exp_folder, brain_mask_folder, data_path, save_path, brain_nii, rot, kfold):
    """
    Post-Process a Cross-Validated or Single prediction ICH segmentation experiment by removing prediction outside the brain.
    exp_folder --> path to the experiment to convert
    brain_mask_folder --> path to the brain_mask for each volumes (as folder of .bmp or as Nifti).
    data_path --> path to the ICH dataset.
    """
    if save_path == 'None':
        save_path = '/'.join(exp_folder.split('/')[:-1]) + '_BrainOnly/'

    if kfold:
        update_Kfold_folder(exp_folder, save_path, brain_mask_folder, data_path, rot=rot, print_progress=True, verbose=True)
    else:
        update_anomaly_pred_folder(os.path.join(exp_folder, 'pred'), save_path, brain_mask_folder, brain_nii, data_path,
                                   rot=rot, print_progress=True)

if __name__ == '__main__':
    main()
