"""
author: Antoine Spahr

date : 09.11.2020

----------

TO DO :
"""
import os
import sys
sys.path.append('../')
import glob
import click
import json
import torch
import nibabel as nib

from src.models.networks.UNet import UNet
from src.models.optim.UNet2D import UNet2D

@click.command()
@click.argument("sample_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path(exists=False))
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--device", type=str, default=None, help='device to work on. Default: None. automatic')
def main(sample_path, save_path, model_path, config_path, device):
    """
    Segment the volume(s) at sample_path using the UNet2D model at model_path and the meta-information in the config file.
    The results is/are saved at save_path. Sample path can be a single nifti volume (.nii or .nii.gz) or a folder containing
    several nifti volumes. If a single file is processed, save_path must be a nifti file (.nii or .nii.gz).
    The config_path must point to a json file with UNet architecture parameters under the key 'net'. It must also contains
    the windowing and and size parameters under the key 'data'. The batch_size and number of worker are specified under
    the key 'train'.
    """
    ############################# initiate model from path and config file #############################
    # load config
    with open(config_path, 'r') as fn:
        cfg = json.load(fn)

    # get the device
    if device is None:
        if torch.cuda.is_available():
            free_mem, device_idx = 0.0, 0
            for d in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(d).total_memory - torch.cuda.memory_reserved(d)
                if mem > free_mem:
                    device_idx = d
                    free_mem = mem
            device = torch.device(f'cuda:{device_idx}')
        else:
            device = torch.device('cpu')
    print(f">>> Device : {device}")

    # define net
    unet = UNet(depth=cfg['net']['depth'], use_3D=cfg['net']['3D'], bilinear=cfg['net']['bilinear'],
                in_channels=cfg['net']['in_channels'], out_channels=cfg['net']['out_channels'], top_filter=cfg['net']['top_filter'],
                midchannels_factor=cfg['net']['midchannels_factor'], p_dropout=cfg['net']['p_dropout'])

    # define model
    unet2D = UNet2D(unet, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'], device=device,
                    print_progress=cfg['print_progress'])
    # initialize model with trained weights
    unet2D.load_model(model_path, map_location=device)
    print(f'>>> Model successfully loaded from {model_path}')

    ############# check if sample path is folder or file and if save path is folder or path ############
    if os.path.isfile(sample_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if sample_path.endswith(('.nii', '.nii.gz')) and save_path.endswith(('.nii', '.nii.gz')):
            sample_fn_list = [sample_path]
            save_fn_list = [save_path]
        else:
            raise ValueError(f"Invalid file extension. Both sample and save path must be .nii or .nii.gz files.")
    elif os.path.isdir(sample_path):
        os.makedirs(save_path, exist_ok=True)
        sample_fn_list = glob.glob(os.path.join(sample_path, '*.nii')) + glob.glob(os.path.join(sample_path, '*.nii.gz'))
        # generate paired save_fn
        save_fn_list = [os.path.join(save_path, os.path.splitext(os.path.basename(f))[0] + '.nii.gz') for f in sample_fn_list]
    else:
        raise ValueError(f"Sample path and save path must be either both file path or both directory path.")

    ############################### call segement_volume for all samples ###############################
    for i, (vol_path, save_path) in enumerate(zip(sample_fn_list, save_fn_list)):
        print(f">>> {i+1:03}/{len(sample_fn_list):03} Segementation of volume {os.path.basename(vol_path)}")
        # load vol
        vol = nib.load(vol_path)
        # segement
        unet2D.segement_volume(vol, save_fn=save_path, window=(cfg['data']['win_center'], cfg['data']['win_width']),
                               input_size=(cfg['data']['size'], cfg['data']['size']), return_pred=False)
        print(f">>> Volume segmentation saved at {save_path}")

if __name__ == '__main__':
    main()
