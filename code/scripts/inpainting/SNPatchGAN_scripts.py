"""
author: Antoine Spahr

date : 17.11.2020

----------

TO DO :
"""
import sys
sys.path.append('../../')
import click
import os
import logging
import json
import random
import torch
import torch.cuda
import numpy as np
import pandas as pd

from src.dataset.datasets import RSNA_Inpaint_dataset, ImgMaskDataset
import src.dataset.transforms as tf
from src.models.networks.InpaintingNetwork import GatedGenerator, PatchDiscriminator
from src.models.optim.SNPatchGAN import SNPatchGAN

from src.utils.python_utils import AttrDict

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train an Inpainting generator with gated convolution through a SN-PatchGAN training scheme. The generator is trained
    to inpaint non-ICH CT scans from the RSNA dataset.
    """
    # Load config file
    cfg = AttrDict.from_json_path(config_path)

    # make outputs dir
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    os.makedirs(out_path, exist_ok=True)
    if cfg.train.validate_epoch:
         os.makedirs(os.path.join(out_path, '/valid_results/'), exist_ok=True)

    # initialize seed
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # set device
    if cfg.device:
        cfg.device = torch.device(cfg.device)
    else:
        cfg.device = get_available_device()

    # set n_thread
    if cfg.n_thread > 0: torch.set_num_threads(cfg.n_thread)

    # initialize logger
    logger = initialize_logger(os.path.join(out_path, 'log.txt'))
    if os.path.exists(os.path.join(out_path, f'checkpoint.pt')):
        logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
    logger.info(f"Experiment : {cfg.exp_name}")

    #--------------------------------------------------------------------
    #                           Make Datasets
    #--------------------------------------------------------------------
    # load RSNA data & keep normal only & and sample the required number
    df_rsna = pd.read_csv(os.path.join(cfg.path.data, 'slice_info.csv'), index_col=0)
    df_rsna_pos = df_rsna[df_rsna.Hemorrhage == 0]
    if cfg.dataset.n_sample >= 0:
        df_rsna_pos = df_rsna_pos.sample(n=cfg.dataset.n_sample, random_state=cfg.seed)
    # make dataset
    train_dataset = RSNA_Inpaint_dataset(df_rsna_pos, cfg.path.data,
                        augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.train.items()],
                        window=(cfg.dataset.win_center, cfg.dataset.win_width), output_size=cfg.dataset.size,
                        **cfg.dataset.mask)

    # load small valid subset and make dataset
    if cfg.train.validate_epoch:
        df_valid = pd.read_csv(os.path.join(cfg.path.data_valid, 'info.csv'), index_col=0)
        valid_dataset = ImgMaskDataset(df_valid, cfg.path.data_valid,
                            augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.eval.items()],
                            window=(cfg.dataset.win_center, cfg.dataset.win_width), output_size=cfg.dataset.size)
    else:
        valid_dataset = None

    logger.info(f"Train Data will be loaded from {cfg.path.data}.")
    logger.info(f"Train contains {len(train_dataset)} samples.")
    logger.info(f"Valid Data will be loaded from {cfg.path.data_valid}.")
    if valid_dataset: logger.info(f"Valid contains {len(valid_dataset)} samples.")
    logger.info(f"CT scans will be windowed on [{cfg.dataset.win_center-cfg.dataset.win_width/2} ; {cfg.dataset.win_center + cfg.dataset.win_width/2}]")
    logger.info(f"CT scans will be resized to {cfg.dataset.size}x{cfg.dataset.size}")
    logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
    if valid_dataset: logger.info(f"Evaluation online data transformation: \n\n {str(valid_dataset.transform)}\n")
    mask_params = [f"--> {k} : {v}" for k, v in cfg.dataset.mask.items()]
    logger.info("Train inpainting masks generated with \n\t" + "\n\t".join(mask_params))

    #--------------------------------------------------------------------
    #                           Make Networks
    #--------------------------------------------------------------------
    cfg.net.gen.context_attention_kwargs['device'] = cfg.device # add device to kwargs of contextual attention module
    generator_net = GatedGenerator(**cfg.net.gen)
    discriminator_net = PatchDiscriminator(**cfg.net.dis)

    gen_params = [f"--> {k} : {v}" for k, v in cfg.net.gen.items()]
    logger.info("Gated Generator Parameters \n\t" + "\n\t".join(gen_params))
    dis_params = [f"--> {k} : {v}" for k, v in cfg.net.dis.items()]
    logger.info("Gated Generator Parameters \n\t" + "\n\t".join(dis_params))

    #--------------------------------------------------------------------
    #                      Make Inpainting GAN model
    #--------------------------------------------------------------------
    gan_model = SNPatchGAN(generator_net, discriminator_net, **cfg.train.model_param)
    train_params = [f"--> {k} : {v}" for k, v in cfg.train.model_param.items()]
    logger.info("GAN Training Parameters \n\t" + "\n\t".join(train_params))

    # load models if provided
    if cfg.train.model_path_to_load.gen:
        gan_model.load_Generator(cfg.train.model_path_to_load.gen, map_location=cfg.device)
    if cfg.train.model_path_to_load.dis:
        gan_model.load_Discriminator(cfg.train.model_path_to_load.dis, map_location=cfg.device)

    #--------------------------------------------------------------------
    #                       Train SN-PatchGAN model
    #--------------------------------------------------------------------
    if cfg.train.model_param.n_epoch > 0:
        gan_model.train(train_dataset, checkpoint_path=os.path.join(out_path, 'Checkpoint.pt'),
                        valid_dataset=valid_dataset, valid_path=os.path.join(out_path, '/valid_results/'),
                        save_freq=cfg.train.valid_save_freq)

    #--------------------------------------------------------------------
    #                   Save outputs, models and config
    #--------------------------------------------------------------------
    # save models
    gan_model.save_models(export_fn=(os.path.join(out_path, 'generator.pt'),
                                     os.path.join(out_path, 'discriminator.pt')), which='both')
    logger.info("Generator model saved at " + os.path.join(out_path, 'generator.pt'))
    logger.info("Discriminator model saved at " + os.path.join(out_path, 'discriminator.pt'))
    # save outputs
    gan_model.save_outputs(export_fn=os.path.join(out_path, 'outputs.json'))
    logger.info("Outputs file saved at " + os.path.join(out_path, 'outputs.json'))
    # save config file
    cfg.device = str(cfg.device) # set device as string to be JSON serializable
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info("Config file saved at " + os.path.join(out_path, 'config.json'))

    # delete any checkpoints
    if os.path.exists(os.path.join(out_path, f'checkpoint.pt')):
        os.remove(os.path.join(out_path, f'checkpoint.pt'))
        logger.info('Checkpoint deleted.')

def initialize_logger(logger_fn):
    """
    Initialize a logger with given file name. It will start a new logger.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    try:
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])
    except IndexError:
        pass
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logger_fn)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    return logger

def get_available_device():
    """

    """
    if torch.cuda.is_available():
        free_mem, device_idx = 0.0, 0
        for d in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(d).total_memory - torch.cuda.memory_allocated(d)
            if mem > free_mem:
                device_idx = d
                free_mem = mem
        return torch.device(f'cuda:{device_idx}')
    else:
        return torch.device('cpu')

if __name__ == '__main__':
    main()
