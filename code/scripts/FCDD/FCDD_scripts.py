"""
author: Antoine Spahr

date : 04.03.2021

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
from sklearn.model_selection import train_test_split

from src.dataset.datasets import RSNA_FCDD_dataset
import src.dataset.transforms as tf
from src.models.networks.FCDD_net import FCDD_CNN_VGG
from src.models.optim.FCDD import FCDD

from src.utils.python_utils import AttrDict

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train an FCDD on the RSNA dataset.
    """
    # Load config file
    cfg = AttrDict.from_json_path(config_path)

    # make outputs dir
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    os.makedirs(out_path, exist_ok=True)
    if cfg.train.validate_epoch:
         os.makedirs(os.path.join(out_path, 'valid_results/'), exist_ok=True)

    # initialize seed
    if cfg.seed != None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # initialize logger
    logger = initialize_logger(os.path.join(out_path, 'log.txt'))
    if os.path.exists(os.path.join(out_path, f'checkpoint.pt')):
        logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
    logger.info(f"Experiment : {cfg.exp_name}")

    # set device
    if cfg.device:
        cfg.device = torch.device(cfg.device)
    else:
        cfg.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Device set to {cfg.device}.")

    #--------------------------------------------------------------------
    #                           Make Datasets
    #--------------------------------------------------------------------
    # load RSNA data & keep normal only & and sample the required number
    df_rsna = pd.read_csv(os.path.join(cfg.path.data, 'slice_info.csv'), index_col=0)

    df_train = df_rsna[df_rsna.Hemorrhage == 0].sample(n=cfg.dataset.n_normal, random_state=cfg.seed)
    if cfg.dataset.n_abnormal > 0:
        df_rsna_neg = df_rsna[df_rsna.Hemorrhage == 1].sample(n=cfg.dataset.n_abnormal, random_state=cfg.seed)
        df_train = pd.concat([df_train, df_rsna_neg], axis=0)

    # df for validation
    if cfg.train.validate_epoch:
        df_rsna_remain = df_rsna[~df_rsna.index.isin(df_train.index)]
        df_valid = df_rsna_remain[df_rsna_remain.Hemorrhage == 0].sample(n=cfg.dataset.n_normal_valid, random_state=cfg.seed)
        if cfg.dataset.n_abnormal_valid > 0:
            df_rsna_neg = df_rsna_remain[df_rsna_remain.Hemorrhage == 1].sample(n=cfg.dataset.n_abnormal_valid, random_state=cfg.seed)
            df_valid = pd.concat([df_valid, df_rsna_neg], axis=0)

    # Make FCDD dataset
    train_dataset = RSNA_FCDD_dataset(df_train, cfg.path.data, artificial_anomaly=cfg.dataset.artificial_anomaly,
                                      anomaly_proba=cfg.dataset.anomaly_proba,
                                      augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.train.items()],
                                      window=(cfg.dataset.win_center, cfg.dataset.win_width), output_size=cfg.dataset.size,
                                      drawing_params=cfg.dataset.drawing_params)
    if cfg.train.validate_epoch:
        valid_dataset = RSNA_FCDD_dataset(df_valid, cfg.path.data, artificial_anomaly=cfg.dataset.artificial_anomaly_valid,
                                          anomaly_proba=cfg.dataset.anomaly_proba,
                                          augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.eval.items()],
                                          window=(cfg.dataset.win_center, cfg.dataset.win_width), output_size=cfg.dataset.size,
                                          drawing_params=cfg.dataset.drawing_params)
    else:
        valid_dataset = None

    logger.info(f"Data loaded from {cfg.path.data}.")
    logger.info(f"Train set contains {len(train_dataset)} samples.")
    if valid_dataset: logger.info(f"Valid set contains {len(valid_dataset)} samples.")
    logger.info(f"CT scans will be windowed on [{cfg.dataset.win_center-cfg.dataset.win_width/2} ; {cfg.dataset.win_center + cfg.dataset.win_width/2}]")
    logger.info(f"CT scans will be resized to {cfg.dataset.size}x{cfg.dataset.size}")
    logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
    if valid_dataset: logger.info(f"Evaluation online data transformation: \n\n {str(valid_dataset.transform)}\n")
    if cfg.dataset.artificial_anomaly:
        draw_params = [f"--> {k} : {v}" for k, v in cfg.dataset.drawing_params.items()]
        logger.info("Artificial Anomaly drawing parameters \n\t" + "\n\t".join(draw_params))

    #--------------------------------------------------------------------
    #                           Make Networks
    #--------------------------------------------------------------------
    net = FCDD_CNN_VGG(in_shape=[cfg.net.in_channels, cfg.dataset.size, cfg.dataset.size], bias=cfg.net.bias)

    #--------------------------------------------------------------------
    #                          Make FCDD model
    #--------------------------------------------------------------------
    cfg.train.model_param.lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.train.model_param.lr_scheduler) # convert scheduler name to scheduler class object
    model = FCDD(net, print_progress=cfg.print_progress,
                 device=cfg.device,  **cfg.train.model_param)
    train_params = [f"--> {k} : {v}" for k, v in cfg.train.model_param.items()]
    logger.info("FCDD Training Parameters \n\t" + "\n\t".join(train_params))

    # load models if provided
    if cfg.train.model_path_to_load:
        model.load_model(cfg.train.model_path_to_load, map_location=cfg.device)
        logger.info(f"FCDD Model loaded from {cfg.train.model_path_to_load}")

    #--------------------------------------------------------------------
    #                          Train FCDD model
    #--------------------------------------------------------------------
    if cfg.train.model_param.n_epoch > 0:
        model.train(train_dataset, checkpoint_path=os.path.join(out_path, 'Checkpoint.pt'),
                    valid_dataset=valid_dataset)

    #--------------------------------------------------------------------
    #               Generate and save few Heatmap with FCDD model
    #--------------------------------------------------------------------
    if cfg.train.validate_epoch:
        if len(valid_dataset) > 100:
            valid_subset = torch.utils.data.random_split(valid_dataset, [100, len(valid_dataset)-100],
                                                        generator=torch.Generator().manual_seed(cfg.seed))[0]
        else:
            valid_subset = valid_dataset
        model.localize_anomalies(valid_subset, save_path=os.path.join(out_path, 'valid_results/'),
                                 **cfg.train.heatmap_param)

    #--------------------------------------------------------------------
    #                   Save outputs, models and config
    #--------------------------------------------------------------------
    # save models
    model.save_model(export_fn=os.path.join(out_path, 'FCDD.pt'))
    logger.info("FCDD model saved at " + os.path.join(out_path, 'FCDD.pt'))
    # save outputs
    model.save_outputs(export_fn=os.path.join(out_path, 'outputs.json'))
    logger.info("Outputs file saved at " + os.path.join(out_path, 'outputs.json'))
    # save config file
    cfg.device = str(cfg.device) # set device as string to be JSON serializable
    cfg.train.model_param.lr_scheduler = str(cfg.train.model_param.lr_scheduler)
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info("Config file saved at " + os.path.join(out_path, 'config.json'))

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

if __name__ == '__main__':
    main()
#
