"""
author: Antoine Spahr

date : 14.12.2020

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
from prettytable import PrettyTable

from sklearn.model_selection import train_test_split

from src.utils.python_utils import AttrDict
from src.models.optim.Classifier import BinaryClassifier
import src.models.networks.ResNet as resnet
from src.dataset.datasets import RSNA_dataset
import src.dataset.transforms as tf

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    ResNet binary classification with the RSNA dataset.
    """
    # load the config file
    cfg = AttrDict.from_json_path(config_path)

    # Make Outputs directories
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    os.makedirs(out_path, exist_ok=True)

    # Initialize random seed
    if cfg.seed != -1:
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

    # Set number of thread
    if cfg.n_thread > 0: torch.set_num_threads(cfg.n_thread)
    # set device, if None use the first one
    if cfg.device:
        cfg.device = torch.device(cfg.device)
    else:
        cfg.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Device set to {cfg.device}. {torch.cuda.device_count()} GPU available, "
                f"{len(cfg.multi_gpu_id) if cfg.multi_gpu_id else 1} used.")

    # Load RSNA data csv
    df_rsna = pd.read_csv(os.path.join(cfg.path.data, 'slice_info.csv'), index_col=0)

    # Keep only fractions sample
    if cfg.dataset.n_data_0 >= 0:
        df_rsna_noICH = df_rsna[df_rsna.Hemorrhage == 0].sample(n=cfg.dataset.n_data_0, random_state=cfg.seed)
    else:
        df_rsna_noICH = df_rsna[df_rsna.Hemorrhage == 0]
    if cfg.dataset.n_data_1 >= 0:
        df_rsna_ICH = df_rsna[df_rsna.Hemorrhage == 1].sample(n=cfg.dataset.n_data_1, random_state=cfg.seed)
    else:
        df_rsna_ICH = df_rsna[df_rsna.Hemorrhage == 1]
    df_rsna = pd.concat([df_rsna_ICH, df_rsna_noICH], axis=0)

    # Split data to keep few for evaluation in a strafied way
    train_df, test_df = train_test_split(df_rsna, test_size=cfg.dataset.frac_eval, stratify=df_rsna.Hemorrhage, random_state=cfg.seed)
    logger.info('\n' + str(get_split_summary_table(df_rsna, train_df, test_df)))

    # Make dataset : Train --> BinaryClassification, Test --> BinaryClassification
    train_RSNA_dataset = RSNA_dataset(train_df, cfg.path.data,
                                 augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.train.items()],
                                 window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size,
                                 mode='binary_classification')
    test_RSNA_dataset = RSNA_dataset(test_df, cfg.path.data,
                                 augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.dataset.augmentation.eval.items()],
                                 window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size,
                                 mode='binary_classification')

    logger.info(f"Data will be loaded from {cfg.path.data}.")
    logger.info(f"CT scans will be windowed on [{cfg.data.win_center-cfg.data.win_width/2} ; {cfg.data.win_center + cfg.data.win_width/2}]")
    logger.info(f"CT scans will be resized to {cfg.data.size}x{cfg.data.size}")
    logger.info(f"Training online data transformation: \n\n {str(train_RSNA_dataset.transform)}\n")
    logger.info(f"Evaluation online data transformation: \n\n {str(test_RSNA_dataset.transform)}\n")

    # Make Resnet Architecture
    resnet_network = getattr(resnet, cfg.net.resnet)(num_classes=cfg.net.num_classes, input_channels=cfg.net.input_channels)
    logger.info(f"Using a {cfg.net.resnet} architecture.")
    if cfg.multi_gpu_id is not None and len(cfg.multi_gpu_id) > 1: # set network for multi-GPU
        resnet_network = torch.nn.DataParallel(resnet_network, device_ids=cfg.multi_gpu_id)
        logger.info("Enabling the resnet for multi-GPU computation.")
    resnet_network = resnet_network.to(cfg.device)
    logger.info(f"The {cfg.net.resnet} has {sum(p.numel() for p in resnet_network.parameters())} parameters.")

    # Make model
    cfg.train.model_param.lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.train.model_param.lr_scheduler) # convert scheduler name to scheduler class object
    cfg.train.model_param.loss_fn = getattr(torch.nn, cfg.train.model_param.loss_fn) # convert loss_fn name to nn.Module class object
    w_ICH = train_df.Hemorrhage.sum() / len(train_df) # define CE weighting from train dataset
    cfg.train.model_param.loss_fn_kwargs['weight'] = torch.tensor([1 - w_ICH, w_ICH], device=cfg.device).float() # add weighting to CE kwargs

    classifier = BinaryClassifier(resnet_network, device=cfg.device, print_progress=cfg.print_progress, **cfg.train.model_param)

    train_params = [f"--> {k} : {v}" for k, v in cfg.train.model_param.items()]
    logger.info("Classifer Training Parameters \n\t" + "\n\t".join(train_params))

    # Load weights if specified
    if cfg.train.model_path_to_load:
        model_path = cfg.train.model_path_to_load
        classifier.load_model(model_path, map_location=cfg.device)
        logger.info(f"Classifer Model succesfully loaded from {cfg.train.model_path_to_load}")

    # Train
    if cfg.train.model_param.n_epoch > 0:
        classifier.train(train_RSNA_dataset, valid_dataset=test_RSNA_dataset,
                        checkpoint_path=os.path.join(out_path, f'checkpoint.pt'))

    # Evaluate
    auc = classifier.evaluate(test_RSNA_dataset, save_tsne=False, return_auc=True)
    logger.info(f"Classifier Test AUC : {auc:.2%}")

    # save model, outputs
    classifier.save_model(os.path.join(out_path, 'resnet.pt'))
    logger.info(f"{cfg.net.resnet} saved at " + os.path.join(out_path, 'resnet.pt'))
    classifier.save_outputs(os.path.join(out_path, 'outputs.json'))
    logger.info("Classifier outputs saved at " + os.path.join(out_path, 'outputs.json'))
    test_df.reset_index(drop=True).to_csv(os.path.join(out_path, 'eval_data_info.csv'))
    logger.info("Evaluation data info saved at " + os.path.join(out_path, 'eval_data_info.csv'))

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

def get_split_summary_table(all_df, train_df, test_df):
    """
    return a table summarising the data split.
    """
    table = PrettyTable()
    table.field_names = ['set', 'N total', 'N non-ICH', 'N ICH', 'frac non-ICH', 'frac ICH']
    for df, name in zip([all_df, train_df, test_df],['All', 'Train', 'Test']):
        table.add_row([name, len(df), len(df[df.Hemorrhage == 0]), len(df[df.Hemorrhage == 1]),
                   f'{len(df[df.Hemorrhage == 0])/len(df):.3%}', f'{len(df[df.Hemorrhage == 1])/len(df):.3%}'])
    return table

if __name__ == '__main__':
    main()
#
