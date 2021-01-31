"""
author: Antoine Spahr

date : 07.01.2021

----------

TO DO :
    -
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

from  sklearn.model_selection import StratifiedKFold

#from src.utils.Config import Config
from src.utils.python_utils import AttrDict
from src.models.optim.UNet2D import UNet2D
from src.dataset.datasets import public_SegICH_AttentionDataset2D
from src.models.optim.LossFunctions import BinaryDiceLoss
import src.dataset.transforms as tf
import src.models.optim.LossFunctions
from src.models.networks.GatedUNet import UNet
from src.postprocessing.analyse_exp import analyse_supervised_exp

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train and evaluate a 2D UNet on the public ICH dataset with the anomaly attention map using the parameters on the
    JSON at the config_path. The evaluation is performed by k-fold cross-validation.
    """
    # load config file
    cfg = AttrDict.from_json_path(config_path)

    # Make Output directories
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    os.makedirs(out_path, exist_ok=True)
    for k in range(cfg.split.n_fold):
        os.makedirs(os.path.join(out_path, f'Fold_{k+1}/pred/'), exist_ok=True)

    # Initialize random seed to given seed
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # Load data csv
    data_info_df = pd.read_csv(os.path.join(cfg.path.data, 'info.csv'), index_col=0)
    patient_df = pd.read_csv(os.path.join(cfg.path.data, 'patient_info.csv'), index_col=0)

    # Generate Cross-Val indices at the patient level
    skf = StratifiedKFold(n_splits=cfg.split.n_fold, shuffle=cfg.split.shuffle, random_state=cfg.seed)
    # iterate over folds and ensure that there are the same amount of ICH positive patient per fold --> Stratiffied CrossVal
    for k, (train_idx, test_idx) in enumerate(skf.split(patient_df.PatientNumber, patient_df.Hemorrhage)):
        # if fold results not already there
        if not os.path.exists(os.path.join(out_path, f'Fold_{k+1}/outputs.json')):
            # initialize logger
            logger = initialize_logger(os.path.join(out_path, 'log.txt'))
            if os.path.exists(os.path.join(out_path, f'Fold_{k+1}/checkpoint.pt')):
                logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
            logger.info(f"Experiment : {cfg.exp_name}")
            logger.info(f"Cross-Validation fold {k+1:02}/{cfg.split.n_fold:02}")

            # check if GPU available
            if cfg.device is not None:
                cfg.device = torch.device(cfg.device)
            else:
                cfg.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            logger.info(f"Device : {cfg.device}")

            # extract train and test DataFrames + print summary (n samples positive and negatives)
            train_df = data_info_df[data_info_df.id.isin(patient_df.loc[train_idx,'PatientNumber'].values)]
            test_df = data_info_df[data_info_df.id.isin(patient_df.loc[test_idx,'PatientNumber'].values)]
            # sample the dataframe to have more or less normal slices
            n_remove = int(max(0, len(train_df[train_df.Hemorrhage == 0]) - cfg.dataset.frac_negative * len(train_df[train_df.Hemorrhage == 1])))
            df_remove = train_df[train_df.Hemorrhage == 0].sample(n=n_remove, random_state=cfg.seed)
            train_df = train_df[~train_df.index.isin(df_remove.index)]
            logger.info('\n' + str(get_split_summary_table(data_info_df, train_df, test_df)))

            # Make Dataset + print online augmentation summary
            train_dataset = public_SegICH_AttentionDataset2D(train_df, cfg.path.data,
                                                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.data.augmentation.train.items()],
                                                    window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size)
            test_dataset = public_SegICH_AttentionDataset2D(test_df, cfg.path.data,
                                                   augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.data.augmentation.eval.items()],
                                                   window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size)
            logger.info(f"Data will be loaded from {cfg.path.data}.")
            logger.info(f"CT scans will be windowed on [{cfg.data.win_center-cfg.data.win_width/2} ; {cfg.data.win_center + cfg.data.win_width/2}]")
            logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
            logger.info(f"Evaluation online data transformation: \n\n {str(test_dataset.transform)}\n")

            # Make architecture (and print summmary ??)
            unet_arch = UNet(**cfg.net)
            unet_arch.to(cfg.device)
            net_params = [f"--> {k} : {v}" for k, v in cfg.net.items()]
            logger.info("U-Net2D architecture \n\t" + "\n\t".join(net_params))
            logger.info(f"The U-Net2D has {sum(p.numel() for p in unet_arch.parameters())} parameters.")

            # Make model
            cfg_train = AttrDict(cfg.train.params)
            cfg_train.lr_scheduler = getattr(torch.optim.lr_scheduler, cfg_train.lr_scheduler)
            cfg_train.loss_fn = getattr(src.models.optim.LossFunctions, cfg_train.loss_fn)
            unet2D = UNet2D(unet_arch, device=cfg.device, print_progress=cfg.print_progress, **cfg_train)
            # print Training hyper-parameters
            train_params = [f"--> {k} : {v}" for k, v in cfg_train.items()]
            logger.info("U-Net2D Training Parameters \n\t" + "\n\t".join(train_params))

            # Load model if required
            if cfg.train.model_path_to_load:
                if isinstance(cfg.train.model_path_to_load, str):
                    model_path = cfg.train.model_path_to_load
                    unet2D.load_model(model_path, map_location=cfg.device)
                elif isinstance(cfg.train.model_path_to_load, list):
                    model_path = cfg.train.model_path_to_load[k]
                    unet2D.load_model(model_path, map_location=cfg.device)
                else:
                    raise ValueError(f'Model path to load type not understood.')
                logger.info(f"2D U-Net model loaded from {model_path}")

            # Train model
            eval_dataset = test_dataset if cfg.train.validate_epoch else None
            unet2D.train(train_dataset, valid_dataset=eval_dataset, checkpoint_path=os.path.join(out_path, f'Fold_{k+1}/checkpoint.pt'))

            # Evaluate model
            unet2D.evaluate(test_dataset, save_path=os.path.join(out_path, f'Fold_{k+1}/pred/'))

            # Save models & outputs
            unet2D.save_model(os.path.join(out_path, f'Fold_{k+1}/trained_unet.pt'))
            logger.info("Trained U-Net saved at " + os.path.join(out_path, f'Fold_{k+1}/trained_unet.pt'))
            unet2D.save_outputs(os.path.join(out_path, f'Fold_{k+1}/outputs.json'))
            logger.info("Trained statistics saved at " + os.path.join(out_path, f'Fold_{k+1}/outputs.json'))

            # delete checkpoint if exists
            if os.path.exists(os.path.join(out_path, f'Fold_{k+1}/checkpoint.pt')):
                os.remove(os.path.join(out_path, f'Fold_{k+1}/checkpoint.pt'))
                logger.info('Checkpoint deleted.')

    # save mean +/- 1.96 std Dice in .txt file
    scores_list = []
    for k in range(cfg.split.n_fold):
        with open(os.path.join(out_path, f'Fold_{k+1}/outputs.json'), 'r') as f:
            out = json.load(f)
        scores_list.append([out['eval']['dice']['all'], out['eval']['dice']['positive']])
    means = np.array(scores_list).mean(axis=0)
    CI95 = 1.96*np.array(scores_list).std(axis=0)
    with open(os.path.join(out_path, 'average_scores.txt'), 'w') as f:
        f.write(f'Dice = {means[0]} +/- {CI95[0]}\n')
        f.write(f'Dice (Positive) = {means[1]} +/- {CI95[1]}\n')
    logger.info('Average Scores saved at ' + os.path.join(out_path, 'average_scores.txt'))

    # generate dataframe of all prediction
    df_list = [pd.read_csv(os.path.join(out_path, f'Fold_{i+1}/pred/volume_prediction_scores.csv')) for i in range(cfg.split.n_fold)]
    all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    all_df.to_csv(os.path.join(out_path, 'all_volume_prediction.csv'))
    logger.info('CSV of all volumes prediction saved at ' + os.path.join(out_path, 'all_volume_prediction.csv'))

    # Save config file
    cfg.device = str(cfg.device)
    #cfg.train.params.lr_scheduler = str(cfg.train.params.lr_scheduler)
    #cfg.train.params.loss_fn = str(cfg.train.params.loss_fn)
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info("Config file saved at " + os.path.join(out_path, 'config.json'))

    # Analyse results
    analyse_supervised_exp(out_path, cfg.path.data, cfg.split.n_fold, save_fn=os.path.join(out_path, 'results_overview.pdf'))
    logger.info('Results overview figure saved at ' + os.path.join(out_path, 'results_overview.pdf'))

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
