"""
author: Antoine Spahr

date : 01.10.2020

----------

TO DO :
- add a summary print of architecture
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

from src.utils.Config import Config
from src.models.optim.UNet2D import UNet2D
from src.dataset.datasets import public_SegICH_Dataset2D
from src.models.optim.LossFunctions import BinaryDiceLoss
import src.dataset.transforms as tf
import src.models.optim.LossFunctions
from src.models.networks.UNet import UNet
from src.postprocessing.analyse_exp import analyse_supervised_exp

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train and evaluate a 2D UNet on the public ICH dataset using the parameters sepcified on the JSON at the
    config_path. The evaluation is performed by k-fold cross-validation.
    """
    # load config file
    cfg = Config(settings=None)
    cfg.load_config(config_path)

    # Make Output directories
    out_path = os.path.join(cfg.settings['path']['OUTPUT'], cfg.settings['exp_name'])# + '/'
    os.makedirs(out_path, exist_ok=True)
    for k in range(cfg.settings['split']['n_fold']):
        os.makedirs(os.path.join(out_path, f'Fold_{k+1}/pred/'), exist_ok=True)

    # Initialize random seed to given seed
    seed = cfg.settings['seed']
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # Load data csv
    data_info_df = pd.read_csv(os.path.join(cfg.settings['path']['DATA'], 'ct_info.csv'))
    data_info_df = data_info_df.drop(data_info_df.columns[0], axis=1)
    patient_df = pd.read_csv(os.path.join(cfg.settings['path']['DATA'], 'patient_info.csv'))
    patient_df = patient_df.drop(patient_df.columns[0], axis=1)

    # Generate Cross-Val indices at the patient level
    skf = StratifiedKFold(n_splits=cfg.settings['split']['n_fold'],
                          shuffle=cfg.settings['split']['shuffle'],
                          random_state=seed)
    # iterate over folds and ensure that there are the same amount of ICH positive patient per fold --> Stratiffied CrossVal
    #scores_list = [] # placeholder for mean test dice of each fold
    for k, (train_idx, test_idx) in enumerate(skf.split(patient_df.PatientNumber, patient_df.Hemorrhage)):
        # if fold results not already there
        if not os.path.exists(os.path.join(out_path, f'Fold_{k+1}/outputs.json')):
            # initialize logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            try:
                logger.handlers[1].stream.close()
                logger.removeHandler(logger.handlers[1])
            except IndexError:
                pass
            logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(out_path, f'Fold_{k+1}/log.txt'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logger.addHandler(file_handler)

            if os.path.exists(os.path.join(out_path, f'Fold_{k+1}/checkpoint.pt')):
                logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)

            logger.info(f"Experiment : {cfg.settings['exp_name']}")
            logger.info(f"Cross-Validation fold {k+1:02}/{cfg.settings['split']['n_fold']:02}")

            # initialize nbr of thread
            if cfg.settings['n_thread'] > 0:
                torch.set_num_threads(cfg.settings['n_thread'])
            logger.info(f"Number of thread : {cfg.settings['n_thread']}")
            # check if GPU available
            cfg.settings['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            logger.info(f"Device : {cfg.settings['device']}")

            # extract train and test DataFrames + print summary (n samples positive and negatives)
            train_df = data_info_df[data_info_df.PatientNumber.isin(patient_df.loc[train_idx,'PatientNumber'].values)]
            test_df = data_info_df[data_info_df.PatientNumber.isin(patient_df.loc[test_idx,'PatientNumber'].values)]
            # sample the dataframe to have more or less normal slices
            n_remove = int(max(0, len(train_df[train_df.Hemorrhage == 0]) - cfg.settings['dataset']['frac_negative'] * len(train_df[train_df.Hemorrhage == 1])))
            df_remove = train_df[train_df.Hemorrhage == 0].sample(n=n_remove, random_state=seed)
            train_df = train_df[~train_df.index.isin(df_remove.index)]
            logger.info('\n' + str(get_split_summary_table(data_info_df, train_df, test_df)))

            # Make Dataset + print online augmentation summary
            train_dataset = public_SegICH_Dataset2D(train_df, cfg.settings['path']['DATA'],
                                                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.settings['data']['augmentation']['train'].items()],
                                                    window=(cfg.settings['data']['win_center'], cfg.settings['data']['win_width']),
                                                    output_size=cfg.settings['data']['size'])
            test_dataset = public_SegICH_Dataset2D(test_df, cfg.settings['path']['DATA'],
                                                   augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.settings['data']['augmentation']['eval'].items()],
                                                   window=(cfg.settings['data']['win_center'], cfg.settings['data']['win_width']),
                                                   output_size=cfg.settings['data']['size'])
            logger.info(f"Data will be loaded from {cfg.settings['path']['DATA']}.")
            logger.info(f"CT scans will be windowed on [{cfg.settings['data']['win_center']-cfg.settings['data']['win_width']/2} ; {cfg.settings['data']['win_center'] + cfg.settings['data']['win_width']/2}]")
            logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
            logger.info(f"Evaluation online data transformation: \n\n {str(test_dataset.transform)}\n")

            # Make architecture (and print summmary ??)
            unet_arch = UNet(depth=cfg.settings['net']['depth'], top_filter=cfg.settings['net']['top_filter'],
                             use_3D=cfg.settings['net']['3D'], in_channels=cfg.settings['net']['in_channels'],
                             out_channels=cfg.settings['net']['out_channels'], bilinear=cfg.settings['net']['bilinear'])
            unet_arch.to(cfg.settings['device'])
            logger.info(f"U-Net2D initialized with a depth of {cfg.settings['net']['depth']}"
                        f" and a number of initial filter of {cfg.settings['net']['top_filter']},")
            logger.info(f"Reconstruction performed with {'Upsample + Conv' if cfg.settings['net']['bilinear'] else 'ConvTranspose'}.")
            logger.info(f"U-Net2D takes {cfg.settings['net']['in_channels']} as input channels and {cfg.settings['net']['out_channels']} as output channels.")
            logger.info(f"The U-Net2D has {sum(p.numel() for p in unet_arch.parameters())} parameters.")

            # Make model
            unet2D = UNet2D(unet_arch, n_epoch=cfg.settings['train']['n_epoch'], batch_size=cfg.settings['train']['batch_size'],
                            lr=cfg.settings['train']['lr'], lr_scheduler=getattr(torch.optim.lr_scheduler, cfg.settings['train']['lr_scheduler']),
                            lr_scheduler_kwargs=cfg.settings['train']['lr_scheduler_kwargs'],
                            loss_fn=getattr(src.models.optim.LossFunctions, cfg.settings['train']['loss_fn']),
                            loss_fn_kwargs=cfg.settings['train']['loss_fn_kwargs'], weight_decay=cfg.settings['train']['weight_decay'],
                            num_workers=cfg.settings['train']['num_workers'], device=cfg.settings['device'],
                            print_progress=cfg.settings['print_progress'])

            # Load model if required
            if cfg.settings['train']['model_path_to_load']:
                if isinstance(cfg.settings['train']['model_path_to_load'], str):
                    model_path = cfg.settings['train']['model_path_to_load']
                    unet2D.load_model(model_path, map_location=cfg.settings['device'])
                elif isinstance(cfg.settings['train']['model_path_to_load'], list):
                    model_path = cfg.settings['train']['model_path_to_load'][k]
                    unet2D.load_model(model_path, map_location=cfg.settings['device'])
                else:
                    raise ValueError(f'Model path to load type not understood.')
                logger.info(f"2D U-Net model loaded from {model_path}")

            # print Training hyper-parameters
            train_params = []
            for key, value in cfg.settings['train'].items():
                train_params.append(f"--> {key} : {value}")
            logger.info('Training settings:\n\t' + '\n\t'.join(train_params))

            # Train model
            eval_dataset = test_dataset if cfg.settings['train']['validate_epoch'] else None
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
    for k in range(cfg.settings['split']['n_fold']):
        with open(os.path.join(out_path, f'Fold_{k+1}/outputs.json'), 'r') as f:
            out = json.load(f)
        scores_list.append([out['eval']['dice']['all'], out['eval']['dice']['ICH']])
    means = np.array(scores_list).mean(axis=0)
    CI95 = 1.96*np.array(scores_list).std(axis=0)
    with open(os.path.join(out_path, 'average_scores.txt'), 'w') as f:
        f.write(f'Dice = {means[0]} +/- {CI95[0]}\n')
        f.write(f'Dice (ICH) = {means[1]} +/- {CI95[1]}\n')
    logger.info('Average Scores saved at ' + os.path.join(out_path, 'average_scores.txt'))

    # generate dataframe of all prediction
    df_list = [pd.read_csv(os.path.join(out_path, f'Fold_{i+1}/pred/volume_prediction_scores.csv')) for i in range(cfg.settings['split']['n_fold'])]
    all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    all_df.to_csv(os.path.join(out_path, 'all_volume_prediction.csv'))
    logger.info('CSV of all volumes prediction saved at ' + os.path.join(out_path, 'all_volume_prediction.csv'))

    # Save config file
    cfg.settings['device'] = str(cfg.settings['device'])
    cfg.save_config(os.path.join(out_path, 'config.json'))
    logger.info("Config file saved at " + os.path.join(out_path, 'config.json'))

    # Analyse results
    analyse_supervised_exp(out_path, cfg.settings['path']['DATA'], os.path.join(out_path, 'results_overview.pdf'))
    logger.info('Results overview figure saved at ' + os.path.join(out_path, 'results_overview.pdf'))

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
