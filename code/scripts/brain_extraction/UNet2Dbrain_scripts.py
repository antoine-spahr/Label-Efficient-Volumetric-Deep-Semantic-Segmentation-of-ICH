"""
author: Antoine Spahr

date : 04.11.2020

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

from  sklearn.model_selection import KFold

#from src.utils.Config import Config
from src.models.optim.UNet2D import UNet2D
from src.dataset.datasets import brain_extract_Dataset2D
from src.models.optim.LossFunctions import BinaryDiceLoss
import src.dataset.transforms as tf
import src.models.optim.LossFunctions
from src.models.networks.UNet import UNet
from src.postprocessing.analyse_exp import analyse_supervised_exp

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Train and evaluate a 2D UNet on the brain extraction dataset using the parameters sepcified on the JSON at the
    config_path. The evaluation is performed by k-fold cross-validation.
    """
    # load config file
    with open(config_path, 'r') as fp:
        cfg = json.load(fp)

    # Make Output directories
    out_path = os.path.join(cfg['path']['OUTPUT'], cfg['exp_name'])
    os.makedirs(out_path, exist_ok=True)
    for k in range(cfg['split']['n_fold']):
        os.makedirs(os.path.join(out_path, f'Fold_{k+1}/pred/'), exist_ok=True)

    # Initialize random seed to given seed
    seed = cfg['seed']
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # Load data csv
    data_info_df = pd.read_csv(os.path.join(cfg['path']['DATA'], 'slice_info.csv'))
    data_info_df = data_info_df.drop(data_info_df.columns[0], axis=1)
    vol_df = pd.read_csv(os.path.join(cfg['path']['DATA'], 'volume_info.csv'))
    vol_df = vol_df.drop(vol_df.columns[0], axis=1)

    # Generate Cross-Val indices at the patient level
    skf = KFold(n_splits=cfg['split']['n_fold'], shuffle=cfg['split']['shuffle'], random_state=seed)
    # iterate over folds
    for k, (train_idx, test_idx) in enumerate(skf.split(vol_df.id)):
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

            logger.info(f"Experiment : {cfg['exp_name']}")
            logger.info(f"Cross-Validation fold {k+1:02}/{cfg['split']['n_fold']:02}")

            # initialize nbr of thread
            if cfg['n_thread'] > 0:
                torch.set_num_threads(cfg['n_thread'])
            logger.info(f"Number of thread : {cfg['n_thread']}")
            # check if GPU available
            #cfg.settings['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if cfg['device'] is not None:
                cfg['device'] = torch.device(cfg['device'])
            else:
                if torch.cuda.is_available():
                    free_mem, device_idx = 0.0, 0
                    for d in range(torch.cuda.device_count()):
                        mem = torch.cuda.get_device_properties(d).total_memory - torch.cuda.memory_allocated(d)
                        if mem > free_mem:
                            device_idx = d
                            free_mem = mem
                    cfg['device'] = torch.device(f'cuda:{device_idx}')
                else:
                    cfg['device'] = torch.device('cpu')
            logger.info(f"Device : {cfg['device']}")

            # extract train and test DataFrames + print summary (n samples positive and negatives)
            train_df = data_info_df[data_info_df.volume.isin(vol_df.loc[train_idx,'id'].values)]
            test_df = data_info_df[data_info_df.volume.isin(vol_df.loc[test_idx,'id'].values)]
            # sample the dataframe to have more or less normal slices
            #n_remove = int(max(0, len(train_df[train_df.Hemorrhage == 0]) - cfg['dataset']['frac_negative'] * len(train_df[train_df.Hemorrhage == 1])))
            #df_remove = train_df[train_df.Hemorrhage == 0].sample(n=n_remove, random_state=seed)
            #train_df = train_df[~train_df.index.isin(df_remove.index)]
            logger.info('\n' + str(get_split_summary_table(data_info_df, train_df, test_df)))

            # Make Dataset + print online augmentation summary
            train_dataset = brain_extract_Dataset2D(train_df, cfg['path']['DATA'],
                                                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['data']['augmentation']['train'].items()],
                                                    window=(cfg['data']['win_center'], cfg['data']['win_width']),
                                                    output_size=cfg['data']['size'])
            test_dataset = brain_extract_Dataset2D(test_df, cfg['path']['DATA'],
                                                   augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['data']['augmentation']['eval'].items()],
                                                   window=(cfg['data']['win_center'], cfg['data']['win_width']),
                                                   output_size=cfg['data']['size'])
            logger.info(f"Data will be loaded from {cfg['path']['DATA']}.")
            logger.info(f"CT scans will be windowed on [{cfg['data']['win_center']-cfg['data']['win_width']/2} ; {cfg['data']['win_center'] + cfg['data']['win_width']/2}]")
            logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
            logger.info(f"Evaluation online data transformation: \n\n {str(test_dataset.transform)}\n")

            # Make architecture (and print summmary ??)
            unet_arch = UNet(depth=cfg['net']['depth'], top_filter=cfg['net']['top_filter'],
                             use_3D=cfg['net']['3D'], in_channels=cfg['net']['in_channels'],
                             out_channels=cfg['net']['out_channels'], bilinear=cfg['net']['bilinear'])
            unet_arch.to(cfg['device'])
            logger.info(f"U-Net2D initialized with a depth of {cfg['net']['depth']}"
                        f" and a number of initial filter of {cfg['net']['top_filter']},")
            logger.info(f"Reconstruction performed with {'Upsample + Conv' if cfg['net']['bilinear'] else 'ConvTranspose'}.")
            logger.info(f"U-Net2D takes {cfg['net']['in_channels']} as input channels and {cfg['net']['out_channels']} as output channels.")
            logger.info(f"The U-Net2D has {sum(p.numel() for p in unet_arch.parameters())} parameters.")

            # Make model
            unet2D = UNet2D(unet_arch, n_epoch=cfg['train']['n_epoch'], batch_size=cfg['train']['batch_size'],
                            lr=cfg['train']['lr'], lr_scheduler=getattr(torch.optim.lr_scheduler, cfg['train']['lr_scheduler']),
                            lr_scheduler_kwargs=cfg['train']['lr_scheduler_kwargs'],
                            loss_fn=getattr(src.models.optim.LossFunctions, cfg['train']['loss_fn']),
                            loss_fn_kwargs=cfg['train']['loss_fn_kwargs'], weight_decay=cfg['train']['weight_decay'],
                            num_workers=cfg['train']['num_workers'], device=cfg['device'],
                            print_progress=cfg['print_progress'])

            # use pretrain weight if required
            if cfg['train']['init_weight']:
                init_state_dict = torch.load(cfg['train']['init_weight'], map_location=cfg['device'])
                unet2D.transfer_weights(init_state_dict, verbose=True)

            # Load model if required
            if cfg['train']['model_path_to_load']:
                if isinstance(cfg['train']['model_path_to_load'], str):
                    model_path = cfg['train']['model_path_to_load']
                    unet2D.load_model(model_path, map_location=cfg['device'])
                elif isinstance(cfg['train']['model_path_to_load'], list):
                    model_path = cfg['train']['model_path_to_load'][k]
                    unet2D.load_model(model_path, map_location=cfg['device'])
                else:
                    raise ValueError(f'Model path to load type not understood.')
                logger.info(f"2D U-Net model loaded from {model_path}")

            # print Training hyper-parameters
            train_params = []
            for key, value in cfg['train'].items():
                train_params.append(f"--> {key} : {value}")
            logger.info('Training settings:\n\t' + '\n\t'.join(train_params))

            # Train model
            eval_dataset = test_dataset if cfg['train']['validate_epoch'] else None
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
    for k in range(cfg['split']['n_fold']):
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
    df_list = [pd.read_csv(os.path.join(out_path, f'Fold_{i+1}/pred/volume_prediction_scores.csv')) for i in range(cfg['split']['n_fold'])]
    all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    all_df.to_csv(os.path.join(out_path, 'all_volume_prediction.csv'))
    logger.info('CSV of all volumes prediction saved at ' + os.path.join(out_path, 'all_volume_prediction.csv'))

    # Save config file
    cfg['device'] = str(cfg['device'])
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info('Config file saved at ' + os.path.join(out_path, 'config.json'))

    # Analyse results
    analyse_supervised_exp(out_path, cfg['path']['DATA'], cfg['split']['n_fold'], save_fn=os.path.join(out_path, 'results_overview.pdf'))
    logger.info('Results overview figure saved at ' + os.path.join(out_path, 'results_overview.pdf'))

def get_split_summary_table(all_df, train_df, test_df):
    """
    return a table summarising the data split.
    """
    table = PrettyTable()
    table.field_names = ['set', 'N total', 'N no-brain', 'N brain', 'frac no-brain', 'frac brain']
    for df, name in zip([all_df, train_df, test_df],['All', 'Train', 'Test']):
        table.add_row([name, len(df), len(df[df.mask_fn == 'None']), len(df[df.mask_fn != 'None']),
                   f"{len(df[df.mask_fn == 'None'])/len(df):.3%}", f"{len(df[df.mask_fn != 'None'])/len(df):.3%}"])
    return table

if __name__ == '__main__':
    main()
