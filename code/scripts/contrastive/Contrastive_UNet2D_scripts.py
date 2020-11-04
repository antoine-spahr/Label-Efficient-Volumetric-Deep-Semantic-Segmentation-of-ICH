"""
author: Antoine Spahr

date : 21.10.2020

----------

TO DO :
- debug
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

from sklearn.model_selection import StratifiedKFold

#from src.utils.Config import Config
from src.models.optim.Contrastive import Contrastive
from src.models.optim.UNet2D import UNet2D
#from src.models.optim.LossFunctions import BinaryDiceLoss, InfoNCELoss
from src.models.networks.UNet import UNet, UNet_Encoder, Partial_UNet
import src.models.optim.LossFunctions
from src.dataset.datasets import public_SegICH_Dataset2D, RSNA_dataset
import src.dataset.transforms as tf
from src.postprocessing.analyse_exp import analyse_supervised_exp, analyse_representation_exp

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    UNet2D pretrained on the global and/or local contrastive task with the RSNA dataset and finetuned with the public data.
    """
    # load the config file
    with open(config_path, 'r') as fp:
        cfg = json.load(fp)

    # Make Outputs directories
    out_path = os.path.join(cfg['path']['output'], cfg['exp_name'])
    out_path_selfsup = os.path.join(out_path, 'contrastive_pretrain/')
    out_path_sup = os.path.join(out_path, 'supervised_train/')
    os.makedirs(out_path_selfsup, exist_ok=True)
    for k in range(cfg['Sup']['split']['n_fold']):
        os.makedirs(os.path.join(out_path_sup, f'Fold_{k+1}/pred/'), exist_ok=True)

    # Initialize random seed
    seed = cfg['seed']
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # Set number of thread
    if cfg['n_thread'] > 0: torch.set_num_threads(cfg['n_thread'])
    # check if GPU available and select the bigger one
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

    ####################################################
    # Self-supervised training on the Contrastive Task #
    ####################################################
    # Initialize Logger
    logger = initialize_logger(os.path.join(out_path_selfsup, 'log.txt'))
    if os.path.exists(os.path.join(out_path_selfsup, f'checkpoint.pt')):
        logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
    logger.info(f"Experiment : {cfg['exp_name']}")

    # Load RSNA data csv
    df_rsna = pd.read_csv(os.path.join(cfg['path']['data']['SSL'], 'slice_info.csv'), index_col=0)

    # Keep only fractions of negative/positive
    df_rsna_pos = df_rsna[df_rsna.Hemorrhage == 1]
    if cfg['SSL']['dataset']['n_positive'] >= 0:
        df_rsna_pos = df_rsna_pos.sample(n=cfg['SSL']['dataset']['n_positive'], random_state=seed)
    n_neg = int(cfg['SSL']['dataset']['frac_negative'] * len(df_rsna_pos))
    df_rsna_neg = df_rsna[df_rsna.Hemorrhage == 0].sample(n=n_neg, random_state=seed)
    df_rsna_samp = pd.concat([df_rsna_pos, df_rsna_neg], axis=0)

    # Split data to keep few for evaluation
    test_df = df_rsna_samp.sample(n=cfg['SSL']['dataset']['n_eval'], random_state=seed)
    train_df = df_rsna_samp.drop(test_df.index)
    df_rsna_neg = df_rsna[(df_rsna.Hemorrhage == 0)]
    test_df = test_df.append(df_rsna_neg[~df_rsna_neg.isin(train_df.index)].sample(n=cfg['SSL']['dataset']['n_eval_neg'], random_state=seed))
    test_df, train_df = test_df.reset_index(), train_df.reset_index()
    logger.info('\n' + str(get_split_summary_table(df_rsna, train_df, test_df)))

    #################### Self-supervised training on the GLOBAL Contrastive Task ####################
    logger.info('Phase 1 : GLOBAL contrastive training.')
    # Make dataset
    train_RSNA_dataset = RSNA_dataset(train_df, cfg['path']['data']['SSL'],
                                 augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['SSL']['dataset']['augmentation']['train'].items()],
                                 window=(cfg['data']['win_center'], cfg['data']['win_width']), output_size=cfg['data']['size'], mode='contrastive',
                                 contrastive_augmentation=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['SSL']['dataset']['contrastive_augmentation']['global'].items()])
    test_RSNA_dataset = RSNA_dataset(test_df, cfg['path']['data']['SSL'],
                                 augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['SSL']['dataset']['augmentation']['eval'].items()],
                                 window=(cfg['data']['win_center'], cfg['data']['win_width']), output_size=cfg['data']['size'],
                                 mode='standard')
    logger.info(f"Data will be loaded from {cfg['path']['data']['SSL']}.")
    logger.info(f"CT scans will be windowed on [{cfg['data']['win_center']-cfg['data']['win_width']/2} ; {cfg['data']['win_center'] + cfg['data']['win_width']/2}]")
    logger.info(f"CT scans will be resized to {cfg['data']['size']}x{cfg['data']['size']}")
    logger.info(f"Training online data transformation: \n\n {str(train_RSNA_dataset.transform + train_RSNA_dataset.contrastive_transform)}\n")
    logger.info(f"Evaluation online data transformation: \n\n {str(test_RSNA_dataset.transform)}\n")

    # Make Encoder architecture
    net_global = UNet_Encoder(depth=cfg['SSL']['global']['net']['depth'], top_filter=cfg['SSL']['global']['net']['top_filter'],
                              use_3D=cfg['SSL']['global']['net']['3D'], in_channels=cfg['SSL']['global']['net']['in_channels'],
                              MLP_head=cfg['SSL']['global']['net']['MLP_head'])
    net_global.to(cfg['device'])
    logger.info(f"U-Net2D Encoder initialized with a depth of {cfg['SSL']['global']['net']['depth']}"
                f" and a number of initial filter of {cfg['SSL']['global']['net']['top_filter']},")
    logger.info(f"Compression is performed with a MLP head : {cfg['SSL']['global']['net']['MLP_head']}.")
    logger.info(f"U-Net2D Encoder takes {cfg['SSL']['global']['net']['in_channels']} channels as input.")
    logger.info(f"The U-Net2D Encoder has {sum(p.numel() for p in net_global.parameters())} parameters.")

    # Make model
    cfg['SSL']['global']['train']['loss_fn_kwargs'].update({"set_size":cfg['SSL']['global']['train']['batch_size'], "device": str(cfg['device'])})
    ctrst_global = Contrastive(net_global, n_epoch=cfg['SSL']['global']['train']['n_epoch'], batch_size=cfg['SSL']['global']['train']['batch_size'],
                               lr=cfg['SSL']['global']['train']['lr'], lr_scheduler=getattr(torch.optim.lr_scheduler, cfg['SSL']['global']['train']['lr_scheduler']),
                               lr_scheduler_kwargs=cfg['SSL']['global']['train']['lr_scheduler_kwargs'],
                               loss_fn=getattr(src.models.optim.LossFunctions, cfg['SSL']['global']['train']['loss_fn']),
                               loss_fn_kwargs=cfg['SSL']['global']['train']['loss_fn_kwargs'],
                               weight_decay=cfg['SSL']['global']['train']['weight_decay'], num_workers=cfg['SSL']['global']['train']['num_workers'],
                               device=cfg['device'], print_progress=cfg['print_progress'], is_global=True)

    # Load weights if specified
    if cfg['SSL']['global']['train']['model_path_to_load']:
        model_path = cfg['SSL']['global']['train']['model_path_to_load']
        ctrst_global.load_model(model_path, map_location=cfg['device'])

    # Train if required
    if cfg['SSL']['global']['train']['n_epoch'] > 0:
        train_params = []
        for key, value in cfg['SSL']['global']['train'].items():
            train_params.append(f"--> {key} : {value}")
        logger.info('Global Training settings:\n\t' + '\n\t'.join(train_params))

        ctrst_global.train(train_RSNA_dataset, checkpoint_path=os.path.join(out_path_selfsup, f'checkpoint_global.pt'))

    # Evaluate
    ctrst_global.evaluate(test_RSNA_dataset)

    # save model
    ctrst_global.save_model(os.path.join(out_path_selfsup, 'pretrained_global.pt'))
    logger.info("U-Net Encoder pretrained on global contrastive task saved at " + os.path.join(out_path_selfsup, 'pretrained_global.pt'))
    ctrst_global.save_outputs(os.path.join(out_path_selfsup, 'outputs.json'))
    logger.info("Global Contrastive outputs saved at " + os.path.join(out_path_selfsup, 'outputs.json'))
    test_df.to_csv(os.path.join(out_path_selfsup, 'eval_data_info.csv'))
    logger.info("Evaluation data info saved at " + os.path.join(out_path_selfsup, 'eval_data_info.csv'))

    # get weights
    global_weights = ctrst_global.get_state_dict()

    #################### Self-supervised training on the LOCAL Contrastive Task ####################
    # train on local contrastive if required. Otherwise to directly fine tuning of supervised.
    if not cfg['SSL']['local']['skip']:
        logger.info('Phase 2 : LOCAL contrastive training.')
        # Make dataset
        train_RSNA_dataset = RSNA_dataset(train_df, cfg['path']['data']['SSL'],
                                     augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['SSL']['dataset']['augmentation']['train'].items()],
                                     window=(cfg['data']['win_center'], cfg['data']['win_width']), output_size=cfg['data']['size'], mode='contrastive',
                                     contrastive_augmentation=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['SSL']['dataset']['contrastive_augmentation']['local'].items()])
        logger.info(f"Data will be loaded from {cfg['path']['data']['SSL']}.")
        logger.info(f"CT scans will be windowed on [{cfg['data']['win_center']-cfg['data']['win_width']/2} ; {cfg['data']['win_center'] + cfg['data']['win_width']/2}]")
        logger.info(f"CT scans will be resized to {cfg['data']['size']}x{cfg['data']['size']}")
        logger.info(f"Training online data transformation: \n\n {str(train_RSNA_dataset.transform + train_RSNA_dataset.contrastive_transform)}\n")

        # Make Partial Unet architecture
        net_local = Partial_UNet(depth=cfg['SSL']['local']['net']['depth'], n_decoder=cfg['SSL']['local']['net']['n_decoder'],
                                 top_filter=cfg['SSL']['local']['net']['top_filter'], use_3D=cfg['SSL']['local']['net']['3D'],
                                 in_channels=cfg['SSL']['local']['net']['in_channels'], head_channel=cfg['SSL']['local']['net']['head_channel'],
                                 bilinear=cfg['SSL']['local']['net']['bilinear'])
        net_local.to(cfg['device'])
        logger.info(f"Partial U-Net2D initialized with a depth of {cfg['SSL']['local']['net']['depth']}"
                    f" and a number of initial filter of {cfg['SSL']['local']['net']['top_filter']},")
        logger.info(f"{cfg['SSL']['local']['net']['n_decoder']} decoder layer with a convolutionnal head: {cfg['SSL']['local']['net']['head_channel']}, are used to generate the feature map.")
        logger.info(f"Partial U-Net2D takes {cfg['SSL']['global']['net']['in_channels']}.")
        logger.info(f"The Partial U-Net2D has {sum(p.numel() for p in net_local.parameters())} parameters.")

        # Make model
        cfg['SSL']['local']['loss_fn_kwargs'].update({"device": str(cfg['device'])})
        ctrst_local = Contrastive(net_local, n_epoch=cfg['SSL']['local']['train']['n_epoch'], batch_size=cfg['SSL']['local']['train']['batch_size'],
                                  lr=cfg['SSL']['local']['train']['lr'], lr_scheduler=getattr(torch.optim.lr_scheduler, cfg['SSL']['local']['train']['lr_scheduler']),
                                  lr_scheduler_kwargs=cfg['SSL']['local']['train']['lr_scheduler_kwargs'],
                                  loss_fn=getattr(src.models.optim.LossFunctions, cfg['SSL']['local']['train']['loss_fn']),
                                  loss_fn_kwargs=cfg['SSL']['local']['loss_fn_kwargs'],
                                  weight_decay=cfg['SSL']['train']['local']['weight_decay'], num_workers=cfg['SSL']['local']['train']['num_workers'],
                                  device=cfg['device'], print_progress=cfg['print_progress'], is_global=False)

        # transfer and freeze Global weights
        logger.info(f"Initialize Partial U-Net2D with the {'freezed' if cfg['SSL']['train']['local']['freeze'] else ''} weights learned with global contrastive on RSNA.")
        ctrst_local.transfer_weights(global_weights, verbose=True, freeze=cfg['SSL']['train']['local']['freeze'])
        logger.info(f"Succesfully tranferred the weights. The partial U-Net2D has now {sum(p.numel() for p in net_local.parameters() if p.requires_grad)} trainable parameters.")

        # Load a model if given
        if cfg['SSL']['local']['train']['model_path_to_load']:
            model_path = cfg['SSL']['local']['train']['model_path_to_load']
            ctrst_local.load_model(model_path, map_location=cfg['device'])

        # Train if required
        if cfg['SSL']['local']['train']['n_epoch'] > 0:
            train_params = []
            for key, value in cfg['SSL']['local']['train'].items():
                train_params.append(f"--> {key} : {value}")
            logger.info('Local Training settings:\n\t' + '\n\t'.join(train_params))

            ctrst_local.train(train_RSNA_dataset, checkpoint_path=os.path.join(out_path_selfsup, f'checkpoint_local.pt'))

        # save model
        ctrst_global.save_model(os.path.join(out_path_selfsup, 'pretrained_local.pt'))
        logger.info("Partial U-Net pretrained on local contrastive task saved at " + os.path.join(out_path_selfsup, 'pretrained_local.pt'))
        ctrst_global.save_outputs(os.path.join(out_path_selfsup, 'outputs_local.json'))
        logger.info("Local Contrastive outputs saved at " + os.path.join(out_path_selfsup, 'outputs_local.json'))

        # get weights
        to_transfer_weights = ctrst_local.get_state_dict()
    else:
        to_transfer_weights = global_weights

    ###################################################################
    # Supervised fine-training of U-Net  with K-Fold Cross-Validation #
    ###################################################################
    # load annotated data csv
    data_info_df = pd.read_csv(os.path.join(cfg['path']['data']['Sup'], 'ct_info.csv'), index_col=0)
    patient_df = pd.read_csv(os.path.join(cfg['path']['data']['Sup'], 'patient_info.csv'), index_col=0)

    # Make K-Fold spolit at patient level
    skf = StratifiedKFold(n_splits=cfg['Sup']['split']['n_fold'], shuffle=cfg['Sup']['split']['shuffle'], random_state=seed)

    # iterate over folds
    for k, (train_idx, test_idx) in enumerate(skf.split(patient_df.PatientNumber, patient_df.Hemorrhage)):
        # check if fold's results already exists
        if not os.path.exists(os.path.join(out_path_sup, f'Fold_{k+1}/outputs.json')):
            # initialize logger
            logger = initialize_logger(os.path.join(out_path_sup, f'Fold_{k+1}/log.txt'))
            if os.path.exists(os.path.join(out_path_sup, f'Fold_{k+1}/checkpoint.pt')):
                logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
            logger.info(f"Experiment : {cfg['exp_name']}")
            logger.info(f"Cross-Validation fold {k+1:02}/{cfg['Sup']['split']['n_fold']:02}")

            # extract train/test slice dataframe
            train_df = data_info_df[data_info_df.PatientNumber.isin(patient_df.loc[train_idx,'PatientNumber'].values)]
            test_df = data_info_df[data_info_df.PatientNumber.isin(patient_df.loc[test_idx,'PatientNumber'].values)]
            # samples train dataframe to adjuste negative/positive fractions
            n_remove = int(max(0, len(train_df[train_df.Hemorrhage == 0]) - cfg['Sup']['dataset']['frac_negative'] * len(train_df[train_df.Hemorrhage == 1])))
            df_remove = train_df[train_df.Hemorrhage == 0].sample(n=n_remove, random_state=seed)
            train_df = train_df[~train_df.index.isin(df_remove.index)]
            logger.info('\n' + str(get_split_summary_table(data_info_df, train_df, test_df)))

            # Make datasets
            train_dataset = public_SegICH_Dataset2D(train_df, cfg['path']['data']['Sup'],
                                                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['Sup']['dataset']['augmentation']['train'].items()],
                                                    window=(cfg['data']['win_center'], cfg['data']['win_width']),
                                                    output_size=cfg['data']['size'])
            test_dataset = public_SegICH_Dataset2D(test_df, cfg['path']['data']['Sup'],
                                                   augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg['Sup']['dataset']['augmentation']['eval'].items()],
                                                   window=(cfg['data']['win_center'], cfg['data']['win_width']),
                                                   output_size=cfg['data']['size'])
            logger.info(f"Data will be loaded from {cfg['path']['data']['Sup']}.")
            logger.info(f"CT scans will be windowed on [{cfg['data']['win_center']-cfg['data']['win_width']/2} ; {cfg['data']['win_center'] + cfg['data']['win_width']/2}]")
            logger.info(f"CT scans will be resized to {cfg['data']['size']}x{cfg['data']['size']}")
            logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
            logger.info(f"Evaluation online data transformation: \n\n {str(test_dataset.transform)}\n")

            # Make U-Net architecture
            unet_sup = UNet(depth=cfg['Sup']['net']['depth'], top_filter=cfg['Sup']['net']['top_filter'],
                             use_3D=cfg['Sup']['net']['3D'], in_channels=cfg['Sup']['net']['in_channels'],
                             out_channels=cfg['Sup']['net']['out_channels'], bilinear=cfg['Sup']['net']['bilinear'])
            unet_sup.to(cfg['device'])
            logger.info(f"U-Net2D initialized with a depth of {cfg['Sup']['net']['depth']}"
                        f" and a number of initial filter of {cfg['Sup']['net']['top_filter']},")
            logger.info(f"Reconstruction performed with {'Upsample + Conv' if cfg['Sup']['net']['bilinear'] else 'ConvTranspose'}.")
            logger.info(f"U-Net2D takes {cfg['Sup']['net']['in_channels']} as input channels and {cfg['Sup']['net']['out_channels']} as output channels.")
            logger.info(f"The U-Net2D has {sum(p.numel() for p in unet_sup.parameters())} parameters.")

            # Make Model
            unet2D = UNet2D(unet_sup, n_epoch=cfg['Sup']['train']['n_epoch'], batch_size=cfg['Sup']['train']['batch_size'],
                            lr=cfg['Sup']['train']['lr'], lr_scheduler=getattr(torch.optim.lr_scheduler, cfg['Sup']['train']['lr_scheduler']),
                            lr_scheduler_kwargs=cfg['Sup']['train']['lr_scheduler_kwargs'],
                            loss_fn=getattr(src.models.optim.LossFunctions, cfg['Sup']['train']['loss_fn']),
                            loss_fn_kwargs=cfg['Sup']['train']['loss_fn_kwargs'], weight_decay=cfg['Sup']['train']['weight_decay'],
                            num_workers=cfg['Sup']['train']['num_workers'], device=cfg['device'], print_progress=cfg['print_progress'])

            # transfer weights learn with context restoration
            logger.info('Initialize U-Net2D with weights learned with context_restoration on RSNA.')
            unet2D.transfer_weights(to_transfer_weights, verbose=True)

            # Print training parameters
            train_params = []
            for key, value in cfg['Sup']['train'].items():
                train_params.append(f"--> {key} : {value}")
            logger.info('Training settings:\n\t' + '\n\t'.join(train_params))

            # Train U-net
            eval_dataset = test_dataset if cfg['Sup']['train']['validate_epoch'] else None
            unet2D.train(train_dataset, valid_dataset=eval_dataset, checkpoint_path=os.path.join(out_path_sup, f'Fold_{k+1}/checkpoint.pt'))

            # Evaluate U-Net
            unet2D.evaluate(test_dataset, save_path=os.path.join(out_path_sup, f'Fold_{k+1}/pred/'))

            # Save models and outputs
            unet2D.save_model(os.path.join(out_path_sup, f'Fold_{k+1}/trained_unet.pt'))
            logger.info("Trained U-Net saved at " + os.path.join(out_path_sup, f'Fold_{k+1}/trained_unet.pt'))
            unet2D.save_outputs(os.path.join(out_path_sup, f'Fold_{k+1}/outputs.json'))
            logger.info("Trained statistics saved at " + os.path.join(out_path_sup, f'Fold_{k+1}/outputs.json'))

            # delete checkpoint if exists
            if os.path.exists(os.path.join(out_path_sup, f'Fold_{k+1}/checkpoint.pt')):
                os.remove(os.path.join(out_path_sup, f'Fold_{k+1}/checkpoint.pt'))
                logger.info('Checkpoint deleted.')

    # save mean +/- 1.96 std Dice over Folds
    save_mean_fold_dice(out_path_sup, cfg['Sup']['split']['n_fold'])
    logger.info('Average Scores saved at ' + os.path.join(out_path_sup, 'average_scores.txt'))

    # Save all volumes prediction csv
    df_list = [pd.read_csv(os.path.join(out_path_sup, f'Fold_{i+1}/pred/volume_prediction_scores.csv')) for i in range(cfg['Sup']['split']['n_fold'])]
    all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    all_df.to_csv(os.path.join(out_path_sup, 'all_volume_prediction.csv'))
    logger.info('CSV of all volumes prediction saved at ' + os.path.join(out_path_sup, 'all_volume_prediction.csv'))

    # Save config file
    cfg['device'] = str(cfg['device'])
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info('Config file saved at ' + os.path.join(out_path, 'config.json'))

    # Analyse results
    analyse_supervised_exp(out_path_sup, cfg['path']['data']['Sup'], n_fold=cfg['Sup']['split']['n_fold'],
                           config_folder=out_path, save_fn=os.path.join(out_path, 'results_supervised_overview.pdf'))
    logger.info('Results overview figure saved at ' + os.path.join(out_path, 'results_supervised_overview.pdf'))
    analyse_representation_exp(out_path_selfsup, save_fn=os.path.join(out_path, 'results_self-supervised_overview.pdf'))
    logger.info('Results overview figure saved at ' + os.path.join(out_path, 'results_self-supervised_overview.pdf'))

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

def save_mean_fold_dice(out_path, n_fold):
    """
    Save the mean dice over folds in a txt file.
    """
    # recover mean Fold Dice
    scores_list = []
    for k in range(n_fold):
        with open(os.path.join(out_path, f'Fold_{k+1}/outputs.json'), 'r') as f:
            out = json.load(f)
        scores_list.append([out['eval']['dice']['all'], out['eval']['dice']['positive']])
    # take mean and std
    means = np.array(scores_list).mean(axis=0)
    CI95 = 1.96*np.array(scores_list).std(axis=0)
    # save in .txt file
    with open(os.path.join(out_path, 'average_scores.txt'), 'w') as f:
        f.write(f'Dice = {means[0]} +/- {CI95[0]}\n')
        f.write(f'Dice (ICH) = {means[1]} +/- {CI95[1]}\n')

if __name__ == '__main__':
    main()
