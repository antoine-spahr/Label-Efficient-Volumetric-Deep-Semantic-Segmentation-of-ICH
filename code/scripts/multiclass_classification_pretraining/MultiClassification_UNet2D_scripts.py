"""
author: Antoine Spahr

date : 03.12.2020

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
from datetime import datetime
from prettytable import PrettyTable

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.python_utils import AttrDict
from src.models.optim.Classifier import MultiClassifier
from src.models.optim.UNet2D import UNet2D
from src.models.networks.UNet import UNet, UNet_Encoder
import src.models.optim.LossFunctions
from src.dataset.datasets import public_SegICH_Dataset2D, RSNA_dataset
import src.dataset.transforms as tf
from src.postprocessing.analyse_exp import analyse_supervised_exp, analyse_representation_exp

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    UNet2D pretrained on binary classification with the RSNA dataset and finetuned with the public data.
    """
    # load the config file
    cfg = AttrDict.from_json_path(config_path)

    # Make Outputs directories
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    out_path_selfsup = os.path.join(out_path, 'classification_pretrain/')
    out_path_sup = os.path.join(out_path, 'supervised_train/')
    os.makedirs(out_path_selfsup, exist_ok=True)
    for k in range(cfg.Sup.split.n_fold):
        os.makedirs(os.path.join(out_path_sup, f'Fold_{k+1}/pred/'), exist_ok=True)

    # Initialize random seed
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # Set number of thread
    if cfg.n_thread > 0: torch.set_num_threads(cfg.n_thread)
    # set device
    if cfg.device:
        cfg.device = torch.device(cfg.device)
    else:
        cfg.device = get_available_device()

    ####################################################
    # Self-supervised training on Multi Classification #
    ####################################################
    # Initialize Logger
    logger = initialize_logger(os.path.join(out_path_selfsup, 'log.txt'))
    if os.path.exists(os.path.join(out_path_selfsup, f'checkpoint.pt')):
        logger.info('\n' + '#'*30 + f'\n Recovering Session \n' + '#'*30)
    logger.info(f"Experiment : {cfg.exp_name}")

    # Load RSNA data csv
    df_rsna = pd.read_csv(os.path.join(cfg.path.data.SSL, 'slice_info.csv'), index_col=0)

    # Keep only fractions sample
    if cfg.SSL.dataset.n_data_1 >= 0:
        df_rsna_ICH = df_rsna[df_rsna.Hemorrhage == 1].sample(n=cfg.SSL.dataset.n_data_1, random_state=cfg.seed)
    else:
        df_rsna_ICH = df_rsna[df_rsna.Hemorrhage == 1]
    if cfg.SSL.dataset.f_data_0 >= 0: # nbr of normal as fraction of nbr of ICH samples
        df_rsna_noICH = df_rsna[df_rsna.Hemorrhage == 0].sample(n=cfg.SSL.dataset.f_data_0 * len(df_rsna_ICH), random_state=cfg.seed)
    else:
        df_rsna_noICH = df_rsna[df_rsna.Hemorrhage == 0]
    df_rsna = pd.concat([df_rsna_ICH, df_rsna_noICH], axis=0)

    # Split data to keep few for evaluation in a strafied way
    train_df, test_df = train_test_split(df_rsna, test_size=cfg.SSL.dataset.frac_eval, stratify=df_rsna.Hemorrhage, random_state=cfg.seed)
    logger.info('\n' + str(get_split_summary_table(df_rsna, train_df, test_df)))

    # Make dataset : Train --> BinaryClassification, Test --> BinaryClassification
    train_RSNA_dataset = RSNA_dataset(train_df, cfg.path.data.SSL,
                                 augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.SSL.dataset.augmentation.train.items()],
                                 window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size,
                                 mode='multi_classification')
    test_RSNA_dataset = RSNA_dataset(test_df, cfg.path.data.SSL,
                                 augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.SSL.dataset.augmentation.eval.items()],
                                 window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size,
                                 mode='multi_classification')

    logger.info(f"Data will be loaded from {cfg.path.data.SSL}.")
    logger.info(f"CT scans will be windowed on [{cfg.data.win_center-cfg.data.win_width/2} ; {cfg.data.win_center + cfg.data.win_width/2}]")
    logger.info(f"CT scans will be resized to {cfg.data.size}x{cfg.data.size}")
    logger.info(f"Training online data transformation: \n\n {str(train_RSNA_dataset.transform)}\n")
    logger.info(f"Evaluation online data transformation: \n\n {str(test_RSNA_dataset.transform)}\n")

    # Make U-Net-Encoder architecture
    net_ssl = UNet_Encoder(**cfg.SSL.net).to(cfg.device)
    net_params = [f"--> {k} : {v}" for k, v in cfg.SSL.net.items()]
    logger.info("UNet like Multi Classifier \n\t" + "\n\t".join(net_params))
    logger.info(f"The Multi Classifier has {sum(p.numel() for p in net_ssl.parameters())} parameters.")

    # Make Model
    cfg.SSL.train.model_param.lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.SSL.train.model_param.lr_scheduler) # convert scheduler name to scheduler class object
    if cfg.SSL.train.model_param.loss_fn == 'BCEWithLogitsLoss':
        df_rsna['no_Hemorrhage'] = 1 - df_rsna.Hemorrhage
        class_weight_list = ((len(df_rsna) - df_rsna[train_RSNA_dataset.class_name].sum()) / df_rsna[train_RSNA_dataset.class_name].sum()).values # define CE weighting from train dataset
        cfg.SSL.train.model_param.loss_fn_kwargs['pos_weight'] = torch.tensor(class_weight_list, device=cfg.device) # add weighting to CE kwargs
    try:
        cfg.SSL.train.model_param.loss_fn = getattr(torch.nn, cfg.SSL.train.model_param.loss_fn) # convert loss_fn name to nn.Module class object
    except AttributeError:
        cfg.SSL.train.model_param.loss_fn = getattr(src.models.optim.LossFunctions, cfg.SSL.train.model_param.loss_fn)

    #torch.tensor([1 - w_ICH, w_ICH], device=cfg.device).float()

    classifier = MultiClassifier(net_ssl, device=cfg.device, print_progress=cfg.print_progress, **cfg.SSL.train.model_param)

    train_params = [f"--> {k} : {v}" for k, v in cfg.SSL.train.model_param.items()]
    logger.info("Classifer Training Parameters \n\t" + "\n\t".join(train_params))

    # Load weights if specified
    if cfg.SSL.train.model_path_to_load:
        model_path = cfg.SSL.train.model_path_to_load
        classifier.load_model(model_path, map_location=cfg.device)
        logger.info(f"Classifer Model succesfully loaded from {cfg.SSL.train.model_path_to_load}")

    # train if needed
    if cfg.SSL.train.model_param.n_epoch > 0:
        classifier.train(train_RSNA_dataset, valid_dataset=test_RSNA_dataset,
                        checkpoint_path=os.path.join(out_path_selfsup, f'checkpoint.pt'))

    # evaluate
    auc, acc, sub_acc, recall, precision, f1 = classifier.evaluate(test_RSNA_dataset, save_tsne=True, return_scores=True)
    logger.info(f"Classifier Test AUC : {auc:.2%}")
    logger.info(f"Classifier Test Accuracy : {acc:.2%}")
    logger.info(f"Classifier Test Subset Accuracy : {sub_acc:.2%}")
    logger.info(f"Classifier Test Recall : {recall:.2%}")
    logger.info(f"Classifier Test Precision : {precision:.2%}")
    logger.info(f"Classifier Test F1-score : {f1:.2%}")

    # save model, outputs
    classifier.save_model_state_dict(os.path.join(out_path_selfsup, 'pretrained_unet_enc.pt'))
    logger.info("Pre-trained U-Net encoder on binary classification saved at " + os.path.join(out_path_selfsup, 'pretrained_unet_enc.pt'))
    classifier.save_outputs(os.path.join(out_path_selfsup, 'outputs.json'))
    logger.info("Classifier outputs saved at " + os.path.join(out_path_selfsup, 'outputs.json'))
    test_df.reset_index(drop=True).to_csv(os.path.join(out_path_selfsup, 'eval_data_info.csv'))
    logger.info("Evaluation data info saved at " + os.path.join(out_path_selfsup, 'eval_data_info.csv'))

    # delete any checkpoints
    if os.path.exists(os.path.join(out_path_selfsup, f'checkpoint.pt')):
        os.remove(os.path.join(out_path_selfsup, f'checkpoint.pt'))
        logger.info('Checkpoint deleted.')

    # get weights state dictionnary
    pretrained_unet_weights = classifier.get_state_dict()

    ###################################################################
    # Supervised fine-training of U-Net  with K-Fold Cross-Validation #
    ###################################################################
    # load annotated data csv
    data_info_df = pd.read_csv(os.path.join(cfg.path.data.Sup, 'ct_info.csv'), index_col=0)
    patient_df = pd.read_csv(os.path.join(cfg.path.data.Sup, 'patient_info.csv'), index_col=0)

    # Make K-Fold spolit at patient level
    skf = StratifiedKFold(n_splits=cfg.Sup.split.n_fold, shuffle=cfg.Sup.split.shuffle, random_state=cfg.seed)

    # define scheduler and loss_fn as object
    cfg.Sup.train.model_param.lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.Sup.train.model_param.lr_scheduler) # convert scheduler name to scheduler class object
    cfg.Sup.train.model_param.loss_fn = getattr(src.models.optim.LossFunctions, cfg.Sup.train.model_param.loss_fn) # convert loss_fn name to nn.Module class object

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
            n_remove = int(max(0, len(train_df[train_df.Hemorrhage == 0]) - cfg.Sup.dataset.frac_negative * len(train_df[train_df.Hemorrhage == 1])))
            df_remove = train_df[train_df.Hemorrhage == 0].sample(n=n_remove, random_state=cfg.seed)
            train_df = train_df[~train_df.index.isin(df_remove.index)]
            logger.info('\n' + str(get_split_summary_table(data_info_df, train_df, test_df)))

            # Make datasets
            train_dataset = public_SegICH_Dataset2D(train_df, cfg.path.data.Sup,
                                                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.Sup.dataset.augmentation.train.items()],
                                                    window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size)
            test_dataset = public_SegICH_Dataset2D(test_df, cfg.path.data.Sup,
                                                   augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.Sup.dataset.augmentation.eval.items()],
                                                   window=(cfg.data.win_center, cfg.data.win_width), output_size=cfg.data.size)
            logger.info(f"Data will be loaded from {cfg.path.data.Sup}.")
            logger.info(f"CT scans will be windowed on [{cfg.data.win_center-cfg.data.win_width/2} ; {cfg.data.win_center + cfg.data.win_width/2}]")
            logger.info(f"CT scans will be resized to {cfg.data.size}x{cfg.data.size}")
            logger.info(f"Training online data transformation: \n\n {str(train_dataset.transform)}\n")
            logger.info(f"Evaluation online data transformation: \n\n {str(test_dataset.transform)}\n")

            # Make U-Net architecture
            unet_sup = UNet(**cfg.Sup.net).to(cfg.device)
            net_params = [f"--> {k} : {v}" for k, v in cfg.Sup.net.items()]
            logger.info("UNet-2D params \n\t" + "\n\t".join(net_params))
            logger.info(f"The U-Net2D has {sum(p.numel() for p in unet_sup.parameters())} parameters.")

            # Make Model
            unet2D = UNet2D(unet_sup, device=cfg.device, print_progress=cfg.print_progress, **cfg.Sup.train.model_param)

            train_params = [f"--> {k} : {v}" for k, v in cfg.Sup.train.model_param.items()]
            logger.info("UNet-2D Training Parameters \n\t" + "\n\t".join(train_params))

            # ????? load model if specified ?????

            # transfer weights learn with context restoration
            logger.info('Initialize U-Net2D with weights learned with context_restoration on RSNA.')
            unet2D.transfer_weights(pretrained_unet_weights, verbose=True)

            # Train U-net
            eval_dataset = test_dataset if cfg.Sup.train.validate_epoch else None
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
    save_mean_fold_dice(out_path_sup, cfg.Sup.split.n_fold)
    logger.info('Average Scores saved at ' + os.path.join(out_path_sup, 'average_scores.txt'))

    # Save all volumes prediction csv
    df_list = [pd.read_csv(os.path.join(out_path_sup, f'Fold_{i+1}/pred/volume_prediction_scores.csv')) for i in range(cfg.Sup.split.n_fold)]
    all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    all_df.to_csv(os.path.join(out_path_sup, 'all_volume_prediction.csv'))
    logger.info('CSV of all volumes prediction saved at ' + os.path.join(out_path_sup, 'all_volume_prediction.csv'))

    # Save config file
    cfg.device = str(cfg.device)
    cfg.SSL.train.model_param.lr_scheduler = str(cfg.SSL.train.model_param.lr_scheduler)
    cfg.Sup.train.model_param.lr_scheduler = str(cfg.Sup.train.model_param.lr_scheduler)
    cfg.SSL.train.model_param.loss_fn = str(cfg.SSL.train.model_param.loss_fn)
    cfg.Sup.train.model_param.loss_fn = str(cfg.Sup.train.model_param.loss_fn)
    cfg.SSL.train.model_param.loss_fn_kwargs.pos_weight = cfg.SSL.train.model_param.loss_fn_kwargs.pos_weight.cpu().data.tolist()
    with open(os.path.join(out_path, 'config.json'), 'w') as fp:
        json.dump(cfg, fp)
    logger.info('Config file saved at ' + os.path.join(out_path, 'config.json'))

    # Analyse results
    analyse_supervised_exp(out_path_sup, cfg.path.data.Sup, n_fold=cfg.Sup.split.n_fold,
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

def get_available_device():
    """
    return the avialable device.
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

if __name__ == "__main__":
    main()
