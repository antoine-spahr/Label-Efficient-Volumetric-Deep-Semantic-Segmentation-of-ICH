"""
author: Antoine Spahr

date : 01.03.2021

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
import torch.nn as nn
import torch.cuda
import numpy as np
import pandas as pd
import skimage.io as io
import skimage.filters
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.dataset.datasets import public_SegICH_Dataset2D
from src.models.networks.AE_net import AE_net
import src.models.networks.ResNet as rn
from src.utils.python_utils import AttrDict

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """ """
    # load config
    cfg = AttrDict.from_json_path(config_path)

    # make outputs dir
    out_path = os.path.join(cfg.path.output, cfg.exp_name)
    os.makedirs(out_path, exist_ok=True)

    # initialize seed
    if cfg.seed != -1:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # initialize logger
    logger = initialize_logger(os.path.join(out_path, 'log.txt'))
    logger.info(f"Experiment : {cfg.exp_name}")

    # set device
    if cfg.device:
        cfg.device = torch.device(cfg.device)
    else:
        cfg.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Device set to {cfg.device}.")

    # get Dataset
    data_info_df = pd.read_csv(os.path.join(cfg.path.data, 'ct_info.csv'), index_col=0)
    dataset = public_SegICH_Dataset2D(data_info_df, cfg.path.data,
                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.data.augmentation.items()],
                    output_size=cfg.data.size, window=(cfg.data.win_center, cfg.data.win_width))

    # load inpainting model
    cfg_ae = AttrDict.from_json_path(cfg.ae_cfg_path)
    ae_net = AE_net(**cfg_ae.net)
    loaded_state_dict = torch.load(cfg.ae_model_path, map_location=cfg.device)
    ae_net.load_state_dict(loaded_state_dict)
    ae_net = ae_net.to(cfg.device).eval()
    logger.info(f"AE model succesfully loaded from {cfg.ae_model_path}")

    # Load Classifier
    if cfg.classifier_model_path is not None:
        cfg_classifier = AttrDict.from_json_path(os.path.join(cfg.classifier_model_path, 'config.json'))
        classifier = getattr(rn, cfg_classifier.net.resnet)(num_classes=cfg_classifier.net.num_classes, input_channels=cfg_classifier.net.input_channels)
        classifier_state_dict = torch.load(os.path.join(cfg.classifier_model_path, 'resnet_state_dict.pt'), map_location=cfg.device)
        classifier.load_state_dict(classifier_state_dict)
        classifier = classifier.to(cfg.device)
        classifier.eval()
        logger.info(f"ResNet classifier model succesfully loaded from {os.path.join(cfg.classifier_model_path, 'resnet_state_dict.pt')}")

    # iterate over dataset
    all_pred = []
    for i, sample in enumerate(dataset):
        # unpack data
        image, target, id, slice = sample
        logger.info("="*25 + f" SAMPLE {i+1:04}/{len(dataset):04} - Volume {id:03} Slice {slice:03} " + "="*25)

        # Classify sample
        if cfg.classifier_model_path is not None:
            with torch.no_grad():
                input_clss = image.unsqueeze(0).to(cfg.device).float()
                pred_score = nn.functional.softmax(classifier(input_clss), dim=1)[:,1] # take columns of softmax of positive class as score
                pred = 1 if pred_score >= cfg.classification_threshold else 0
        else:
            pred = 1 # if not classifier given, all slices are processed

        # process slice if classifier has detected Hemorrhage
        if pred == 1:
            logger.info(f"ICH detected. Compute anomaly mask through AE reconstruction.")
            # Detect anomalies using the robuste approach
            ad_map, ad_mask = compute_anomaly(ae_net, image, alpha_low=cfg.alpha_low, alpha_high=cfg.alpha_high, device=cfg.device)
            logger.info(f"{ad_mask.sum()} anomalous pixels detected.")
            # save ad_mask
            ad_mask_fn = f"{id}/{slice}_anomalies.bmp"
            save_path = os.path.join(out_path, 'pred/', ad_mask_fn)
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            io.imsave(save_path, img_as_ubyte(ad_mask), check_contrast=False)
            # save anomaly map
            ad_map_fn = f"{id}/{slice}_map_anomalies.png"
            save_path_map = os.path.join(out_path, 'pred/', ad_map_fn)
            io.imsave(save_path_map, img_as_ubyte(rescale_intensity(ad_map, out_range=(0.0, 1.0))), check_contrast=False)
        else:
            logger.info(f"No ICH detected. Set the anomaly mask to zeros.")
            ad_mask = np.zeros_like(target[0].numpy())
            ad_mask_fn, ad_map_fn = 'None', 'None'

        # compute confusion matrix with target ICH mask
        tn, fp, fn, tp = confusion_matrix(target[0].numpy().ravel(), ad_mask.ravel(), labels=[0,1]).ravel()
        auc = roc_auc_score(target[0].numpy().ravel(), ad_map.ravel()) if torch.any(target[0]) else 'None'
        # append to all_pred list
        all_pred.append({'id': id.item(), 'slice': slice.item(), 'label':target.max().item(), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'AUC': auc, 'ad_mask_fn': ad_mask_fn, 'ad_map_fn': ad_map_fn})

    # make a dataframe of all predictions
    slice_df = pd.DataFrame(all_pred)
    volume_df = slice_df[['id', 'label', 'TP', 'TN', 'FP', 'FN']].groupby('id').agg({'label':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})

    # Compute Dice and Volume Dice
    slice_df['Dice'] = (2*slice_df.TP + 1) / (2*slice_df.TP + slice_df.FP + slice_df.FN + 1)
    volume_df['Dice'] = (2*volume_df.TP + 1) / (2*volume_df.TP + volume_df.FP + volume_df.FN + 1)
    logger.info(f"Mean slice dice : {slice_df.Dice.mean(axis=0):.3f}")
    logger.info(f"Mean volume dice : {volume_df.Dice.mean(axis=0):.3f}")
    logger.info(f"Mean posiitve slice AUC {slice_df[slice_df.label == 1].AUC.mean(axis=0):.3f}")

    # Save Scores and Config
    slice_df.to_csv(os.path.join(out_path, 'slice_predictions.csv'))
    logger.info(f"Slice prediction csv saved at {os.path.join(out_path, 'slice_predictions.csv')}")
    volume_df.to_csv(os.path.join(out_path, 'volume_predictions.csv'))
    logger.info(f"Volume prediction csv saved at {os.path.join(out_path, 'volume_predictions.csv')}")
    cfg.device = str(cfg.device)
    with open(os.path.join(out_path, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    logger.info(f"Config file saved at {os.path.join(out_path, 'config.json')}")

def compute_anomaly(ae_net, im, alpha_low=1.5, alpha_high=3.0, device='cuda'):
    """
    Compute anomaly map and mask.
    ----------
    INPUT
        |---- ae_net (nn.Module) trained AE network.
        |---- im (torch.tensor) input image with dimension (C x H x W).
        |---- alpha_low (float) the lower threshold as fraction of IQR : t_low = q75(err) + alpha_low * IQR(err).
        |---- alpha_high (float) the higher threshold as fraction of IQR : t_low = q75(err) + alpha_high * IQR(err).
        |---- device (str) device to work on.
    OUTPUT
        |---- ad_map (np.array) the reconstruction error map.
        |---- ad_mask (np.array) the thresholded error map = anoamly mask.
    """
    with torch.no_grad():
        im = im.unsqueeze(0).to(device).float()
        rec = ae_net(im)
        ad_map = torch.abs(im - rec).squeeze().cpu().numpy()
        # thresholding
        IQR = (np.quantile(ad_map, 0.75) - np.quantile(ad_map, 0.25))
        t_high = np.quantile(ad_map, 0.75) + alpha_high * IQR
        t_low = np.quantile(ad_map, 0.75) + alpha_low * IQR
        ad_mask = skimage.filters.apply_hysteresis_threshold(ad_map, t_low, t_high)

    return ad_map, ad_mask

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
