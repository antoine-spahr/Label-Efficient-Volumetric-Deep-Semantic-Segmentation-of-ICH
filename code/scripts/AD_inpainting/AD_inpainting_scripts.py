"""
author: Antoine Spahr

date : 07.12.2020

----------

To Do:
    -
"""

import os
import sys
sys.path.append('../../')
import logging
import click
import json
import random
import pandas as pd
import numpy as np
from skimage import img_as_ubyte
import skimage.io as io
import torch
import torch.nn as nn
import torch.cuda
from sklearn.metrics import confusion_matrix

from src.dataset.datasets import public_SegICH_Dataset2D
from src.models.optim.InpaintAnomalyDetector import InpaintAnomalyDetector, robust_anomaly_detect
from src.models.networks.InpaintingNetwork import SAGatedGenerator
import src.models.networks.ResNet as rn
from src.utils.python_utils import AttrDict

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """
    Segmente ICH using the anomaly inpainting approach on a whole dataset and compute slice/volume dice.
    """
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

    # set device
    cfg.device = torch.device(cfg.device) if cfg.device else get_available_device()

    # initialize logger
    logger = initialize_logger(os.path.join(out_path, 'log.txt'))
    logger.info(f"Experiment : {cfg.exp_name}")

    # get Dataset
    data_info_df = pd.read_csv(os.path.join(cfg.path.data, 'ct_info.csv'), index_col=0)
    #data_info_df = data_info_df[sum([data_info_df.CT_fn.str.contains(s) for s in ['49/16', '51/39', '71/15', '71/22', '75/22']]) > 0]
    dataset = public_SegICH_Dataset2D(data_info_df, cfg.path.data,
                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.data.augmentation.items()],
                    output_size=cfg.data.size, window=(cfg.data.win_center, cfg.data.win_width))

    # load inpainting model
    cfg_inpaint = AttrDict.from_json_path(cfg.inpainter_cfg_path)
    cfg_inpaint.net.gen.return_coarse = False
    inpaint_net = SAGatedGenerator(**cfg_inpaint.net.gen)
    loaded_state_dict = torch.load(cfg.inpainter_model_path, map_location=cfg.device)
    inpaint_net.load_state_dict(loaded_state_dict)
    inpaint_net = inpaint_net.to(cfg.device) # inpainter not in eval mode beacuse batch norm layers are not stabilized (because GAN optimization)
    logger.info(f"Inpainter model succesfully loaded from {cfg.inpainter_model_path}")

    # make AD inpainter Module
    ad_inpainter = InpaintAnomalyDetector(inpaint_net, device=cfg.device, **cfg.model_param)

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
            logger.info(f"ICH detected. Compute anomaly mask through inpainting.")
            # Detect anomalies using the robuste approach
            ad_mask, ano_map, intermediate_masks = robust_anomaly_detect(image, ad_inpainter, save_dir=None, verbose=True, return_intermediate=True, **cfg.robust_param)
            # save ad_mask
            ad_mask_fn = f"{id}/{slice}_anomalies.bmp"
            save_path = os.path.join(out_path, 'pred/', ad_mask_fn)
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            io.imsave(save_path, img_as_ubyte(ad_mask), check_contrast=False)
            # save anomaly map
            ad_map_fn = f"{id}/{slice}_map_anomalies.png"
            save_path_map = os.path.join(out_path, 'pred/', ad_map_fn)
            io.imsave(save_path_map, img_as_ubyte(ano_map), check_contrast=False)
            # save intermediate mask
            for j, m in enumerate(intermediate_masks):
                if not os.path.isdir(os.path.join(out_path, f"pred/{id}/intermediate_masks/")):
                    os.makedirs(os.path.join(out_path, f"pred/{id}/intermediate_masks/"), exist_ok=True)
                io.imsave(os.path.join(out_path, f"pred/{id}/intermediate_masks/{slice}_anomalies_{j+1}.bmp"), img_as_ubyte(m), check_contrast=False)
        else:
            logger.info(f"No ICH detected. Set the anomaly mask to zeros.")
            ad_mask = np.zeros_like(target[0].numpy())
            ad_mask_fn, ad_map_fn = 'None', 'None'

        # compute confusion matrix with target ICH mask
        tn, fp, fn, tp = confusion_matrix(target[0].numpy().ravel(), ad_mask.ravel(), labels=[0,1]).ravel()
        # append to all_pred list
        all_pred.append({'id': id.item(), 'slice': slice.item(), 'label':target.max().item(), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'ad_mask_fn': ad_mask_fn, 'ad_map_fn': ad_map_fn})

    # make a dataframe of all predictions
    slice_df = pd.DataFrame(all_pred)
    volume_df = slice_df[['id', 'label', 'TP', 'TN', 'FP', 'FN']].groupby('id').agg({'label':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})

    # Compute Dice and Volume Dice
    slice_df['Dice'] = (2*slice_df.TP + 1) / (2*slice_df.TP + slice_df.FP + slice_df.FN + 1)
    volume_df['Dice'] = (2*volume_df.TP + 1) / (2*volume_df.TP + volume_df.FP + volume_df.FN + 1)
    logger.info(f"Mean slice dice : {slice_df.Dice.mean(axis=0):.3f}")
    logger.info(f"Mean volume dice : {volume_df.Dice.mean(axis=0):.3f}")

    # Save Scores and Config
    slice_df.to_csv(os.path.join(out_path, 'slice_predictions.csv'))
    logger.info(f"Slice prediction csv saved at {os.path.join(out_path, 'slice_predictions.csv')}")
    volume_df.to_csv(os.path.join(out_path, 'volume_predictions.csv'))
    logger.info(f"Volume prediction csv saved at {os.path.join(out_path, 'volume_predictions.csv')}")
    cfg.device = str(cfg.device)
    with open(os.path.join(out_path, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    logger.info(f"Config file saved at {os.path.join(out_path, 'config.json')}")

def get_available_device():
    """
    Return available device.
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
