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
import torch.nn as nn
import numpy as np
import pandas as pd
import skimage.io as io
from skimage import img_as_ubyte
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.dataset.datasets import public_SegICH_Dataset2D
from src.models.networks.FCDD_net import FCDD_CNN_VGG
from src.models.optim.FCDD import FCDD
import src.models.networks.ResNet as rn
from src.utils.python_utils import AttrDict
from src.utils.tensor_utils import batch_binary_confusion_matrix
from src.utils.print_utils import print_progessbar

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
def main(config_path):
    """  """
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

    #-------------------------------------------
    #       Make Dataset
    #-------------------------------------------

    data_info_df = pd.read_csv(os.path.join(cfg.path.data, 'ct_info.csv'), index_col=0)
    dataset = public_SegICH_Dataset2D(data_info_df, cfg.path.data,
                    augmentation_transform=[getattr(tf, tf_name)(**tf_kwargs) for tf_name, tf_kwargs in cfg.data.augmentation.items()],
                    output_size=cfg.data.size, window=(cfg.data.win_center, cfg.data.win_width))

    #-------------------------------------------
    #       Load FCDD Model
    #-------------------------------------------

    cfg_fcdd = AttrDict.from_json_path(cfg.fcdd_cfg_path)
    fcdd_net = FCDD_CNN_VGG(in_shape=(cfg_fcdd.net.in_channels, 256, 256), bias=cfg_fcdd.net.bias)
    loaded_state_dict = torch.load(cfg.fcdd_model_path, map_location=cfg.device)
    fcdd_net.load_state_dict(loaded_state_dict)
    fcdd_net = fcdd_net.to(cfg.device).eval()
    logger.info(f"FCDD model succesfully loaded from {cfg.fcdd_model_path}")

    # make FCDD object
    fcdd = FCDD(fcdd_net, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                device=cfg.device, print_progress=cfg.print_progress)

    #-------------------------------------------
    #       Load Classifier Model
    #-------------------------------------------

    # Load Classifier
    if cfg.classifier_model_path is not None:
        cfg_classifier = AttrDict.from_json_path(os.path.join(cfg.classifier_model_path, 'config.json'))
        classifier = getattr(rn, cfg_classifier.net.resnet)(num_classes=cfg_classifier.net.num_classes, input_channels=cfg_classifier.net.input_channels)
        classifier_state_dict = torch.load(os.path.join(cfg.classifier_model_path, 'resnet_state_dict.pt'), map_location=cfg.device)
        classifier.load_state_dict(classifier_state_dict)
        classifier = classifier.to(cfg.device)
        classifier.eval()
        logger.info(f"ResNet classifier model succesfully loaded from {os.path.join(cfg.classifier_model_path, 'resnet_state_dict.pt')}")

    #-------------------------------------------
    #       Generate Heat-Map for each slice
    #-------------------------------------------

    with torch.no_grad():
        # make loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                             shuffle=False, worker_init_fn=lambda _: np.random.seed())
        fcdd_net.eval()

        min_val, max_val = fcdd.get_min_max(loader, **cfg.heatmap_param)

        # computing and saving heatmaps
        out = dict(id=[], slice=[], label=[], ad_map_fn=[], ad_mask_fn=[],
                   TP=[], TN=[], FP=[], FN=[], AUC=[], classifier_pred=[])
        for b, data in enumerate(loader):
            im, mask, id, slice = data
            im = im.to(cfg.device).float()
            mask = mask.to(cfg.device).float()

            # get heatmap
            heatmap = fcdd.generate_heatmap(im, reception=cfg.heatmap_param.reception, std=cfg.heatmap_param.std,
                                            cpu=cfg.heatmap_param.cpu)
            # scaling
            heatmap = ((heatmap - min_val) / (max_val - min_val)).clamp(0,1)

            # Threshold
            ad_mask = torch.where(heatmap >= cfg.heatmap_threshold, torch.ones_like(heatmap, device=heatmap.device),
                                                                    torch.zeros_like(heatmap, device=heatmap.device))

            # Compute CM
            tn, fp, fn, tp  = batch_binary_confusion_matrix(ad_mask, mask.to(heatmap.device))

            # Save heatmaps/mask
            map_fn, mask_fn = [], []
            for i in range(im.shape[0]):
                # Save AD Map
                ad_map_fn = f"{id[i]}/{slice[i]}_map_anomalies.png"
                save_path = os.path.join(out_path, 'pred/', ad_map_fn)
                if not os.path.isdir(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                ad_map = heatmap[i].squeeze().cpu().numpy()
                io.imsave(save_path, img_as_ubyte(ad_map), check_contrast=False)
                # save ad_mask
                ad_mask_fn = f"{id[i]}/{slice[i]}_anomalies.bmp"
                save_path = os.path.join(out_path, 'pred/', ad_mask_fn)
                io.imsave(save_path, img_as_ubyte(ad_mask[i].squeeze().cpu().numpy()), check_contrast=False)

                map_fn.append(ad_map_fn)
                mask_fn.append(ad_mask_fn)

            # apply classifier ResNet-18
            if cfg.classifier_model_path is not None:
                pred_score = nn.functional.softmax(classifier(im), dim=1)[:,1] # take columns of softmax of positive class as score
                clss_pred = torch.where(pred_score >= cfg.classification_threshold, torch.ones_like(pred_score, device=pred_score.device),
                                                                                    torch.zeros_like(pred_score, device=pred_score.device))
            else:
                clss_pred = [None]*im.shape[0]

            # Save Values
            out['id'] += id.cpu().tolist()
            out['slice'] += slice.cpu().tolist()
            out['label'] += mask.reshape(mask.shape[0], -1).max(dim=1)[0].cpu().tolist()
            out['ad_map_fn'] += map_fn
            out['ad_mask_fn'] += mask_fn
            out['TN'] += tn.cpu().tolist()
            out['FP'] += fp.cpu().tolist()
            out['FN'] += fn.cpu().tolist()
            out['TP'] += tp.cpu().tolist()
            out['AUC'] += [roc_auc_score(mask[i].cpu().numpy().ravel(), heatmap[i].cpu().numpy().ravel()) if torch.any(mask[i]>0) else 'None' for i in range(im.shape[0])]
            out['classifier_pred'] += clss_pred.cpu().tolist()

            if cfg.print_progress:
                print_progessbar(b, len(loader), Name='Heatmap Generation Batch', Size=100, erase=True)

    # make df and save as csv
    slice_df = pd.DataFrame(out)
    volume_df = slice_df[['id', 'label', 'TP', 'TN', 'FP', 'FN']].groupby('id').agg({'label':'max', 'TP':'sum', 'TN':'sum', 'FP':'sum', 'FN':'sum'})

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
