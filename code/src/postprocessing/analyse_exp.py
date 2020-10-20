"""
author: Antoine Spahr

date : 12.10.2020

----------

TO DO :
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import skimage.io as io
import skimage.transform
import json
import glob
import os
import sys
sys.path.append('../../')
from src.utils.plot_utils import metric_barplot, curve_std, imshow_pred
from src.utils.ct_utils import window_ct

def analyse_supervised_exp(exp_folder, data_path, save_fn='results_overview.pdf'):
    """
    Generate a summary figure of the supervised ICH segmentation experiment.
    """
    # utility function
    def h_padcat(arr1, arr2):
        out_len = max([arr1.shape[0], arr2.shape[0]])
        arr1_pad = np.pad(arr1, ((0,out_len-arr1.shape[0]),(0,0)), constant_values=np.nan)
        arr2_pad = np.pad(arr2, ((0,out_len-arr2.shape[0]),(0,0)), constant_values=np.nan)
        return np.concatenate([arr1_pad, arr2_pad], axis=1)
    ########## get data
    # get losses
    loss_list = []
    for train_stat_fn in glob.glob(os.path.join(exp_folder, 'Fold_*/outputs.json')):
        with open(train_stat_fn, 'r') as fn:
            loss_list.append(np.array(json.load(fn)['train']['evolution']))

    all = np.stack(loss_list, axis=2)[:,[1,2,3],:]
    data_evo = [np.concatenate([np.expand_dims(np.arange(1, all.shape[0]+1), axis=1), all[:,i,:]], axis=1) for i in range(all.shape[1])]

    # load performances
    results_df = pd.read_csv(os.path.join(exp_folder, 'all_volume_prediction.csv'))
    results_df = results_df.drop(columns=results_df.columns[0])

    # load performances at slice level
    with open(os.path.join(exp_folder, 'config.json'), 'r') as f:
        cfg = json.load(f)
    df_list = []
    for i in range(cfg['split']['n_fold']):
        df_tmp = pd.read_csv(os.path.join(exp_folder, f'Fold_{i+1}/pred/slice_prediction_scores.csv'))
        df_tmp['Fold'] = i+1
        df_list.append(df_tmp)
    slice_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    slice_df = slice_df.drop(columns=slice_df.columns[0])

    ########## PLOT
    fontsize=12
    n_samp = 10
    fig = plt.figure(figsize=(15,12))
    gs = fig.add_gridspec(nrows=7, ncols=n_samp, height_ratios=[0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], hspace=0.4)
    # Loss Evolution Plot
    ax_evo = fig.add_subplot(gs[:2,:4])
    colors_evo = ['black', 'tomato', 'dodgerblue']
    serie_names_evo = ['Train Loss', 'Dice (all)', 'Dice (ICH)']
    curve_std(data_evo, serie_names_evo, colors=colors_evo, ax=ax_evo, lw=1, CI_alpha=0.05, rep_alpha=0.25, plot_rep=True,
              plot_mean=True, plot_CI=True, legend=True,
              legend_kwargs=dict(loc='upper left', ncol=3, frameon=False, framealpha=0.0,
                                 fontsize=fontsize, bbox_to_anchor=(0.0, -0.3), bbox_transform=ax_evo.transAxes))
    ax_evo.set_xlabel('Epoch [-]', fontsize=fontsize)
    ax_evo.set_ylabel('Dice Loss [-] ; Dice Coeff. [-]', fontsize=fontsize)
    ax_evo.set_title('Training evolution', fontsize=fontsize, fontweight='bold', loc='left')
    ax_evo.set_xlim([1, data_evo[0].shape[0]])
    ax_evo.set_ylim([0,1])
    ax_evo.tick_params(axis='both', labelsize=fontsize)
    ax_evo.spines['top'].set_visible(False)
    ax_evo.spines['right'].set_visible(False)

    # Conf Mat BarPlot
    ax_cm = fig.add_subplot(gs[1,5:7])
    ax_cm_bis = fig.add_subplot(gs[0,5:7])
    # make data
    data_cm = [results_df[['TP', 'TN', 'FP', 'FN']].values, results_df.loc[results_df.ICH == 1, ['TP', 'TN', 'FP', 'FN']].values, results_df.loc[results_df.ICH == 0, ['TP', 'TN', 'FP', 'FN']].values]
    serie_names_cm = ['All', 'ICH only', 'Non-ICH only']
    group_names_cm = ['TP', 'TN', 'FP', 'FN']
    colors_cm = ['tomato', 'dodgerblue', 'cornflowerblue']
    metric_barplot(data_cm, serie_names=serie_names_cm, group_names=group_names_cm, colors=colors_cm,
                   ax=ax_cm, fontsize=fontsize, jitter=True, jitter_color='gray', jitter_alpha=0.25, legend=True,
                   legend_kwargs=dict(loc='upper left', ncol=1, frameon=False, framealpha=0.0,
                                      fontsize=fontsize, bbox_to_anchor=(0.0, -0.4), bbox_transform=ax_cm.transAxes),
                   display_val=False)
    ax_cm.set_ylim([0, (results_df[['TP', 'FP', 'FN']].values.mean(axis=0)+2.2*results_df[['TP', 'FP', 'FN']].values.std(axis=0)).max()])
    ax_cm.set_ylabel('Count [-]', fontsize=fontsize)
    ax_cm.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
    # duplicate plot for long tail bar of True negative
    metric_barplot(data_cm, serie_names=serie_names_cm, group_names=group_names_cm, colors=colors_cm,
                   ax=ax_cm_bis, fontsize=fontsize, jitter=True, jitter_color='gray', jitter_alpha=0.25, legend=False,
                   legend_kwargs=None,
                   display_val=False)
    ax_cm_bis.set_ylim(bottom=results_df['TN'].values.mean()-2.2*results_df['TN'].values.std())
    ax_cm_bis.set_title('Volumetric Classification', fontsize=fontsize, fontweight='bold', loc='left')
    ax_cm_bis.spines['bottom'].set_visible(False)
    ax_cm_bis.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
    ax_cm_bis.set_xticklabels([])
    ax_cm_bis.set_xticks([])
    d=.015
    ax_cm.plot((-d,+d),(1-d,1+d),transform=ax_cm.transAxes, color='k', clip_on=False)
    ax_cm_bis.plot((-d,+d),(-d,+d),transform=ax_cm_bis.transAxes, color='k', clip_on=False)

    # Dice barplot
    ax_dice = fig.add_subplot(gs[0:2,8:])
    data_dice = [h_padcat(h_padcat(results_df[['Dice']].values, results_df.loc[results_df.ICH == 1, ['Dice']].values), results_df.loc[results_df.ICH == 0, ['Dice']].values),
                 h_padcat(h_padcat(slice_df[['Dice']].values, slice_df.loc[slice_df.ICH == 1, ['Dice']].values), slice_df.loc[slice_df.ICH == 0, ['Dice']].values)]

    group_names_dice = ['All', 'ICH only', 'Non-ICH only']
    serie_names_dice = ['Volume Dice', 'Slice Dice']
    colors_dice = ['xkcd:pumpkin orange', 'xkcd:peach']#, 'cornflowerblue']
    metric_barplot(data_dice, serie_names=serie_names_dice, group_names=group_names_dice, colors=colors_dice,
                   ax=ax_dice, fontsize=fontsize, jitter=True, jitter_color='gray', jitter_alpha=0.25, legend=True,
                   legend_kwargs=dict(loc='upper left', ncol=1, frameon=False, framealpha=0.0,
                                      fontsize=fontsize, bbox_to_anchor=(0.0, -0.3), bbox_transform=ax_dice.transAxes),
                   display_val=False, display_format='.2%', display_pos='top', tick_angle=20)
    ax_dice.set_ylim([0,1])
    ax_dice.set_ylabel('Dice [-]', fontsize=12)
    ax_dice.set_title('Dice Coefficients', fontsize=fontsize, fontweight='bold', loc='left')

    # Pred sample Highest Dice with ICH
    for k, (asc, is_ICH) in enumerate(zip([False, True, False, True], [1, 1, 0, 0])):
        # get n_samp highest/lowest dice for ICH
        samp_df = slice_df[slice_df.ICH == is_ICH].sort_values(by='Dice', ascending=asc).iloc[:n_samp,:]
        axs = []
        for i, samp_row in enumerate(samp_df.iterrows()):
            samp_row = samp_row[1]
            ax_i = fig.add_subplot(gs[k+3, i])
            axs.append(ax_i)
            # load image and window it
            slice_im = io.imread(os.path.join(data_path, f'Patient_CT/{samp_row.PatientID:03}/{samp_row.Slice}.tif'))
            slice_im = window_ct(slice_im, win_center=cfg['data']['win_center'], win_width=cfg['data']['win_width'], out_range=(0,1))
            # load truth mask
            if is_ICH == 1:
                slice_trg = io.imread(os.path.join(data_path, f'Patient_CT/{samp_row.PatientID:03}/{samp_row.Slice}_ICH_Seg.bmp'))
            else:
                slice_trg = np.zeros_like(slice_im)
            slice_trg = slice_trg.astype('bool')
            # load prediction
            slice_pred = io.imread(os.path.join(exp_folder, f'Fold_{samp_row.Fold}/pred/{samp_row.PatientID}/{samp_row.Slice}.bmp'))
            slice_pred = skimage.transform.resize(slice_pred, slice_trg.shape, order=0)
            slice_pred = slice_pred.astype('bool')
            # plot all
            imshow_pred(slice_im, slice_pred, target=slice_trg, ax=ax_i, im_cmap='gray', pred_color='xkcd:vermillion', target_color='forestgreen',
                        pred_alpha=0.7, target_alpha=1, legend=False, legend_kwargs=None)
            ax_i.text(0, 1.1, f' {samp_row.PatientID:03} / {samp_row.Slice:02}', fontsize=10, fontweight='bold', color='white', ha='left', va='top', transform=ax_i.transAxes)

        pos = axs[0].get_position()
        pos3 = axs[1].get_position()
        pos4 = axs[-1].get_position()
        fig.patches.extend([plt.Rectangle((pos.x0-0.5*pos.width, pos4.y0-0.1*pos.height),
                                          0.5*pos.width + (pos3.x0-pos.x0)*len(axs),
                                          1.3*pos.height,
                                          fc='black', ec='black', alpha=1, zorder=-1,
                                          transform=fig.transFigure, figure=fig)])
        axs[0].text(-0.25, 0.5, f"{'Low' if asc else 'High'}est Dice\n({'non-' if is_ICH == 0 else ''}ICH)",
                 fontsize=10, fontweight='bold', ha='center', va='center', rotation=90, color='lightgray', transform=axs[0].transAxes)


    handles = [matplotlib.patches.Patch(facecolor='forestgreen', alpha=1),
               matplotlib.patches.Patch(facecolor='xkcd:vermillion', alpha=0.7)]
    labels = ['Ground Truth', 'Prediction']
    axs[n_samp//2].legend(handles, labels, loc='upper center', ncol=2, frameon=False, framealpha=0.0,
              fontsize=12, bbox_to_anchor=(0.5, 0.0), bbox_transform=axs[n_samp//2].transAxes)

    # Save figure
    fig.savefig(save_fn, dpi=300, bbox_inches='tight')


#analyse_supervised_exp('../../../outputs/UNet2D_Debug', '../../../data/publicSegICH2D/', 'test.pdf')
