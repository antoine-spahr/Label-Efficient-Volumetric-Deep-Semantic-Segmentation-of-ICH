"""
author: Antoine Spahr

date : 30.10.2020

----------

TO DO :
"""
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import nibabel as nib
import numpy as np
import os
import click
import sys
import ast
sys.path.append('../')

from src.utils.plot_utils import imshow_pred
from src.utils.ct_utils import window_ct

@click.command()
@click.argument("vol_fn", type=click.Path(exists=True))
@click.argument("slice", type=str)
@click.option("--pred_fn", type=click.Path(exists=True), default=None, help="The prediction mask to display if provided.")
@click.option("--trgt_fn", type=click.Path(exists=True), default=None, help="The target mask to display if provided.")
@click.option("--pred_color", type=str, default='tomato', help='The color of the prediction mask.')
@click.option("--trgt_color", type=str, default='forestgreen', help='The color of the target mask.')
@click.option("--win", type=str, default='[50, 200]', help="The windowing to apply to the CT-scan as [win_center, win_width]. Default: [50, 200].")
@click.option("--cam_view", type=str, default=None, help="The camera position to be specified at pyvista with shape [(pos1, pos2, pos3), (foc1, foc2, foc3), (viewup1, viewup2, viewup3)]. Default: isotropic view")
@click.option("--isoval", type=float, default=1.0, help="The Isovalue used to generate a mesh form the volume. Default is 1.0.")
@click.option("--vol_alpha", type=float, default=0.3, help="The volume opacity for the 3D rendering. Default 0.3.")
@click.option("--overlap/--no_overlap", default=True, help="Whether the target and prediction are plotteed on the same image or separately. Default True.")
@click.option("--save_fn", type=click.Path(exists=False), default=None, help="Where to save the figure. Default is the current location named as slice1_slice2_slice3.pdf.")
def main(vol_fn, slice, pred_fn, trgt_fn, pred_color, trgt_color, win, cam_view, isoval, vol_alpha, overlap, save_fn):
    """
    Provide an axial, sagital, coronal and 3D view of the Nifti volume at vol_fn. The view are cross sections given by
    the integer in slice ([axial, sagital, coronal]). If a prediction and/or target is provided, the mask is/are overlaid
    on top on the views.
    """
    slice = ast.literal_eval(slice)
    win = ast.literal_eval(win)
    cam_view = ast.literal_eval(cam_view) if cam_view else None
    # load volume
    vol_nii = nib.load(vol_fn)
    aspect_ratio = vol_nii.header['pixdim'][3] / vol_nii.header['pixdim'][2]
    vol = np.rot90(vol_nii.get_fdata(), k=1, axes=(0,1))
    vol = window_ct(vol, win_center=win[0], win_width=win[1], out_range=(0,1))
    # load prediction
    if pred_fn:
        pred_nii = nib.load(pred_fn)
        pred = np.rot90(pred_nii.get_fdata(), k=1, axes=(0,1))

    # load prediction
    if trgt_fn:
        trgt_nii = nib.load(trgt_fn)
        trgt = np.rot90(trgt_nii.get_fdata(), k=1, axes=(0,1))

    # get 3D rendering
    data = pv.wrap(vol)
    data.spacing = vol_nii.header['pixdim'][1:4]
    surface = data.contour([isoval],)
    if pred_fn:
        data_pred = pv.wrap(pred)
        data_pred.spacing = pred_nii.header['pixdim'][1:4]
        surface_pred = data_pred.contour([1],)
    if trgt_fn:
        data_trgt = pv.wrap(trgt)
        data_trgt.spacing = trgt_nii.header['pixdim'][1:4]
        surface_trgt = data_trgt.contour([1],)
    cpos = cam_view

    if not overlap and pred_fn is not None and trgt_fn is not None:
        # make 3D pred rendering
        p = pv.Plotter(off_screen=True, window_size=[512, 512])
        p.background_color = 'black'
        p.add_mesh(surface, opacity=vol_alpha, clim=data.get_data_range(), color='lightgray')
        p.add_mesh(surface_pred, opacity=1, color=pred_color)
        if cpos:
            p.camera_position = cpos
        else:
            p.view_isometric()
        _, vol3Drender_pred = p.show(screenshot=True)
        # make 3D trgt rendering
        p = pv.Plotter(off_screen=True, window_size=[512, 512])
        p.background_color = 'black'
        p.add_mesh(surface, opacity=vol_alpha, clim=data.get_data_range(), color='lightgray')
        p.add_mesh(surface_trgt, opacity=1, color=trgt_color)
        if cpos:
            p.camera_position = cpos
        else:
            p.view_isometric()
        _, vol3Drender_trgt = p.show(screenshot=True)

        # Make figure
        if pred_fn is None:
            pred = np.zeros_like(vol).astype(bool)
        if trgt_fn is None:
            trgt = np.zeros_like(vol).astype(bool)

        fig, axs = plt.subplots(2,4,figsize=(10,5))
        # Axial
        imshow_pred(vol[:,:,slice[0]], pred[:,:,slice[0]].astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect='equal', interpolation='nearest'), legend=False, ax=axs[0,0])
        axs[0,0].set_axis_off()
        axs[0,0].set_title('Axial', color='white')
        imshow_pred(vol[:,:,slice[0]], np.zeros_like(vol)[:,:,slice[0]].astype(bool), trgt[:,:,slice[0]].astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect='equal', interpolation='nearest'), legend=False, ax=axs[1,0])
        axs[1,0].set_axis_off()
        # Sagital
        legend, legend_kwargs = False, None
        if pred_fn is not None or trgt_fn is not None:
            legend = True
            legend_kwargs = dict(loc='upper center', ncol=2, frameon=False, labelcolor='white',
                                 framealpha=0.0, fontsize=10, bbox_to_anchor=(0.5, -0.2),
                                 bbox_transform=axs[1,1].transAxes)
        imshow_pred(np.rot90(vol[:,slice[1],:], axes=(0,1)), np.rot90(pred[:,slice[1],:], axes=(0,1)).astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect=aspect_ratio, interpolation='nearest'), legend=False, ax=axs[0,1])
        axs[0,1].set_axis_off()
        axs[0,1].set_title('Sagital', color='white')
        imshow_pred(np.rot90(vol[:,slice[1],:], axes=(0,1)), np.rot90(np.zeros_like(vol)[:,slice[1],:], axes=(0,1)).astype(bool),
                    np.rot90(trgt[:,slice[1],:], axes=(0,1)).astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect=aspect_ratio, interpolation='nearest'), legend=legend, legend_kwargs=legend_kwargs, ax=axs[1,1])
        axs[1,1].set_axis_off()
        # Coronal
        imshow_pred(np.rot90(vol[slice[2],:,:], axes=(0,1)), np.rot90(pred[slice[2],:,:], axes=(0,1)).astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect=aspect_ratio, interpolation='nearest'), legend=False, ax=axs[0,2])
        axs[0,2].set_axis_off()
        axs[0,2].set_title('Coronal', color='white')
        imshow_pred(np.rot90(vol[slice[2],:,:], axes=(0,1)), np.rot90(np.zeros_like(vol)[slice[2],:,:], axes=(0,1)).astype(bool),
                    np.rot90(trgt[slice[2],:,:], axes=(0,1)).astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect=aspect_ratio, interpolation='nearest'), legend=False, ax=axs[1,2])
        axs[1,2].set_axis_off()
        # 3D rendering
        axs[0,3].imshow(vol3Drender_pred, cmap='gray')
        axs[0,3].set_axis_off()
        axs[0,3].set_title('3D rendering', color='white')
        axs[1,3].imshow(vol3Drender_trgt, cmap='gray')
        axs[1,3].set_axis_off()
        # save figure
        fig.set_facecolor('black')
        fig.tight_layout()
        save_fn = save_fn if save_fn else f'A{slice[0]}_S{slice[1]}_C{slice[2]}.pdf'
        fig.savefig(save_fn, dpi=300, bbox_inches='tight')
    else:
        # make 3D rendering
        p = pv.Plotter(off_screen=True, window_size=[512, 512])
        p.background_color = 'black'
        p.add_mesh(surface, opacity=vol_alpha, clim=data.get_data_range(), color='lightgray')
        if pred_fn:
            p.add_mesh(surface_pred, opacity=1, color=pred_color)
        if trgt_fn:
            p.add_mesh(surface_trgt, opacity=1, color=trgt_color)
        if cpos:
            p.camera_position = cpos
        else:
            p.view_isometric()
        _, vol3Drender = p.show(screenshot=True)

        # Make figure
        if pred_fn is None:
            pred = np.zeros_like(vol).astype(bool)
        if trgt_fn is None:
            trgt = np.zeros_like(vol).astype(bool)

        fig, axs = plt.subplots(1,4,figsize=(10,6))
        # Axial
        imshow_pred(vol[:,:,slice[0]], pred[:,:,slice[0]].astype(bool), trgt[:,:,slice[0]].astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect='equal', interpolation='nearest'), legend=False, ax=axs[0])
        axs[0].set_axis_off()
        axs[0].set_title('Axial', color='white')
        # Sagital
        legend, legend_kwargs = False, None
        if pred_fn is not None or trgt_fn is not None:
            legend = True if trgt_fn is not None and pred_fn is not None else False
            legend_kwargs = dict(loc='upper center', ncol=2, frameon=False, labelcolor='white',
                                 framealpha=0.0, fontsize=10, bbox_to_anchor=(0.5, -0.1),
                                 bbox_transform=axs[1].transAxes)
        imshow_pred(np.rot90(vol[:,slice[1],:], axes=(0,1)), np.rot90(pred[:,slice[1],:], axes=(0,1)).astype(bool),
                    np.rot90(trgt[:,slice[1],:], axes=(0,1)).astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect=aspect_ratio, interpolation='nearest'), legend=legend, legend_kwargs=legend_kwargs, ax=axs[1])
        axs[1].set_axis_off()
        axs[1].set_title('Sagital', color='white')
        # Coronal
        imshow_pred(np.rot90(vol[slice[2],:,:], axes=(0,1)), np.rot90(pred[slice[2],:,:], axes=(0,1)).astype(bool),
                    np.rot90(trgt[slice[2],:,:], axes=(0,1)).astype(bool),
                    im_cmap='gray', pred_color=pred_color, pred_alpha=0.8, target_color=trgt_color, target_alpha=0.8,
                    imshow_kwargs=dict(aspect=aspect_ratio, interpolation='nearest'), legend=False, ax=axs[2])
        axs[2].set_axis_off()
        axs[2].set_title('Coronal', color='white')
        # 3D rendering
        axs[3].imshow(vol3Drender, cmap='gray')
        axs[3].set_axis_off()
        axs[3].set_title('3D rendering', color='white')
        # save figure
        fig.set_facecolor('black')
        fig.tight_layout()
        save_fn = save_fn if save_fn else f'A{slice[0]}_S{slice[1]}_C{slice[2]}.pdf'
        fig.savefig(save_fn, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
