"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :

"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skimage.io as io
import skimage
import imageio
import warnings

def draw_cruved_rect(x1, x2, h1, h2, y1, y2, ax, fc='lightgray', ec='gray', lw=1, alpha=0.3):
    """
    Draw a curved rectangle between (x1, y1), (x2, y2), (x2, y2 + h2), (x1, y1 + h1).
    ----------
    INPUT
        |---- x1 (float) lower left corner x position.
        |---- x2 (float) lower right corner x position.
        |---- h1 (float) left height.
        |---- h2 (float) right height.
        |---- y1 (float) lower left corner y position.
        |---- x2 (float) lower right corner y position.
        |---- fc (matplotlib color) facecolor of the rectangle.
        |---- ec (matplotlib color) edgecolor of the rectangle.
        |---- lw (float) the linewidth of the rectangle's edge.
        |---- alpha (flaot) the transparency of the rectangle.
    OUTPUT
        |---- None
    """
    if h1 != 0 or h2 != 0:
        x05 = (x2+x1)/2
        v = np.array([[x1, y1],
                      [x05, y1],
                      [x05, y2],
                      [x2, y2],
                      [x2, y2 + h2],
                      [x05, y2 + h2],
                      [x05, y1 + h1],
                      [x1, y1 + h1]])

        p = matplotlib.path.Path(v, codes = [1,4,4,4,2,4,4,4], closed=True)
        ax.add_patch(matplotlib.patches.PathPatch(p, lw=lw, ec=ec, fc=fc, alpha=alpha, zorder=-1))

def pred2GIF(img_list, mask_list, out_filename, fps=2, mask_color='xkcd:vermillion'):
    """
    Convert a list of images and mask into a GIF where the mask is overlaid to the image.
    ----------
    INPUT
        |---- img_list (list of 2D np.array) the list of images to convert in GIF.
        |---- mask_list (list of 2D boolean np.array) the list of mask to overlaid on the GIF.
        |---- out_filename (str) the output GIF file name.
        |---- fps (int) the FPS speed of the GIF.
        |---- mask_color (matplotlib color) the color of the mask.
    OUTPUT
        |---- None
    """
    output_list = []
    # draw images and save them in a list
    for img, mask in zip(img_list, mask_list):
        #im = plt.imshow(f(x, y), animated=True)
        #output_list.append([im])
        xpixels = img.shape[1]
        ypixels = img.shape[0]
        dpi = 72
        xinch = xpixels / dpi
        yinch = ypixels / dpi

        fig, ax = plt.subplots(1, 1, figsize=(xinch,yinch), dpi=dpi)
        canvas = matplotlib.backends.backend_agg.FigureCanvas(fig)

        fig.set_facecolor('black')
        # plot frame
        ax.imshow(img, cmap='gray', aspect='equal')
        mask = skimage.img_as_bool(mask)
        if np.any(mask):
            ax.imshow(np.ma.masked_where(mask == False, mask), cmap=matplotlib.colors.ListedColormap([mask_color, 'white']), alpha=.8, aspect='equal')
        ax.set_axis_off()

        fig.tight_layout(pad=0)
        ax.margins(0)
        canvas.draw()
        # get figure as np.array
        out_img = np.array(canvas.renderer.buffer_rgba(), dtype=np.uint8)
        plt.close()

        output_list.append(out_img)

    # make the gif
    imageio.mimsave(out_filename, output_list, fps=fps)

def curve_std(data, serie_names, colors=None, ax=None, lw=1, CI_alpha=0.25, rep_alpha=0.5,
              plot_rep=False, plot_mean=True, plot_CI=True, legend=True, legend_kwargs=None, fontsize=12):
    """
    Plot the curve evolutions for all the replicate in list of results, as well as
    the 95% confidence interval (CI) of the mean curve evolution.
    ----------
    INPUT
        |---- data (list of 2D np.array) list of experiemnt results where the
        |           curve to plot is stored. First columns is the x-axis values
        |           and they each following columns is a replicate of the y-axis.
        |           The len of the list represents the different curves to plot.
        |---- serie_names (list of str) the name for each array in data. Will
        |           appear in the legend.
        |---- colors (list of str) the color to use for each curves.
        |---- ax (matplotlib.Axes) the axes where to plot.
        |---- lw (int) the line width.
        |---- CI_alpha (float) the transparency of the CI.
        |---- rep_alpha (float) the transparancy of replicate lines.
        |---- plot_rep (bool) whether to plot all the replicate lines.
        |---- plot_mean (bool) whether to plot the mean curve evolution.
        |---- plot_CI (bool) whether to plot the confidence interval.
        |---- legend (bool) whether to add the legend.
        |---- legend_kwargs (dict) the keyword arguments for the legend.
        |---- fontsize (int) the fontsize to use.
    OUTPUT
        |---- None
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    # set color scheme
    n = len(data)
    if colors is None: colors = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()), size=n)

    # plot each curve
    for curve_data, color, name in zip(data, colors, serie_names):
        x_data = curve_data[:,0]
        y_data = curve_data[:,1:]
        # plot curve
        if plot_mean:
            ax.plot(x_data, y_data.mean(axis=1), color=color, lw=lw, label=name)
        if plot_rep:
            for rep_i in range(y_data.shape[1]):
                lab = name if rep_i == y_data.shape[1] - 1 else None
                ax.plot(x_data, y_data[:,rep_i], color=color, lw=lw, label=lab, alpha=rep_alpha)
        # plot CI
        if plot_CI:
            ax.fill_between(x_data, y_data.mean(axis=1) + 1.96*y_data.std(axis=1),
                                   y_data.mean(axis=1) - 1.96*y_data.std(axis=1),
                                   color=color, alpha=0.25, linewidth=0, label=name+' CI')
        # add legend
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if legend_kwargs is None:
                ax.legend(handles, labels, loc='upper left', ncol=1, frameon=False, framealpha=0.0,
                          fontsize=fontsize, bbox_to_anchor=(1, 1.1), bbox_transform=ax.transAxes)
            elif isinstance(legend_kwargs, dict):
                ax.legend(handles, labels, **legend_kwargs)

def metric_barplot(metrics_scores, serie_names, group_names, colors=None, w=None,
                   ax=None, fontsize=12, jitter=False, jitter_color=None, jitter_alpha=0.5, gap=None,
                   tick_angle=0, legend=True, legend_kwargs=None, display_val=False, display_format='.2%',
                   display_pos='bottom'):
    """
    Plot a grouped barplot from the passed array, for various metrics.
    ----------
    INPUT
        |---- metric_scores (list of 2D np.array) the data to plot each element
        |           of the list is a np.array (N_replicats x N_group). The lenght
        |           of the lists gives the number of series plotted.
        |---- series_name (list of str) the names for each series (to appear in
        |           the legend).
        |---- group_names (list of str) the names of the groups (the x-ticks labels).
        |---- colors (list of str) the colors for each series. If None, colors
        |           are randomly picked.
        |---- w (float) the bar width. If None, w is automoatically computed.
        |---- ax (matplotlib Axes) the axes where to plot.
        |---- fontsize (int) the fontsize to use for the texts.
    OUTPUT
        |---- None
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    n = len(metrics_scores)
    if colors is None: colors = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()), size=n)
    if jitter_color is None: jitter_color = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()))

    offsets = list(np.arange(-(n-1),(n-1)+2, 2))
    if w is None: w = 0.9/n
    ind = np.arange(metrics_scores[0].shape[1]) # number of different groups
    if gap:
        ind = np.where(ind + 1 > gap, ind + 0.5, ind)

    for metric, offset, name, color in zip(metrics_scores, offsets, serie_names, colors):
        means = np.nanmean(metric, axis=0)
        stds = np.nanstd(metric, axis=0)
        ax.bar(ind + offset*w/2, means, width=w, yerr=1.96*stds,
               fc=color, ec='black', lw=1, label=name)

        if jitter:
            for j, x in enumerate(ind):
                ax.scatter(np.random.normal(x + offset*w/2, 0.02, metric.shape[0]),
                           metric[:,j], c=jitter_color, marker='o', s=20, lw=0, alpha=jitter_alpha, zorder=4)

        if display_val:
            for i, x in enumerate(ind):
                if display_pos == 'bottom':
                    ax.text(x + offset*w/2, means[i]-1.96*stds[i], ('{0:'+display_format+'}').format(means[i]),
                            fontsize=fontsize, ha='center', va='top', rotation=90)
                elif display_pos == 'top':
                    ax.text(x + offset*w/2, means[i]+1.96*stds[i], ('{0:'+display_format+'}').format(means[i]),
                            fontsize=fontsize, ha='center', va='bottom', rotation=90)
                else:
                    raise ValueError('Unsupported display_pos parameter. Mus be top or bottom.')

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if jitter:
            handles += [matplotlib.lines.Line2D((0,0),(0,0), lw=0, alpha=jitter_alpha, marker='o',
                        markerfacecolor=jitter_color, markeredgecolor=jitter_color, markersize=7)]
            labels += ['Measures']

        if legend_kwargs is None:
            ax.legend(handles, labels, loc='upper right', ncol=1, frameon=False, framealpha=0.0,
                      fontsize=fontsize, bbox_to_anchor=(1, 1.1), bbox_transform=ax.transAxes)
        elif isinstance(legend_kwargs, dict):
            ax.legend(handles, labels, **legend_kwargs)

    ax.set_xticks(ind)
    ha = 'center' if tick_angle == 0 else 'right'
    ax.set_xticklabels(group_names, rotation=tick_angle, ha=ha)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('', fontsize=fontsize)
    #ax.set_ylim([0,1])

def add_stat_significance(pairs, data, serie_names, group_names, w=None, mode='adjusted',
    h_offset=0.06, h_gap=0.02, fontsize=12, stat_test='ttest', stat_test_param=dict(equal_var=False, nan_policy='omit'),
    stat_display='symbol', avoid_cross=True, link_color='lightgray', text_color='gray', ax=None, text_rot=0):
    """
    Compute and display significance comparison between two bars of a metric barplot.
    ----------
    INPUT
        |----
    OUTPUT
        |----
    """
    # find axes
    ax = plt.gca() if ax is None else ax
    # get current heights and other inputs
    if mode == 'adjusted':
        heights = [np.nanmean(arr, axis=0)+1.96*np.nanstd(arr, axis=0) for arr in data]
    elif mode == 'flat':
        h = [np.nanmean(arr, axis=0)+1.96*np.nanstd(arr, axis=0) for arr in data]
        max_h = np.concatenate(h, axis=0).max()
        heights = [np.ones(arr.shape[1])*max_h for arr in data]
    elif isinstance(mode, int) or isinstance(mode, float):
        heights = [np.ones(arr.shape[1])*mode for arr in data]
    else:
        raise ValueError(f'Height mode {mode} not supported. Please use flat or adjusted or a digit.')

    # get x data by series and group
    ind = np.arange(data[0].shape[1])
    n = len(data)
    if w is None: w = 0.9/n
    offsets = list(np.arange(-(n-1),(n-1)+2, 2))
    posx = [ind + offset*w/2 for offset in offsets]

    for p in pairs:
        # get index of pair
        s1, g1 = serie_names.index(p[0][1]), group_names.index(p[0][0])
        s2, g2 = serie_names.index(p[1][1]), group_names.index(p[1][0])
        # get data
        data1 = data[s1][:,g1]
        data2 = data[s2][:,g2]
        h1 = heights[s1][g1]
        h2 = heights[s2][g2]

        # get max height between the two bars
        if posx[s1][g1] < posx[s2][g2]:
            gl, sl, hl = g1, s1, h1
            gh, sh, hh = g2, s2, h2
        else:
            gl, sl, hl = g2, s2, h2
            gh, sh, hh = g1, s1, h1

        low = gl * len(serie_names) + sl
        high = gh * len(serie_names) + sh
        heights_arr = np.array(heights).transpose().ravel()[low:high+1]
        x = [posx[sl][gl]]*2 + [posx[sh][gh]]*2
        y = [hl + h_gap, heights_arr.max() + h_offset, heights_arr.max() + h_offset, hh + h_gap]

        # perform test
        if stat_test == 'ttest':
            pval = ttest_ind(data1, data2, **stat_test_param)[1]
        else:
            raise ValueError(f'Usupported statisical test {stat_test}. Supported: ttest.')

        # get string symbol : larger that 10% --> NS ; between 5 and 10% --> . ; between 1 and 5% --> * ; between 0.1 and 1% --> ** ; below 0.1% --> ***
        if stat_display == 'symbol':
            if pval > 0.1:
                significance = 'ns'
            elif pval > 0.05:
                significance = '.'
            elif pval > 0.01:
                significance = '*'
            elif pval > 0.001:
                significance = '**'
            else:
                significance = '***'
        elif stat_display == 'value':
            significance = f'{pval:.2g}'
        else:
            raise ValueError(f'Usupported statisical display type {stat_display}. Supported: symbol or value.')

        # update heights
        if avoid_cross:
            # update all columns between data1 and data2 to avoid any crossing
            if gl != gh:
                for s in range(sl, len(serie_names)):
                    heights[s][gl] = heights_arr.max() + h_offset
                for g in range(gl+1, gh):
                    for s in range(len(serie_names)):
                        heights[s][g] = heights_arr.max() + h_offset
                for s in range(sh+1):
                    heights[s][gh] = heights_arr.max() + h_offset
            else:
                for s in range(sl, sh+1):
                    heights[s][gl] = heights_arr.max() + h_offset
        else:
            # update only data1 and data2 alowing crossing
            heights[s1][g1] = heights_arr.max() + h_offset
            heights[s2][g2] = heights_arr.max() + h_offset

        # plot
        ax.plot(x, y, lw=2, color=link_color)
        ax.text((x[0]+x[-1])/2, y[1], significance, ha='center', va='bottom', fontsize=fontsize, color=text_color, rotation=text_rot)
        ax.set_ylim([0, heights_arr.max()])

def imshow_pred(im, pred, target=None, ax=None, im_cmap=None, pred_color='tomato', target_color='forestgreen',
                pred_alpha=0.8, target_alpha=0.8, legend=True, legend_kwargs=None):
    """

    """
    if pred.dtype != 'bool':
        pred = pred.astype('bool')
        warnings.warn('Casting the prediction to boolean. Pass a boolean array to silence the warning.')
    if target is not None:
        if target.dtype != 'bool':
            target = target.astype('bool')
            warnings.warn('Casting the target to boolean. Pass a boolean array to silence the warning.')
    # find axes
    ax = plt.gca() if ax is None else ax
    ax.set_axis_off()
    # plot image
    ax.imshow(im, cmap=im_cmap)
    # show target
    if target is not None:
        if np.any(target):
            ax.imshow(np.ma.masked_where(target == False, target), cmap=matplotlib.colors.ListedColormap([target_color]),
                      alpha=target_alpha, aspect='equal')
    # show pred
    if np.any(pred):
        ax.imshow(np.ma.masked_where(pred == False, pred), cmap=matplotlib.colors.ListedColormap([pred_color]),
                  alpha=pred_alpha, aspect='equal')

    if legend:
        if not legend_kwargs:
            legend_kwargs = dict(loc='upper center', ncol=2, frameon=False, framealpha=0.0, fontsize=12, bbox_to_anchor=(0.5, 0.0), bbox_transform=ax.transAxes)
        handles = [matplotlib.patches.Patch(facecolor=target_color, alpha=target_alpha),
                   matplotlib.patches.Patch(facecolor=pred_color, alpha=pred_alpha)]
        labels = ['Ground Truth', 'Prediction']
        ax.legend(handles, labels, **legend_kwargs)
