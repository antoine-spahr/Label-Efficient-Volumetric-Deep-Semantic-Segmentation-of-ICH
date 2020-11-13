"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :

"""
import matplotlib.pyplot as plt
import matplotlib
import pyvista as pv
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
                pred_alpha=0.8, target_alpha=0.8, imshow_kwargs=dict(aspect='equal'), legend=True, legend_kwargs=None):
    """
    Enables to plot a image (as plt.imshow) with a prediction and/or a ground truth mask over it.
    ----------
    INPUT
        |---- im (np.array) the image to display (is input to plt.imshow()).
        |---- pred (boolean np.array) prediction mask.
        |---- target (boolean np.array) the ground truth mask
        |---- ax (plt.Axes) axes on which to plot.
        |---- im_cmap (string) a valid matplotlib colormap to color im.
        |---- pred_color (str) color to be used for the prediction mask.
        |---- target_color (str) color to be used for the ground truth mask.
        |---- pred_alpha (float) color opacity to be used for the prediction mask.
        |---- target_alpha (float) color opacity to be used for the ground truth mask.
        |---- imshow_kwargs (dict) keyword argument to be passed to imshow. Mostly relevant for controling the ascpect ratio.
        |---- legend (bool) whether to add a legend.
        |---- legend_kwargs (dict) keywords arguments for the legend.
    OUTPUT
        |---- None
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
    im_cmap = 'gray' if im_cmap is None else im_cmap
    ax.imshow(im, cmap=im_cmap, **imshow_kwargs)
    # show target
    if target is not None:
        if np.any(target):
            ax.imshow(np.ma.masked_where(target == False, target), cmap=matplotlib.colors.ListedColormap([target_color]),
                      alpha=target_alpha, **imshow_kwargs)
    # show pred
    if np.any(pred):
        ax.imshow(np.ma.masked_where(pred == False, pred), cmap=matplotlib.colors.ListedColormap([pred_color]),
                  alpha=pred_alpha, **imshow_kwargs)

    if legend:
        if not legend_kwargs:
            legend_kwargs = dict(loc='upper center', ncol=2, frameon=False, framealpha=0.0, fontsize=12, bbox_to_anchor=(0.5, 0.0), bbox_transform=ax.transAxes)
        handles = [matplotlib.patches.Patch(facecolor=target_color, alpha=target_alpha),
                   matplotlib.patches.Patch(facecolor=pred_color, alpha=pred_alpha)]
        labels = ['Ground Truth', 'Prediction']
        ax.legend(handles, labels, **legend_kwargs)

def plot_tsne(embed, color_code, colors=None, ax=None, scatter_kwargs=dict(s=10, marker='o', alpha=0.8),
              legend=False, code_name=None, legend_kwargs=None):
    """
    Plot a 2D embedding (ideally from t-SNE reduction).
    ----------
    INPUT
        |---- embed (2D np.array) the embedding to plot (sample x 2).
        |---- color_code (1D array) the class belonging of each samples in embed.
        |---- colors (list of str) the color to use for each class (must have the same length as the number of class).
        |               If None colors are picked randomly.
        |---- ax (plt.axes) the ax on which to plot.
        |---- scatter_kwargs (dict) the keyword arguments for the matplotlib scatter function.
        |---- legend (bool) whether to add a legend
        |---- code_name (dict) the name associated with each class.
        |---- legend_kwargs (dict) the keyword argument to use for the legend.
    OUTPUT
        |---- None
    """
    ax = plt.gca() if ax is None else ax
    n = len(np.unique(color_code))
    if colors is None: colors = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()), size=n)

    labels, handles = [], []
    for grp, color in zip(np.unique(color_code), colors):
        labels.append(code_name[grp] if code_name is not None else grp)
        handles.append(matplotlib.patches.Patch(facecolor=color, alpha=scatter_kwargs['alpha'] if 'alpha' in scatter_kwargs.keys() else 1))
        ax.scatter(embed[color_code == grp, 0], embed[color_code == grp, 1], color=color, **scatter_kwargs)
    ax.set_axis_off()

    if legend:
        ax.legend(handles, labels, **legend_kwargs)

def boxplot_hist(data, ax=None, box_w=0.1, box_x=0.5, boxplot_kwargs=None, box_fc='lightgray', half_box=False, hist_width=0.2,
                 hist_gap=0.1, shared_hist_axis=False, hist_kwargs=None, hist_ax_label='Count [-]', scatter_data=False,
                 scatter_width=None, scatter_kwargs=None):
    """
    Plot a combinatation of boxplot and histogram (vertically).
    ----------
    INPUT
        |---- data (list of 1D array-like) the data to display as a histogram + boxplot. Each array of list is one series
        |               that will be displayed as one box-hist.
        |---- ax (plt.Axes) axes where to plot the boxhist.
        |---- box_w (float or 1D-array) the boxplot width of the box. If float, the widths is the same for all box.
        |---- box_x (float or 1D-array) the x-position of the boxplot. If float, the x positions are equally spaced and
        |               the value passed represent the spacing.
        |---- boxplot_kwargs (dict) keyword arguments to be passed to the matplotlib boxplot.
        |---- box_fc (str) boxplot facecolor. Set to None for no facecolor.
        |---- half_box (bool) whether to display the boxplot as only a half-box (with box_w) and sticking the histogram
        |               to the side of the half-box.
        |---- hist_width (float) the space allocated for the histogram on the plot in term of data x-value.
        |---- hist_gap (float) the gap between the boxplot side and the histogram in term of data x-value. Ignored if
        |               half-box is true.
        |---- shared_hist_axis (bool) whether the histograms of the different box_hist should be similar.
        |---- hist_kwargs (dict) keyword arguments for the matplotlib hist method.
        |---- hist_ax_label (str or list of str) the ax label for the hist. If single string, the same is used for each
        |           distribution. If list, it must have the same number of entry as in data.
        |---- scatter_data (float) whether to display the data as a scatter plot on top of the boxplot.
        |---- scatter_width (float) the x-scatter amplitude. Data points will be scatter horizontally from a normal
        |               distribution with a std of scatter_width. If None, scatter_width = 0.1*box_w
        |---- scatter_kwargs (dict) the keyword arguments to be passed to matplotlib scatter.
    OUTPUT
        |---- None
    """
    # control inputs and define default behaviour
    ax = plt.gca() if ax is None else ax
    box_x = np.arange(0, len(data)*box_x, box_x)[:len(data)] if isinstance(box_x, float) else box_x
    box_w = np.repeat(box_w, len(data)) if isinstance(box_w, float) else box_w
    if boxplot_kwargs is None:
        boxplot_kwargs = dict(capprops=dict(lw=2, color='black'),
                              boxprops=dict(lw=2, color='black'),
                              whiskerprops=dict(lw=2, color='black'),
                              flierprops=dict(marker='x', markeredgewidth=1, markerfacecolor='gray', markersize=5),
                              medianprops=dict(lw=2, color='xkcd:vermillion'),
                              meanprops=dict(lw=2, linestyle='-', color='dodgerblue'),
                              showmeans=True, meanline=True,
                              patch_artist=True)
    if hist_kwargs is None:
        hist_kwargs = dict(bins=20, orientation='horizontal', histtype='stepfilled', facecolor='lightgray', edgecolor='black', lw=1, alpha=1)
    hist_ax_label = [hist_ax_label]*len(data) if isinstance(hist_ax_label, str) else hist_ax_label
    if scatter_kwargs is None:
        scatter_kwargs = dict(c='gray', marker='o', s=10, lw=0, alpha=0.1, zorder=4)

    # BOXPLOT
    bp = ax.boxplot(data, positions=box_x, widths=box_w, **boxplot_kwargs)
    # Color box patches
    if box_fc:
        for patch in bp['boxes']:
            patch.set(facecolor='lightgray')
    # Move whisker, cap and flier if halfbox
    if half_box:
        for i, (x_i, w_i) in enumerate(zip(box_x, box_w)):
            bp['whiskers'][2*i].set_xdata([x_i+w_i/2, x_i+w_i/2])
            bp['whiskers'][2*i+1].set_xdata([x_i+w_i/2, x_i+w_i/2])
            bp['caps'][2*i].set_xdata(bp['caps'][2*i].get_xdata() + w_i/2)
            bp['caps'][2*i+1].set_xdata(bp['caps'][2*i+1].get_xdata() + w_i/2)
            bp['fliers'][i].set_xdata(bp['fliers'][i].get_xdata() + w_i/2)

    # HISTOGRAMS
    ax_in_list = []
    for data_i, x_i, w_i, ax_lab in zip(data, box_x, box_w, hist_ax_label):
        # add scatter plot of data
        if scatter_data:
            sigma = 0.1*w_i if scatter_width is None else scatter_width
            ax.scatter(np.random.normal(x_i, scatter_width, data_i.shape[0]), data_i, **scatter_kwargs)
        # add an inset ax for the histogram
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        gap = w_i/2 if half_box else w_i/2 + hist_gap
        ax_in = ax.inset_axes([x_i+gap, 0, hist_width, 1], transform=trans, sharey=ax, zorder=0)
        ax_in_list.append(ax_in)
        hist_kwargs.update({'orientation':'horizontal'})
        ax_in.hist(data_i, **hist_kwargs)

        ax_in.spines['bottom'].set_visible(False)
        ax_in.spines['right'].set_visible(False)
        ax_in.spines['left'].set_visible(False)
        ax_in.set_xlabel(ax_lab)
        ax_in.xaxis.set_label_position('top')
        ax_in.tick_params(labelleft=False, left=False,
                          labelbottom=False, bottom=False,
                          labeltop=True, top=True)
    if shared_hist_axis:
        for ax_i in ax_in_list[1:]:
            ax_in_list[0].get_shared_x_axes().join(ax_in_list[0], ax_i)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def boxplot_hist_h (data, ax=None, box_h=0.1, box_y=0.5, boxplot_kwargs=None, box_fc='lightgray', half_box=False, hist_height=0.2,
                    hist_gap=0.1, shared_hist_axis=False, hist_kwargs=None, hist_ax_label='Count [-]', scatter_data=False,
                    scatter_width=None, scatter_kwargs=None):
    """
    Plot a combinatation of boxplot and histogram (horizontally).
    ----------
    INPUT
        |---- data (list of 1D array-like) the data to display as a histogram + boxplot horizontal. Each array of list is one series
        |               that will be displayed as one box-hist.
        |---- ax (plt.Axes) axes where to plot the boxhist.
        |---- box_h (float or 1D-array) the boxplot hight of the box. If float, the widths is the same for all box.
        |---- box_y (float or 1D-array) the y-position of the boxplot. If float, the y positions are equally spaced and
        |               the value passed represent the spacing.
        |---- boxplot_kwargs (dict) keyword arguments to be passed to the matplotlib boxplot.
        |---- box_fc (str) boxplot facecolor. Set to None for no facecolor.
        |---- half_box (bool) whether to display the boxplot as only a half-box (with box_h) and sticking the histogram
        |               to the side of the half-box.
        |---- hist_height (float) the space allocated for the histogram on the plot in term of data y-value.
        |---- hist_gap (float) the gap between the boxplot side and the histogram in term of data y-value. Ignored if
        |               half-box is true.
        |---- shared_hist_axis (bool) whether the histograms of the different box_hist should be similar.
        |---- hist_kwargs (dict) keyword arguments for the matplotlib hist method.
        |---- hist_ax_label (str or list of str) the ax label for the hist. If single string, the same is used for each
        |           distribution. If list, it must have the same number of entry as in data.
        |---- scatter_data (float) whether to display the data as a scatter plot on top of the boxplot.
        |---- scatter_width (float) the x-scatter amplitude. Data points will be scatter horizontally from a normal
        |               distribution with a std of scatter_width. If None, scatter_width = 0.1*box_h
        |---- scatter_kwargs (dict) the keyword arguments to be passed to matplotlib scatter.
    OUTPUT
        |---- None
    """
    # control inputs and define default behaviour
    ax = plt.gca() if ax is None else ax
    box_y = np.arange(0, len(data)*box_y, box_y)[:len(data)] if isinstance(box_y, float) else box_y
    box_h = np.repeat(box_h, len(data)) if isinstance(box_h, float) else box_h
    if boxplot_kwargs is None:
        boxplot_kwargs = dict(capprops=dict(lw=2, color='black'),
                              boxprops=dict(lw=2, color='black'),
                              whiskerprops=dict(lw=2, color='black'),
                              flierprops=dict(marker='x', markeredgewidth=1, markerfacecolor='gray', markersize=5),
                              medianprops=dict(lw=2, color='xkcd:vermillion'),
                              meanprops=dict(lw=2, linestyle='-', color='dodgerblue'),
                              showmeans=True, meanline=True,
                              patch_artist=True,
                              vert=False)
    if hist_kwargs is None:
        hist_kwargs = dict(bins=20, orientation='vertical', histtype='stepfilled', facecolor='lightgray', edgecolor='black', lw=1, alpha=1)
    hist_ax_label = [hist_ax_label]*len(data) if isinstance(hist_ax_label, str) else hist_ax_label
    if scatter_kwargs is None:
        scatter_kwargs = dict(c='gray', marker='o', s=10, lw=0, alpha=0.1, zorder=4)

    # BOXPLOT
    boxplot_kwargs.update({'vert':False})
    bp = ax.boxplot(data, positions=box_y, widths=box_h, **boxplot_kwargs)
    # Color box patches
    if box_fc:
        for patch in bp['boxes']:
            patch.set(facecolor='lightgray')
    # Move whisker, cap and flier if halfbox
    if half_box:
        for i, (y_i, h_i) in enumerate(zip(box_y, box_h)):
            bp['whiskers'][2*i].set_ydata([y_i+h_i/2, y_i+h_i/2])
            bp['whiskers'][2*i+1].set_ydata([y_i+h_i/2, y_i+h_i/2])
            bp['caps'][2*i].set_ydata(bp['caps'][2*i].get_ydata() + h_i/2)
            bp['caps'][2*i+1].set_ydata(bp['caps'][2*i+1].get_ydata() + h_i/2)
            bp['fliers'][i].set_ydata(bp['fliers'][i].get_ydata() + h_i/2)

    # HISTOGRAMS
    ax_in_list = []
    for data_i, y_i, h_i, ax_lab in zip(data, box_y, box_h, hist_ax_label):
        # add scatter plot of data
        if scatter_data:
            sigma = 0.1*h_i if scatter_width is None else scatter_width
            ax.scatter(data_i, np.random.normal(y_i, sigma, data_i.shape[0]), **scatter_kwargs)
        # add an inset ax for the histogram
        trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        gap = h_i/2 if half_box else h_i/2 + hist_gap
        ax_in = ax.inset_axes([0, y_i+gap, 1, hist_height], transform=trans, sharex=ax, zorder=0)
        ax_in_list.append(ax_in)
        hist_kwargs.update({'orientation':'vertical'})
        ax_in.hist(data_i, **hist_kwargs)

        ax_in.spines['top'].set_visible(False)
        ax_in.spines['left'].set_visible(False)
        #ax_in.spines['bottom'].set_visible(False)
        ax_in.set_ylabel(ax_lab)
        ax_in.yaxis.set_label_position('right')
        ax_in.tick_params(labelleft=False, left=False,
                          labelbottom=False, bottom=False,
                          labelright=True, right=True)

    if shared_hist_axis:
        for ax_i in ax_in_list[1:]:
            ax_in_list[0].get_shared_y_axes().join(ax_in_list[0], ax_i)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
