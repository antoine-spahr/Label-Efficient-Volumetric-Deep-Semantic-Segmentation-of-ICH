"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :

"""

import matplotlib.pyplot as plt
import matplotlib
import skimage.io as io
import skimage
import imageio

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
