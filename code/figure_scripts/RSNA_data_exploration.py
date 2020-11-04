"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :
"""
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pydicom
import skimage
import imageio

from src.utils.plot_utils import draw_cruved_rect
from src.utils.ct_utils import window_ct

#%%#####################################################################################################################
# Load CSV
########################################################################################################################
DATA_PATH = '../../data/RSNA/'
diagnos_df = pd.read_csv(DATA_PATH + 'slice_info.csv', index_col=0)

#%%#####################################################################################################################
# Figure : Data repartition
########################################################################################################################
def human_format(num, pos=None):
    """
    Format large number using a human interpretable unit (kilo, mega, ...).
    ----------
    INPUT
        |---- num (int) -> the number to reformat
    OUTPUT
        |---- num (str) -> the reformated number
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

diagnos_df['No_Hemorrhage'] = 1 - diagnos_df.Hemorrhage

fig, ax = plt.subplots(1, 1, figsize=(7,4))
color = 'xkcd:mango'

# bar plot ICH vs no-ICH
ax.bar([0.5, 1.5], diagnos_df.No_Hemorrhage.value_counts().values, tick_label=['No ICH', 'ICH'],
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax.set_xticklabels(['No ICH', 'ICH'], fontsize=12, rotation=0, ha='center')
ax.set_title('ICH by CT Slice', fontsize=12, fontweight='bold', loc='left')
ax.set_ylabel('Number of CT Slice', fontsize=12)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

# barplot ICH by subtype
ax_in = ax.inset_axes([4, 0, 4, ax.get_ylim()[1]], transform=ax.transData)
ax.set_xlim([0,6])
ICH_types = ['intraventricular', 'intraparenchymal', 'subarachnoid', 'epidural', 'subdural']
ax_in.bar([0.5 + i for i in range(len(ICH_types))], diagnos_df[ICH_types].sum(axis=0).values, tick_label=ICH_types,
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax_in.set_title('Number of Slices by ICH Type', fontsize=12, fontweight='bold', loc='left')
ax_in.set_ylabel('Number of Slices', fontsize=12)
ax_in.set_xticklabels(ICH_types, fontsize=12, rotation=15, ha='right')
ax_in.yaxis.set_ticks_position('right')
ax_in.yaxis.set_label_position("right")
ax_in.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
ax_in.spines['left'].set_visible(False)
ax_in.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax_bis = ax.inset_axes([ax.get_xlim()[0],ax.get_ylim()[0], 2 ,ax.get_ylim()[1]], transform=ax.transData, zorder=-5)
ax_bis.spines['right'].set_visible(False)
ax_bis.spines['top'].set_visible(False)
ax_bis.xaxis.set_major_locator(plt.NullLocator())
ax_bis.yaxis.set_major_locator(plt.NullLocator())

draw_cruved_rect(1.5+0.7/2, 4, diagnos_df.No_Hemorrhage.value_counts().values[1], ax.get_ylim()[1], 0, 0, ax=ax,
                 fc='lightgray', ec='black', lw=0.0, alpha=0.5)
ax_in.patch.set_facecolor('lightgray')
ax_in.patch.set_alpha(0.5)

fig.savefig('../../figures/RSNA_data_exploration/data_stats.pdf', dpi=300, bbox_inches='tight')
plt.show()
