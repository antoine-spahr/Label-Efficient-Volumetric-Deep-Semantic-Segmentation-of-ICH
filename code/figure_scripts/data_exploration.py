"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :
"""
import sys
sys.path.append('../')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import skimage.io as io
import skimage
import imageio

from src.utils.plot_utils import draw_cruved_rect, pred2GIF

#%%#####################################################################################################################
# Load data
########################################################################################################################
DATA_PATH = '../data/ICH_open/'

diagnos_df = pd.read_csv(DATA_PATH + 'hemorrhage_diagnosis.csv')
patient_df = pd.read_csv(DATA_PATH + 'patient_demographics.csv')

#%%#####################################################################################################################
# Print Basic informations
########################################################################################################################
print(f'>>> Number of patients : {len(diagnos_df.PatientNumber.unique())}')
print(f'>>> Number of CT slices : {len(diagnos_df)}')
print(f'>>> Number of ICH-positive CT : {len(diagnos_df[diagnos_df.No_Hemorrhage == 0])}')

#%%#####################################################################################################################
# Metadata Figure : age and sex distribution of patient
########################################################################################################################
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[0.75, 0.25]))
color = 'xkcd:mango'#'cornflowerblue'

ax1.hist(patient_df['Age\n(years)'], color=color, bins=80//5, range=(0,80))
ax1.hist(patient_df['Age\n(years)'], histtype='step', color='black', bins=80//5, range=(0,80), linewidth=1)
ax1.set_xlabel('Patient age')
ax1.set_ylabel('Count [-]')
ax1.set_title('Patients Age Ditribution')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.bar([0.5, 1.5], patient_df.Gender.value_counts().values, tick_label=patient_df.Gender.value_counts().index,
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax2.set_title('Patients Gender Ditribution')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.savefig('../figures/metadata_stat.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%#####################################################################################################################
# Distribution of CT slices per patient + Number of patient with ICH + distribution of number of ICH slices
########################################################################################################################
fig = plt.figure(figsize=(10,7))
gs = fig.add_gridspec(2, 2, width_ratios=[0.4, 0.6], wspace=0.2, hspace=0.6)
color = 'xkcd:mango'

# distribution of CT splices per patient
ax1 = fig.add_subplot(gs[0,0])
min_val, max_val = diagnos_df.PatientNumber.value_counts().values.min(), diagnos_df.PatientNumber.value_counts().values.max()
bin_range=1
ax1.hist(diagnos_df.PatientNumber.value_counts().values, color=color, bins=(max_val - min_val)//bin_range, range=(min_val, max_val))
ax1.hist(diagnos_df.PatientNumber.value_counts().values, histtype='step', color='black', bins=(max_val - min_val)//bin_range, range=(min_val, max_val), linewidth=1)
ax1.set_xlabel('CT slice per Patient')
ax1.set_ylabel('Count [-]')
ax1.set_title('CT Slice Distribution', fontweight='bold', loc='left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Number of Slice with ICH
ax2 = fig.add_subplot(gs[1,1])
ax2.bar([0.5, 1.5], diagnos_df.No_Hemorrhage.value_counts().values, tick_label=['No ICH', 'ICH'],
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax2.set_title('ICH by CT Slice', fontweight='bold', loc='left')
ax2.set_ylabel('Number of CT Slice')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Number of Volums with ICH
diagnos_df['Hemorrhage'] = 1 - diagnos_df.No_Hemorrhage # make a columns positive for ICH
hemorrhage_df = diagnos_df[['PatientNumber','Hemorrhage']].groupby('PatientNumber').sum().rename(columns={'Hemorrhage':'N_Hemorrhage'})
hemorrhage_df['Hemorrhage'] = (hemorrhage_df.N_Hemorrhage > 0).astype('int') # add column if patient has ICH

ax3 = fig.add_subplot(gs[0,1])
ax3.bar([0.5, 1.5], hemorrhage_df.Hemorrhage.value_counts().values, tick_label=['No ICH', 'ICH'],
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax3.set_title('ICH by Patient', fontweight='bold', loc='left')
ax3.set_ylabel('Number of Patient')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# distribution of number of ICH slice per ICH-positive patient
ax4 = fig.add_subplot(gs[1,0])
df_positive = hemorrhage_df[hemorrhage_df.N_Hemorrhage > 0]
min_val, max_val = 0, df_positive.N_Hemorrhage.max()
bin_range=1
ax4.hist(df_positive.N_Hemorrhage, color=color, bins=(max_val - min_val)//bin_range, range=(min_val, max_val))
ax4.hist(df_positive.N_Hemorrhage, histtype='step', color='black', bins=(max_val - min_val)//bin_range, range=(min_val, max_val), linewidth=1)
ax4.set_xlabel('ICH CT slice per ICH-positive Patient')
ax4.set_ylabel('Count [-]')
ax4.set_title('ICH CT Slice Distribution', fontweight='bold', loc='left')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# distribution of ICH type number of slices
ax5 = ax2.inset_axes([4, 0, 4, ax2.get_ylim()[1]], transform=ax2.transData)#fig.add_subplot(gs[1,2])
ax2.set_xlim([0,6])
ICH_types = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
ax5.bar([0.5 + i for i in range(len(ICH_types))], diagnos_df[ICH_types].sum(axis=0).values, tick_label=ICH_types,
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax5.set_title('Number of Slices by ICH Type', fontweight='bold', loc='left')
ax5.set_ylabel('Number of Slices')
ax5.set_xticklabels(ICH_types, rotation=25, ha='right')
ax5.yaxis.set_ticks_position('right')
ax5.yaxis.set_label_position("right")
ax5.spines['left'].set_visible(False)
ax5.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2_bis = ax2.inset_axes([ax2.get_xlim()[0],ax2.get_ylim()[0], 2 ,ax2.get_ylim()[1]], transform=ax2.transData, zorder=-5)
ax2_bis.spines['right'].set_visible(False)
ax2_bis.spines['top'].set_visible(False)
ax2_bis.xaxis.set_major_locator(plt.NullLocator())
ax2_bis.yaxis.set_major_locator(plt.NullLocator())

draw_cruved_rect(1.5+0.7/2, 4, diagnos_df.No_Hemorrhage.value_counts().values[1], ax2.get_ylim()[1], 0, 0, ax=ax2, fc='lightgray', ec='black', lw=0.0, alpha=0.5)
ax5.patch.set_facecolor('lightgray')
ax5.patch.set_alpha(0.5)

# Number of patient per ICH type
ax6 = ax3.inset_axes([4, 0, 4, ax3.get_ylim()[1]], transform=ax3.transData)#fig.add_subplot(gs[0,2])
ax3.set_xlim([0,6])
ICH_types = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']
ax6.bar([0.5 + i for i in range(len(ICH_types))], (diagnos_df[['PatientNumber']+ICH_types].groupby('PatientNumber').sum() > 0).sum(axis=0).values, tick_label=ICH_types,
        width=0.8, color=color, edgecolor='black', linewidth=1)
ax6.set_title('Number of Patient by ICH Type', fontweight='bold', loc='left')
ax6.set_ylabel('Number of Patient')
ax6.set_xticklabels(ICH_types, rotation=25, ha='right')
ax6.yaxis.set_ticks_position('right')
ax6.yaxis.set_label_position("right")
ax6.spines['left'].set_visible(False)
ax6.spines['top'].set_visible(False)

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3_bis = ax3.inset_axes([ax3.get_xlim()[0], ax3.get_ylim()[0], 2, ax3.get_ylim()[1]], transform=ax3.transData, zorder=-5)
ax3_bis.spines['right'].set_visible(False)
ax3_bis.spines['top'].set_visible(False)
ax3_bis.xaxis.set_major_locator(plt.NullLocator())
ax3_bis.yaxis.set_major_locator(plt.NullLocator())

draw_cruved_rect(1.5+0.7/2, 4, hemorrhage_df.Hemorrhage.value_counts().values[1], ax3.get_ylim()[1], 0, 0, ax=ax3, fc='lightgray', ec='black', lw=0.0, alpha=0.5)
ax6.patch.set_facecolor('lightgray')
ax6.patch.set_alpha(0.5)

fig.savefig('../figures/data_stats.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%#####################################################################################################################
# Sample of ICH slices
########################################################################################################################
# make df of file names
CT_file_df = diagnos_df.copy()
CT_file_df['filename'] = CT_file_df.apply(lambda row: f'{row.PatientNumber:03}/brain/{row.SliceNumber}.jpg', axis=1)
CT_file_df['mask_filename'] = CT_file_df.apply(lambda row: 'None' if row.No_Hemorrhage == 1 else f'{row.PatientNumber:03}/brain/{row.SliceNumber}_HGE_Seg.jpg', axis=1)

fig, axs = plt.subplots(4, 6, figsize=(16, 10), gridspec_kw=dict(hspace=0.0, wspace=0.2))

no_ICH_list = CT_file_df[CT_file_df.No_Hemorrhage == 1].filename.sample(n=12, random_state=1).values
ICH_list = CT_file_df[CT_file_df.No_Hemorrhage == 0].filename.sample(n=12, random_state=69).values
ICH_mask_list = CT_file_df[CT_file_df.No_Hemorrhage == 0].mask_filename.sample(n=12, random_state=69).values

for ax, fn in zip(axs[:,:3].reshape(-1), no_ICH_list):
    ax.imshow(io.imread(DATA_PATH + 'Patients_CT/' + fn), cmap='gray')
    ax.set_axis_off()

for ax, fn, mask_fn in zip(axs[:,3:].reshape(-1), ICH_list, ICH_mask_list):
    ax.imshow(io.imread(DATA_PATH + 'Patients_CT/' + fn), cmap='gray')
    mask = skimage.img_as_bool(io.imread(DATA_PATH + 'Patients_CT/' + mask_fn))
    ax.imshow(np.ma.masked_where(mask == False, mask), cmap=matplotlib.colors.ListedColormap(['xkcd:vermillion', 'white']), alpha=.8)
    ax.set_axis_off()

pos = axs[0,0].get_position()
pos2 = axs[0,1].get_position()
pos3 = axs[1,0].get_position()
pos4 = axs[-1,0].get_position()
fig.patches.extend([plt.Rectangle((pos.x0-0.05*pos.width, pos4.y0-0.05*pos.height),
                                  (pos2.x0-pos.x0)*3 - 0.1*pos.width,
                                  (pos.y0-pos3.y0)*4 - (pos.y0-pos3.y1) + 0.40*pos.height,
                                  fc='black', ec='black', alpha=1, zorder=-1,
                                  transform=fig.transFigure, figure=fig)])
axs[0,1].text(0.5, 1.2, f'Non ICH Slices',
         fontsize=14, fontweight='bold', ha='center', va='center', color='lightgray', transform=axs[0,1].transAxes)

pos = axs[0,3].get_position()
pos2 = axs[0,4].get_position()
pos3 = axs[1,3].get_position()
pos4 = axs[-1,3].get_position()
fig.patches.extend([plt.Rectangle((pos.x0-0.05*pos.width, pos4.y0-0.05*pos.height),
                                  (pos2.x0-pos.x0)*3 - 0.1*pos.width,
                                  (pos.y0-pos3.y0)*4 - (pos.y0-pos3.y1) + 0.40*pos.height,
                                  fc='black', ec='black', alpha=1, zorder=-1,
                                  transform=fig.transFigure, figure=fig)])
axs[0,4].text(0.5, 1.2, f'ICH Slices',
         fontsize=14, fontweight='bold', ha='center', va='center', color='lightgray', transform=axs[0,4].transAxes)

fig.savefig('../figures/CT_sample.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%#####################################################################################################################
# Make some volums as GIF
#######################################################################################################################
# get images and mask
ICH_patient_list = CT_file_df[CT_file_df.No_Hemorrhage == 0].PatientNumber.unique()
PatientID = ICH_patient_list[20]
CT_list, mask_list = [], []
for _, row in CT_file_df[CT_file_df.PatientNumber == PatientID].iterrows():
    CT_list.append(io.imread(DATA_PATH + 'Patients_CT/' + row.filename))
    if row.mask_filename == 'None':
        mask_list.append(np.zeros_like(CT_list[-1]))
    else:
        mask_list.append(skimage.img_as_bool(io.imread(DATA_PATH + 'Patients_CT/' + row.mask_filename)))

pred2GIF(CT_list, mask_list, f'{PatientID}_CT.gif', fps=4)
