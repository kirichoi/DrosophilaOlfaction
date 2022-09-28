# -*- coding: utf-8 -*-
"""
Olfactory responses of Drosophila are encoded in the organization of projection neurons

Kiri Choi, Won Kyu Kim, Changbong Hyeon
School of Computational Sciences, Korea Institute for Advanced Study, Seoul 02455, Korea

This script reproduces figures related to the labaled-line study based on the 
hemibrain dataset that uses uPNs that innervate all three neuropils and KCs and
LHNs making synapses with the uPNs, using higher synaptic weight threshold
"""

import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib import colors
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
import pandas as pd
import scipy.cluster
import scipy.optimize
from collections import Counter
import copy
import sklearn.cluster
from neuprint.utils import connection_table_to_matrix

os.chdir(os.path.dirname(__file__))

FAFB_glo_info = pd.read_csv('./1-s2.0-S0960982220308587-mmc4.csv')

glo_info = np.load('./hemibrain/neuprint_PNallinrvt_df.pkl', allow_pickle=True) # Path to glomerulus label information
uPNid = glo_info['bodyId']

glo_list_neuron = np.array(glo_info['type'])
glo_list_neuron = [i.split('_', 1)[0] for i in glo_list_neuron]
glo_list_neuron = [i.split('+', 1) for i in glo_list_neuron]

for j,i in enumerate(glo_list_neuron):
    if len(i) > 1 and i[1] != '':
        neuprintId = uPNid.iloc[j]
        ugloidx = np.where(FAFB_glo_info['hemibrain_bodyid'] == neuprintId)[0]
        uglo = FAFB_glo_info['top_glomerulus'][ugloidx]
        glo_list_neuron[j] = [uglo.iloc[0]]
    if len(i) > 1 and i[1] == '':
        glo_list_neuron[j] = [i[0]]

glo_list_neuron = np.array([item for sublist in glo_list_neuron for item in sublist])

glo_list = np.unique(glo_list_neuron)

#%% Updated glomerulus label

glo_list_neuron_new = copy.deepcopy(glo_list_neuron)
glo_list_new = copy.deepcopy(glo_list)

vc3m = np.where(glo_list_neuron_new == 'VC3m')
vc3l = np.where(glo_list_neuron_new == 'VC3l')
vc5 = np.where(glo_list_neuron_new == 'VC5')

glo_list_neuron_new[vc3m] = 'VC5'
glo_list_neuron_new[vc3l] = 'VC3'
glo_list_neuron_new[vc5] = 'VM6'

vc3m = np.where(glo_list_new == 'VC3m')
vc3l = np.where(glo_list_new == 'VC3l')
vc5 = np.where(glo_list_new == 'VC5')

glo_list_new[vc3m] = 'VC5'
glo_list_new[vc3l] = 'VC3'
glo_list_new[vc5] = 'VM6'

#%%

glo_idx = []

for i in range(len(glo_list_new)):
    glo_idx.append(list(np.where(glo_list_neuron_new == glo_list_new[i])[0]))

glo_idx_flat = [item for sublist in glo_idx for item in sublist]

morph_dist_MB_r_new = np.load(r'./hemibrain/morph_dist_MB_r_neuprint.npy')
morph_dist_LH_r_new = np.load(r'./hemibrain/morph_dist_LH_r_neuprint.npy')

L_MB_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_MB_r_new), method='complete', optimal_ordering=True)
L_LH_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_LH_r_new), method='complete', optimal_ordering=True)

PNKC_df = pd.read_pickle(r'./hemibrain/neuron_PNKC_df_8.pkl')
PNLH_df = pd.read_pickle(r'./hemibrain/neuron_PNLH_df_8.pkl')

conn_PNKC_df = pd.read_pickle(r'./hemibrain/conn_PNKC_df_8.pkl')
conn_PNLH_df = pd.read_pickle(r'./hemibrain/conn_PNLH_df_8.pkl')


KCmatrix = connection_table_to_matrix(conn_PNKC_df, 'bodyId')
KCmatrix[KCmatrix > 1] = 1

a = KCmatrix.align(KCmatrix.T)
b = []
for i in a:
    b.append(i.fillna(0).astype(int))

KCmatrix_adj = pd.DataFrame(np.sum(b, axis=0))
KCmatrix_adj.columns = a[0].columns
KCmatrix_adj.index = a[0].index

LHmatrix = connection_table_to_matrix(conn_PNLH_df, 'bodyId')
LHmatrix[LHmatrix > 1] = 1

a = LHmatrix.align(LHmatrix.T)
b = []
for i in a:
    b.append(i.fillna(0).astype(int))

LHmatrix_adj = pd.DataFrame(np.sum(b, axis=0))
LHmatrix_adj.columns = a[0].columns
LHmatrix_adj.index = a[0].index

KCglo_list_neuron_sortedidx = []

for i in KCmatrix.index:
    KCglo_list_neuron_sortedidx.append(np.where(glo_info['bodyId'] == i)[0][0])
    
KCglo_list_neuron_sorted = glo_list_neuron_new[KCglo_list_neuron_sortedidx]
    

LHglo_list_neuron_sortedidx = []

for i in LHmatrix.index:
    LHglo_list_neuron_sortedidx.append(np.where(glo_info['bodyId'] == i)[0][0])
    
LHglo_list_neuron_sorted = glo_list_neuron_new[LHglo_list_neuron_sortedidx]

#%% Calculating homotype-specific connections

type_idx = [46, 37, 12, 11, 40, 43, 41, 
            49, 20, 21, 16, 19, 29, 35,
            28, 47, 48, 10, 3, 44, 6,
            39, 33, 50, 5,
            38, 17, 45, 24, 22, 23, 27, 42,
            36, 7, 2, 0, 4, 9, 14, 15, 18,
            31, 34,
            30, 32,
            26, 8, 1, 25, 13, 
            53, 57, 51, 52, 54, 55, 56]

attavdict2 = {'DL2d': '#027000', 'DL2v': '#027000', 'VL1': '#027000', 'VL2a': '#027000', 'VM1': '#027000', 'VM4': '#027000', 'VC5': '#027000',
             'DM1': '#5dad2f', 'DM4': '#5dad2f', 'DM5': '#5dad2f', 'DM6': '#5dad2f', 'VA4': '#5dad2f', 'VC2': '#5dad2f', 'VM7d': '#5dad2f',
             'DA3': '#05cf02', 'DC1': '#05cf02', 'DL1': '#05cf02', 'VA3': '#05cf02', 'VM2': '#05cf02', 'VM5d': '#05cf02', 'VM5v': '#05cf02',  
             'DA4m': '#858585', 'VA7m': '#858585', 'VM7v': '#858585', 'VM6': '#858585', 
             'DM2': '#17becf', 'DP1l': '#17becf', 'DP1m': '#17becf', 'V': '#17becf', 'VA2': '#17becf', 'VC4': '#17becf', 'VL2p': '#17becf', 'VM3': '#17becf', 
             'D': '#bf0000', 'DA2': '#bf0000', 'DA4l': '#bf0000', 'VC3': '#bf0000', 'DC2': '#bf0000', 'DC4': '#bf0000', 'DL4': '#bf0000', 'DL5': '#bf0000', 'DM3': '#bf0000',
             'VA6': '#d4d296', 'VC1': '#d4d296', 
             'VA5': '#91451f', 'VA7l': '#91451f', 
             'DA1': '#700099', 'DC3': '#700099', 'DL3': '#700099', 'VA1d': '#700099', 'VA1v': '#700099',
             'VP1d': '#ff00ff', 'VP1l': '#ff00ff', 'VP1m': '#ff00ff', 'VP2': '#ff00ff', 'VP3': '#ff00ff', 'VP4': '#ff00ff', 'VP5': '#ff00ff'}

gsum_KC = np.empty((len(glo_list_new), np.shape(KCmatrix)[1]))

for j,i in enumerate(glo_list_new):
    gidx = np.where(KCglo_list_neuron_sorted == i)[0]
    g = np.array(KCmatrix)[gidx]
    gsum_KC[j] = np.sum(g, axis=0)

gsum_KC_binary = copy.deepcopy(gsum_KC)
gsum_KC_binary[gsum_KC_binary > 1] = 1

glabeled_KC = []
glabeled_info_KC = []
glabeled_glo_KC = []
glabeled_glo_info_KC = []
glabeled_KC_idx = []

for i in range(len(gsum_KC)):
    glabeled_KC_idx.append(np.nonzero(gsum_KC[i])[0])
    glabeled_glo_KC_temp = []
    gsumc = np.where(gsum_KC[i] != 0)[0]
    labeled = []
    for j in range(len(gsumc)):
        nznum = np.nonzero(gsum_KC[:,gsumc[j]])[0]
        if len(nznum) < 2:
            labeled.append(True)
            glabeled_glo_KC_temp.append(glo_list_new[nznum])
        else:
            labeled.append(False)
            glabeled_glo_KC_temp.append(glo_list_new[nznum])
    glabeled_KC.append(labeled)
    glabeled_info_KC.append(Counter(labeled))
    full = [item for sublist in glabeled_glo_KC_temp for item in sublist]
    glabeled_glo_KC.append(np.unique(full))
    glabeled_glo_info_KC.append([len(glabeled_glo_KC_temp), Counter(full)])
    

gsum_LH = np.empty((len(glo_list_new), np.shape(LHmatrix)[1]))

for j,i in enumerate(glo_list_new):
    gidx = np.where(LHglo_list_neuron_sorted == i)[0]
    g = np.array(LHmatrix)[gidx]
    gsum_LH[j] = np.sum(g, axis=0)

gsum_LH_binary = copy.deepcopy(gsum_LH)
gsum_LH_binary[gsum_LH_binary > 1] = 1

glabeled_LH = []
glabeled_info_LH = []
glabeled_glo_LH = []
glabeled_glo_info_LH = []
glabeled_LH_idx = []

for i in range(len(gsum_LH)):
    glabeled_LH_idx.append(np.nonzero(gsum_LH[i])[0])
    glabeled_glo_LH_temp = []
    gsumc = np.where(gsum_LH[i] != 0)[0]
    labeled = []
    for j in range(len(gsumc)):
        nznum = np.nonzero(gsum_LH[:,gsumc[j]])[0]
        if len(nznum) < 2:
            labeled.append(True)
            glabeled_glo_LH_temp.append(glo_list_new[nznum])
        else:
            labeled.append(False)
            glabeled_glo_LH_temp.append(glo_list_new[nznum])
    glabeled_LH.append(labeled)
    glabeled_info_LH.append(Counter(labeled))
    full = [item for sublist in glabeled_glo_LH_temp for item in sublist]
    glabeled_glo_LH.append(np.unique(full))
    glabeled_glo_info_LH.append([len(glabeled_glo_LH_temp), Counter(full)])

#%% Figure 10-Figure supplement 1

KC_unique = []
KC_nonunique = []

for i in glabeled_info_KC:
    KC_unique.append(i[True])
    KC_nonunique.append(i[False])

fig, ax = plt.subplots(figsize=(12,3))
x = np.arange(len(glo_list_new))
width = .4

bar1 = ax.bar(x - width/2, np.array(KC_unique)[type_idx], width, capsize=5, label=r'$N_{unique}$', color='tab:blue')
ax2 = ax.twinx()
bar3 = ax2.bar(x + width/2, np.divide(KC_unique, np.add(KC_unique, KC_nonunique))[type_idx], width, capsize=5, label=r'$N_{unique}/N_{all}$', color='tab:red')
ax.set_ylim(0, 16)
ax2.set_ylim(0, 1)
ax.set_xlim(-1, len(glo_list_new))
ax.set_ylabel(r'$N_{\alpha,{\rm sp}}^{\rm PN-KC}$', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_new[type_idx], fontsize=13, rotation=90)
ax.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
ax2.set_ylabel(r'$f_{\alpha}$', fontsize=17, rotation=-90, labelpad=25)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
[t.set_color(i) for (i,t) in zip(list(attavdict2.values()), ax.xaxis.get_ticklabels())]
plt.tight_layout()
plt.show()


LH_unique = []
LH_nonunique = []

for i in glabeled_info_LH:
    LH_unique.append(i[True])
    LH_nonunique.append(i[False])

fig, ax = plt.subplots(figsize=(12,3))
x = np.arange(len(glo_list_new))
width = .4

bar1 = ax.bar(x - width/2, np.array(LH_unique)[type_idx], width, capsize=5, label=r'$N_{unique}$', color='tab:blue')
ax2 = ax.twinx()
bar3 = ax2.bar(x + width/2, np.divide(LH_unique, np.add(LH_unique, LH_nonunique))[type_idx], width, capsize=5, label=r'$N_{unique}/N_{all}$', color='tab:red')
ax.set_ylim(0, 32)
ax2.set_ylim(0, 1)
ax.set_xlim(-1, len(glo_list_new))
ax.set_ylabel(r'$N_{\alpha,{\rm sp}}^{\rm PN-LHN}$', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_new[type_idx], fontsize=13, rotation=90)
ax.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
ax2.set_ylabel(r'$f_{\alpha}$', fontsize=17, rotation=-90, labelpad=25)
# ax2.set_ylabel('Ratio', fontsize=17)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
[t.set_color(i) for (i,t) in zip(list(attavdict2.values()), ax.xaxis.get_ticklabels())]
plt.tight_layout()
plt.show()

#%% Calculating common synapse matrices

hconn_KC = np.zeros((len(glo_list_new), len(glo_list_new)))
hconn_KC_weighted = np.zeros((len(glo_list_new), len(glo_list_new)))
hconn_KC_nonweighted = np.zeros((len(glo_list_new), len(glo_list_new)))

for i in range(len(hconn_KC)):
    for j in range(len(hconn_KC)):
        if glo_list_new[j] in glabeled_glo_KC[i]:
            hconn_KC[i][j] = 1
        
for i in range(len(hconn_KC)):
    for j in range(len(hconn_KC)):
        hconn_KC_weighted[i][j] = glabeled_glo_info_KC[i][1][glo_list_new[j]]/glabeled_glo_info_KC[i][0]
        hconn_KC_nonweighted[i][j] = glabeled_glo_info_KC[i][1][glo_list_new[j]]
        
hconn_LH = np.zeros((len(glo_list_new), len(glo_list_new)))
hconn_LH_weighted = np.zeros((len(glo_list_new), len(glo_list_new)))
hconn_LH_nonweighted = np.zeros((len(glo_list_new), len(glo_list_new)))

for i in range(len(hconn_LH)):
    for j in range(len(hconn_LH)):
        if glo_list_new[j] in glabeled_glo_LH[i]:
            hconn_LH[i][j] = 1

for i in range(len(hconn_LH)):
    for j in range(len(hconn_LH)):
        hconn_LH_weighted[i][j] = glabeled_glo_info_LH[i][1][glo_list_new[j]]/glabeled_glo_info_LH[i][0]
        hconn_LH_nonweighted[i][j] = glabeled_glo_info_LH[i][1][glo_list_new[j]]

hconn_KC_df = pd.DataFrame(hconn_KC)
hconn_KC_df = hconn_KC_df.reindex(type_idx, axis=0)
hconn_KC_df = hconn_KC_df.reindex(type_idx, axis=1)

hconn_KC_weighted_df = pd.DataFrame(hconn_KC_weighted)
hconn_KC_weighted_df = hconn_KC_weighted_df.reindex(type_idx, axis=0)
hconn_KC_weighted_df = hconn_KC_weighted_df.reindex(type_idx, axis=1)

hconn_KC_nonweighted_df = pd.DataFrame(hconn_KC_nonweighted)
hconn_KC_nonweighted_df = hconn_KC_nonweighted_df.reindex(type_idx, axis=0)
hconn_KC_nonweighted_df = hconn_KC_nonweighted_df.reindex(type_idx, axis=1)

hconn_LH_df = pd.DataFrame(hconn_LH)
hconn_LH_df = hconn_LH_df.reindex(type_idx, axis=0)
hconn_LH_df = hconn_LH_df.reindex(type_idx, axis=1)

hconn_LH_weighted_df = pd.DataFrame(hconn_LH_weighted)
hconn_LH_weighted_df = hconn_LH_weighted_df.reindex(type_idx, axis=0)
hconn_LH_weighted_df = hconn_LH_weighted_df.reindex(type_idx, axis=1)

hconn_LH_nonweighted_df = pd.DataFrame(hconn_LH_nonweighted)
hconn_LH_nonweighted_df = hconn_LH_nonweighted_df.reindex(type_idx, axis=0)
hconn_LH_nonweighted_df = hconn_LH_nonweighted_df.reindex(type_idx, axis=1)

def make_colormap(seq, name='mycmap'):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return colors.LinearSegmentedColormap(name, cdict)

def generate_cmap(lowColor, highColor, lowBorder, highBorder):
    """Apply edge colors till borders and middle is in grey color"""
    c = colors.ColorConverter().to_rgb
    return make_colormap([c(lowColor), c('w'), lowBorder, c('w'), .3,
                          c('w'), highBorder, c('w'), c(highColor)])

#%% Figure 11-Figure supplement 2

custom_cmap = generate_cmap('tab:blue', 'r', .299, .301)
custom_cmap.set_bad(color='k')

masked_array = np.ma.masked_where(hconn_KC_nonweighted_df == 0.0, hconn_KC_nonweighted_df)

fig = plt.figure(figsize=(8,8))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(masked_array, cmap=custom_cmap, 
                vmax=40)
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(np.arange(len(glo_list)+1))
ax3.set_yticks(np.arange(len(glo_list)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(glo_list))+0.5)))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(glo_list))+0.5)))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add(np.repeat('KC-', len(glo_list_new)), glo_list_new[type_idx])))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add(np.repeat('KC-', len(glo_list_new)), glo_list_new[type_idx])))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
cbar = plt.colorbar(im, fraction=0.045)
cbar.ax.tick_params(labelsize=15)
plt.show()

plt.show()


masked_array = np.ma.masked_where(hconn_LH_nonweighted_df == 0.0, hconn_LH_nonweighted_df)

fig = plt.figure(figsize=(8,8))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(masked_array, cmap=custom_cmap, 
                vmax=np.quantile(hconn_LH_nonweighted[np.triu_indices_from(hconn_LH_nonweighted, k=1)], 0.97))
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline1 = ax2.get_grid_helper().new_fixed_axis
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax2.axis["top"] = new_axisline1(loc="top", axes=ax2, offset=offset1)
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax2.axis["top"].minor_ticks.set_ticksize(0)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax2.axis["bottom"].set_visible(False)
ax3.axis["right"].set_visible(False)
ax2.set_xticks(np.arange(len(glo_list)+1))
ax3.set_yticks(np.arange(len(glo_list)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(glo_list))+0.5)))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(glo_list))+0.5)))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add(np.repeat('LHN-', len(glo_list_new)), glo_list_new[type_idx])))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add(np.repeat('LHN-', len(glo_list_new)), glo_list_new[type_idx])))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
cbar = plt.colorbar(im, fraction=0.045)
cbar.ax.tick_params(labelsize=15)
plt.show()

