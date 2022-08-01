# -*- coding: utf-8 -*-
"""
Olfactory responses of Drosophila are encoded in the organization of projection neurons

Kiri Choi, Won Kyu Kim, Changbong Hyeon
School of Computational Sciences, Korea Institute for Advanced Study, Seoul 02455, Korea

This script reproduces figures related to the labaled-line study based on the 
hemibrain dataset that uses uPNs that innervate all three neuropils and KCs and
LHNs making synapses with the uPNs
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

PNKC_df = pd.read_pickle(r'./hemibrain/neuron_PNKC_df.pkl')
PNLH_df = pd.read_pickle(r'./hemibrain/neuron_PNLH_df.pkl')

conn_PNKC_df = pd.read_pickle(r'./hemibrain/conn_PNKC_df.pkl')
conn_PNLH_df = pd.read_pickle(r'./hemibrain/conn_PNLH_df.pkl')


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

#%% Figure S6

fig = plt.figure(figsize=(np.shape(gsum_KC_binary)[1]*0.01,8))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(gsum_KC_binary[type_idx], cmap='Greys', interpolation='None')
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax3.axis["right"].set_visible(False)
ax3.set_yticks(np.arange(len(glo_list_new)+1))
ax3.invert_yaxis()
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(glo_list_new))+0.5)))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_new[type_idx]))
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
# plt.savefig(r'./Revision figures/conn_raw_KC_1.pdf', dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=(np.shape(gsum_LH_binary)[1]*0.01,8))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(gsum_LH_binary[type_idx], cmap='Greys', interpolation='None')
ax1.axis["left"].set_visible(False)
ax1.axis["bottom"].set_visible(False)
ax3 = ax1.twinx()
offset1 = 0, 10
offset2 = -10, 0
new_axisline2 = ax3.get_grid_helper().new_fixed_axis
ax3.axis["left"] = new_axisline2(loc="left", axes=ax3, offset=offset2)
ax3.axis["left"].minor_ticks.set_ticksize(0)
ax3.axis["right"].set_visible(False)
ax3.set_yticks(np.arange(len(glo_list_new)+1))
ax3.invert_yaxis()
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(glo_list_new))+0.5)))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_new[type_idx]))
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
# plt.savefig(r'./Revision figures/conn_raw_LH_1.pdf', dpi=300, bbox_inches='tight')
plt.show()

    
#%% Figure 10

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
ax2.set_ylim(0, 0.5)
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
ax.set_ylim(0, 16)
ax2.set_ylim(0, 0.5)
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

#%% Figure 11

custom_cmap = generate_cmap('tab:blue', 'r', .299, .301)
custom_cmap.set_bad(color='k')

masked_array = np.ma.masked_where(hconn_KC_nonweighted_df == 0.0, hconn_KC_nonweighted_df)

fig = plt.figure(figsize=(8,8))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(masked_array, cmap=custom_cmap, 
                vmax=np.quantile(hconn_LH_nonweighted[np.triu_indices_from(hconn_LH_nonweighted, k=1)], 0.99))
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
                vmax=np.quantile(hconn_LH_nonweighted[np.triu_indices_from(hconn_LH_nonweighted, k=1)], 0.99))
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

#%% Figure S7

fig, ax = plt.subplots(figsize=(12,3))
x = np.arange(len(glo_list_new))

bar1 = ax.bar(x, np.diag(hconn_KC_nonweighted_df), width, capsize=5, label=r'$N_{unique}$', color='tab:blue')
ax.set_ylim(0, 500)
ax.set_xlim(-1, len(glo_list_new))
ax.set_ylabel(r'$N_{\alpha,\alpha}^{\rm PN-KC}$', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_new[type_idx], fontsize=13, rotation=90)
ax.tick_params(axis="y", labelsize=15)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
[t.set_color(i) for (i,t) in zip(list(attavdict2.values()), ax.xaxis.get_ticklabels())]
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(12,3))
x = np.arange(len(glo_list_new))

bar1 = ax.bar(x, np.diag(hconn_LH_nonweighted_df), width, capsize=5, label=r'$N_{unique}$', color='tab:blue')
ax.set_ylim(0, 500)
ax.set_xlim(-1, len(glo_list_new))
ax.set_ylabel(r'$N_{\alpha,\alpha}^{\rm PN-LHN}$', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(glo_list_new[type_idx], fontsize=13, rotation=90)
ax.tick_params(axis="y", labelsize=15)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
[t.set_color(i) for (i,t) in zip(list(attavdict2.values()), ax.xaxis.get_ticklabels())]
plt.tight_layout()
plt.show()

#%% Connectivity-based clustering using cosine distance

cos_sim_PNKC = np.zeros((len(KCmatrix), len(KCmatrix)))

for i in range(len(KCmatrix)):
    for j in range(len(KCmatrix)):
        if i == j:
            cos_sim_PNKC[i][j] = 0
        elif cos_sim_PNKC[j][i] != 0:
            cos_sim_PNKC[i][j] = cos_sim_PNKC[j][i]
        else:
            cos_sim_PNKC[i][j] = scipy.spatial.distance.cosine([np.array(KCmatrix)[i]], [np.array(KCmatrix)[j]])

cos_sim_PNLH = np.zeros((len(LHmatrix), len(LHmatrix)))

for i in range(len(LHmatrix)):
    for j in range(len(LHmatrix)):
        if i == j:
            cos_sim_PNLH[i][j] = 0
        elif cos_sim_PNLH[j][i] != 0:
            cos_sim_PNLH[i][j] = cos_sim_PNLH[j][i]
        else:
            cos_sim_PNLH[i][j] = scipy.spatial.distance.cosine([np.array(LHmatrix)[i]], [np.array(LHmatrix)[j]])            

L_PNKC_cosine = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(cos_sim_PNKC), method='ward', optimal_ordering=True)

L_PNLH_cosine = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(cos_sim_PNLH), method='ward', optimal_ordering=True)


#%% Untangle tanglegram of spatial proximity-based vs connectivity-based clustering - Figure 12

import tanglegram as tg

L_MB_new_ind_un, L_PNKC_cosine_un = tg.untangle(L_MB_new_ind, L_PNKC_cosine, 
                                                labels1=np.array(uPNid)[glo_idx_flat], labels2=np.array(KCmatrix.index), method='step1side')

fig = tg.plot(L_MB_new_ind, L_PNKC_cosine, labelsA=np.array(uPNid)[glo_idx_flat], 
              labelsB=np.array(KCmatrix.index), sort='step1side', figsize=(8, 15))
plt.show()


L_LH_new_ind_un, L_PNLH_cosine_un = tg.untangle(L_LH_new_ind, L_PNLH_cosine,
                                                labels1=np.array(uPNid)[glo_idx_flat], labels2=np.array(LHmatrix.index), method='step1side')

fig = tg.plot(L_LH_new_ind, L_PNLH_cosine, labelsA=np.array(uPNid)[glo_idx_flat], 
              labelsB=np.array(LHmatrix.index), sort='step1side', figsize=(8, 15))
plt.show()

matplotlib.rc_file_defaults()

fig, ax = plt.subplots(figsize=(20, 3))
R_MB_new_ind_cosine_un = scipy.cluster.hierarchy.dendrogram(L_MB_new_ind_un,
                                        orientation='top',
                                        labels=np.array(uPNid)[glo_idx_flat],
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=1.1)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()


fig, ax = plt.subplots(figsize=(20, 3))
R_PNKC_cosine_un = scipy.cluster.hierarchy.dendrogram(L_PNKC_cosine_un,
                                        orientation='top',
                                        labels=np.array(KCmatrix.index),
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=1.15)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()


fig, ax = plt.subplots(figsize=(20, 3))
R_LH_new_ind_cosine_un = scipy.cluster.hierarchy.dendrogram(L_LH_new_ind_un,
                                        orientation='top',
                                        labels=np.array(uPNid)[glo_idx_flat],
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=1.1)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()


fig, ax = plt.subplots(figsize=(20, 3))
R_PNLH_cosine_un = scipy.cluster.hierarchy.dendrogram(L_PNLH_cosine_un,
                                        orientation='top',
                                        labels=np.array(LHmatrix.index),
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=1.15)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()


#%% Tree-cutting using dynamic cut tree hybrid method

from dynamicTreeCut import cutreeHybrid

ind_MB = cutreeHybrid(L_MB_new_ind, scipy.spatial.distance.squareform(morph_dist_MB_r_new), minClusterSize=4)['labels']

ind_LH = cutreeHybrid(L_LH_new_ind, scipy.spatial.distance.squareform(morph_dist_LH_r_new), minClusterSize=4)['labels']

ind_PNKC_cosine = cutreeHybrid(L_PNKC_cosine, scipy.spatial.distance.squareform(cos_sim_PNKC), minClusterSize=4)['labels']

ind_PNLH_cosine = cutreeHybrid(L_PNLH_cosine, scipy.spatial.distance.squareform(cos_sim_PNLH), minClusterSize=4)['labels']


#%% Bakers Gamma Index

from bisect import bisect_left, bisect_right

red = []
red_ind = []
for j,i in enumerate(np.array(uPNid)[glo_idx_flat][R_MB_new_ind_cosine_un['leaves']]):
    if i in np.array(KCmatrix.index):
        red.append(i)
        red_ind.append(np.where(np.array(KCmatrix.index) == i)[0][0])
        
def bakers_gamma(x, y):
    disc = 0
    conc = 0

    for i in range(len(y)):
        cur_disc = bisect_left(x, y[i])
        cur_ties = bisect_right(x, y[i]) - cur_disc
        disc += cur_disc
        conc += len(x) - cur_ties - cur_disc

    bakers_gamma = (conc-disc)/(conc+disc)
    
    return bakers_gamma


print('Bakers Gamma Index - PN-KC')
print(bakers_gamma(np.flip(np.array(KCmatrix.index)[R_PNKC_cosine_un['leaves']]), 
                   np.flip(red)))
print('Bakers Gamma Index - PN-LHN')
print(bakers_gamma(np.flip(np.array(LHmatrix.index)[R_PNLH_cosine_un['leaves']]), 
                   np.flip(np.array(uPNid)[glo_idx_flat][R_LH_new_ind_cosine_un['leaves']])))

print('Normalized Mutual Information - PN-KC')
print(sklearn.metrics.normalized_mutual_info_score(ind_MB[red_ind],
                                                   ind_PNKC_cosine))
print('Normalized Mutual Information - PN-LHN')
print(sklearn.metrics.normalized_mutual_info_score(ind_LH,
                                                   ind_PNLH_cosine))

print('Homogeneity, Completeness, V-Measure - PN-KC')
print(sklearn.metrics.homogeneity_completeness_v_measure(ind_MB[red_ind],
                                                         ind_PNKC_cosine))
print('Homogeneity, Completeness, V-Measure - PN-LHN')
print(sklearn.metrics.homogeneity_completeness_v_measure(ind_LH,
                                                         ind_PNLH_cosine))


#%% Cophenetic distance correlation

re1 = []

for i in np.array(KCmatrix.index)[R_PNKC_cosine_un['leaves']]:
    re1.append(np.where(i == np.array(uPNid)[glo_idx_flat][R_MB_new_ind_cosine_un['leaves']])[0][0])

MB_cpd = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(L_MB_new_ind_un))
MB_cpd = MB_cpd[re1]
MB_cpd = MB_cpd[:,re1]

PNKC_cosine_cpd = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(L_PNKC_cosine_un))
PNKC_cosine_cpd = PNKC_cosine_cpd[R_PNKC_cosine_un['leaves']]
PNKC_cosine_cpd = PNKC_cosine_cpd[:,R_PNKC_cosine_un['leaves']]

re2 = []

for i in np.array(LHmatrix.index)[R_PNLH_cosine_un['leaves']]:
    re2.append(np.where(i == np.array(uPNid)[glo_idx_flat][R_LH_new_ind_cosine_un['leaves']])[0][0])

LH_cpd = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(L_LH_new_ind_un))
LH_cpd = LH_cpd[re2]
LH_cpd = LH_cpd[:,re2]

PNLH_cosine_cpd = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(L_PNLH_cosine_un))
PNLH_cosine_cpd = PNLH_cosine_cpd[R_PNLH_cosine_un['leaves']]
PNLH_cosine_cpd = PNLH_cosine_cpd[:,R_PNLH_cosine_un['leaves']]

print('Cophenetic distance correlation - PN-KC (r, p-value)')
print(scipy.stats.pearsonr(scipy.spatial.distance.squareform(MB_cpd), scipy.spatial.distance.squareform(PNKC_cosine_cpd)))
print('Cophenetic distance correlation - PN-LHN (r, p-value)')
print(scipy.stats.pearsonr(scipy.spatial.distance.squareform(LH_cpd), scipy.spatial.distance.squareform(PNLH_cosine_cpd)))


#%% Pearson's chisquare test

def generate_fix_sum_random_vec(limit, num_elem):
    v = np.zeros(num_elem)
    for i in range(limit):
        p = np.random.randint(num_elem)
        v[p] += 1
    
    return v

def cramers_v(contingency_matrix):
    chi_val, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_matrix)
    n = contingency_matrix.sum().sum()
    phi2 = chi_val/n
    r,k = contingency_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return chi_val, np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))), p_val

a1 = pd.crosstab(KCglo_list_neuron_sorted, ind_PNKC_cosine)
a2 = pd.crosstab(LHglo_list_neuron_sorted, ind_PNLH_cosine)

print("The output is in order of: chi-square value, Cramer's V, and p-value")
print('Glomerular Labels vs C^PN-KC')
print(cramers_v(a1))
print('Glomerular Labels vs C^PN-LHN')
print(cramers_v(a2))

orig1 = np.array(a1)

p1 = []

for i in range(1000):
    shu1 = np.zeros(np.shape(orig1), dtype=int)
    for j in range(len(orig1)):
        shu1[j] = generate_fix_sum_random_vec(np.sum(orig1[j]), len(orig1[j]))
    while len(np.where(np.sum(shu1, axis=0) == 0)[0]) > 0:
        shu1 = np.zeros(np.shape(orig1), dtype=int)
        for j in range(len(orig1)):
            shu1[j] = generate_fix_sum_random_vec(np.sum(orig1[j]), len(orig1[j]))
    shu1 = pd.DataFrame(shu1)
    shu1.index = a1.index
    shu1.columns = a1.columns
    a,b,c = cramers_v(shu1)
    
    p1.append(a)

orig2 = np.array(a2)

p2 = []

for i in range(1000):
    shu2 = np.zeros(np.shape(orig2), dtype=int)
    for j in range(len(orig2)):
        shu2[j] = generate_fix_sum_random_vec(np.sum(orig2[j]), len(orig2[j]))
    while len(np.where(np.sum(shu2, axis=0) == 0)[0]) > 0:
        shu2 = np.zeros(np.shape(orig2), dtype=int)
        for j in range(len(orig2)):
            shu2[j] = generate_fix_sum_random_vec(np.sum(orig2[j]), len(orig2[j]))
    shu2 = pd.DataFrame(shu2)
    shu2.index = a2.index
    shu2.columns = a2.columns
    a,b,c = cramers_v(shu2)
    
    p2.append(a)

print('Monte Carlo: Glomerular Labels vs C^PN-KC (mean, std)')
print(np.mean(p1), np.std(p1))
print('Monte Carlo: Glomerular Labels vs C^PN-LHN (mean, std)')
print(np.mean(p2), np.std(p2))
    
odor_dict = {'DL2d': '#027000', 'DL2v': '#027000', 'VL1': '#027000', 'VL2a': '#027000', 'VM1': '#027000', 'VM4': '#027000', 'VC5': '#027000',
             'DM1': '#5dad2f', 'DM4': '#5dad2f', 'DM5': '#5dad2f', 'DM6': '#5dad2f', 'VA4': '#5dad2f', 'VC2': '#5dad2f', 'VM7d': '#5dad2f',
             'DA3': '#05cf02', 'DC1': '#05cf02', 'DL1': '#05cf02', 'VA3': '#05cf02', 'VM2': '#05cf02', 'VM5d': '#05cf02', 'VM5v': '#05cf02',  
             'DA4m': '#858585', 'VA7m': '#858585', 'VM7v': '#858585', 'VM6': '#858585', 
             'DM2': '#17becf', 'DP1l': '#17becf', 'DP1m': '#17becf', 'V': '#17becf', 'VA2': '#17becf', 'VC4': '#17becf', 'VL2p': '#17becf', 'VM3': '#17becf', 
             'D': '#bf0000', 'DA2': '#bf0000', 'DA4l': '#bf0000', 'VC3': '#bf0000', 'DC2': '#bf0000', 'DC4': '#bf0000', 'DL4': '#bf0000', 'DL5': '#bf0000', 'DM3': '#bf0000',
             'VA6': '#d4d296', 'VC1': '#d4d296', 
             'VA5': '#91451f', 'VA7l': '#91451f', 
             'DA1': '#700099', 'DC3': '#700099', 'DL3': '#700099', 'VA1d': '#700099', 'VA1v': '#700099',
             'VP1d': '#ff00ff', 'VP1l': '#ff00ff', 'VP1m': '#ff00ff', 'VP2': '#ff00ff', 'VP3': '#ff00ff', 'VP4': '#ff00ff', 'VP5': '#ff00ff'}

grp1 = []

for i in glo_list_neuron_new:
    if odor_dict[i] == '#027000':
        grp1.append(1)
    elif odor_dict[i] == '#5dad2f':
        grp1.append(2)
    elif odor_dict[i] == '#05cf02':
        grp1.append(3)
    elif odor_dict[i] == '#858585':
        grp1.append(4)
    elif odor_dict[i] == '#17becf':
        grp1.append(5)
    elif odor_dict[i] == '#bf0000':
        grp1.append(6)
    elif odor_dict[i] == '#d4d296':
        grp1.append(7)
    elif odor_dict[i] == '#91451f':
        grp1.append(8)
    elif odor_dict[i] == '#700099':
        grp1.append(9)
    else:
        grp1.append(10)

grp1 = np.array(grp1)

a3 = pd.crosstab(grp1[KCglo_list_neuron_sortedidx], ind_PNKC_cosine)
a4 = pd.crosstab(grp1[LHglo_list_neuron_sortedidx], ind_PNLH_cosine)

print('Odor Type vs C^PN-KC')
print(cramers_v(a3))
print('Odor Type vs C^PN-LHN')
print(cramers_v(a4))

orig3 = np.array(a3)

p3 = []

for i in range(1000):
    shu3 = np.zeros(np.shape(orig3), dtype=int)
    for j in range(len(orig3)):
        shu3[j] = generate_fix_sum_random_vec(np.sum(orig3[j]), len(orig3[j]))
    while len(np.where(np.sum(shu3, axis=0) == 0)[0]) > 0:
        shu3 = np.zeros(np.shape(orig3), dtype=int)
        for j in range(len(orig3)):
            shu3[j] = generate_fix_sum_random_vec(np.sum(orig3[j]), len(orig3[j]))
    shu3 = pd.DataFrame(shu3)
    shu3.index = a3.index
    shu3.columns = a3.columns
    a,b,c = cramers_v(shu3)
    
    p3.append(a)

orig4 = np.array(a4)

p4 = []

for i in range(1000):
    shu4 = np.zeros(np.shape(orig4), dtype=int)
    for j in range(len(orig4)):
        shu4[j] = generate_fix_sum_random_vec(np.sum(orig4[j]), len(orig4[j]))
    while len(np.where(np.sum(shu4, axis=0) == 0)[0]) > 0:
        shu4 = np.zeros(np.shape(orig4), dtype=int)
        for j in range(len(orig4)):
            shu4[j] = generate_fix_sum_random_vec(np.sum(orig4[j]), len(orig4[j]))
    shu4 = pd.DataFrame(shu4)
    shu4.index = a4.index
    shu4.columns = a4.columns
    a,b,c = cramers_v(shu4)
    
    p4.append(a)

print('Monte Carlo: Odor Type vs C^PN-KC (mean, std)')
print(np.mean(p3), np.std(p3))
print('Monte Carlo: Odor Type vs C^PN-LHN (mean, std)')
print(np.mean(p4), np.std(p4))

odor_dict2 = {'DM6': '#d62728', 'VL2a': '#d62728', 'V': '#d62728', 'DL5': '#d62728', 'DM2': '#d62728', 'VM3': '#d62728',
            'DP1m': '#d62728', 'VL2p': '#d62728', 'DM3': '#d62728', 'VA3': '#2ca02c',
            'VA6': '#d62728', 'DM5': '#d62728', 'DL1': '#d62728', 'D': '#d62728',
            'DC1': '#d62728', 'DC2': '#d62728', 'VA7l': '#d62728', 'VA5': '#d62728',
            'DC3': '#d62728', 'DA2': '#d62728', 'DL4': '#d62728', 'DC4': '#d62728', 
            'DA4l': '#d62728', 'VC3': '#d62728',  'VA7m': '#d62728', 'DA4m': '#d62728', 'VM7d': '#2ca02c',
            'VA2': '#2ca02c', 'DM1': '#2ca02c', 'DM4': '#2ca02c', 'VM5v': '#2ca02c', 
            'VC2': '#2ca02c', 'VM2': '#2ca02c', 'VM5d': '#2ca02c',
            'DA3': '#2ca02c', 'VM4': '#2ca02c', 'VM1': '#2ca02c', 
            'VC1': '#2ca02c', 'VA1v': '#2ca02c', 'DA1': '#2ca02c', 'DL3': '#2ca02c',
            'VM7v': '#2ca02c', 'DP1l': '#000000', 'VC4': '#000000', 'VA4': '#000000', 
            'DL2d': '#000000', 'DL2v': '#000000', 'VC5': '#000000', 'VL1': '#000000', 'VA1d': '#000000',
            'VP1d': '#000000', 'VP1l': '#000000', 
            'VP1m': '#000000', 'VP2': '#000000', 'VP3': '#000000', 'VP4': '#000000', 'VP5': '#000000', 'VM6': '#000000'}

grp2 = []

for i in glo_list_neuron_new:
    if odor_dict2[i] == '#d62728':
        grp2.append(1)
    elif odor_dict2[i] == '#2ca02c':
        grp2.append(2)
    else:
        grp2.append(3)

grp2 = np.array(grp2)

a5 = pd.crosstab(grp2[KCglo_list_neuron_sortedidx], ind_PNKC_cosine)
a6 = pd.crosstab(grp2[LHglo_list_neuron_sortedidx], ind_PNLH_cosine)

print('Odor valence vs C^PN-KC')
print(cramers_v(a5))
print('Odor valence vs C^PN-LHN')
print(cramers_v(a6))

orig5 = np.array(a5)

p5 = []

for i in range(1000):
    shu5 = np.zeros(np.shape(orig5), dtype=int)
    for j in range(len(orig5)):
        shu5[j] = generate_fix_sum_random_vec(np.sum(orig5[j]), len(orig5[j]))
    while len(np.where(np.sum(shu5, axis=0) == 0)[0]) > 0:
        shu5 = np.zeros(np.shape(orig5), dtype=int)
        for j in range(len(orig5)):
            shu5[j] = generate_fix_sum_random_vec(np.sum(orig5[j]), len(orig5[j]))
    shu5 = pd.DataFrame(shu5)
    shu5.index = a5.index
    shu5.columns = a5.columns
    a,b,c = cramers_v(shu5)
    
    p5.append(a)

orig6 = np.array(a6)

p6 = []

for i in range(1000):
    shu6 = np.zeros(np.shape(orig6), dtype=int)
    for j in range(len(orig6)):
        shu6[j] = generate_fix_sum_random_vec(np.sum(orig6[j]), len(orig6[j]))
    while len(np.where(np.sum(shu6, axis=0) == 0)[0]) > 0:
        shu6 = np.zeros(np.shape(orig6), dtype=int)
        for j in range(len(orig6)):
            shu6[j] = generate_fix_sum_random_vec(np.sum(orig6[j]), len(orig6[j]))
    shu6 = pd.DataFrame(shu6)
    shu6.index = a6.index
    shu6.columns = a6.columns
    a,b,c = cramers_v(shu6)
    
    p6.append(a)

print('Monte Carlo: Odor valence vs C^PN-KC (mean, std)')
print(np.mean(p5), np.std(p5))
print('Monte Carlo: Odor valence vs C^PN-LHN (mean, std)')
print(np.mean(p6), np.std(p6))


