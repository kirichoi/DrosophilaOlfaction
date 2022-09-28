# -*- coding: utf-8 -*-
"""
Olfactory responses of Drosophila are encoded in the organization of projection neurons

Kiri Choi, Won Kyu Kim, Changbong Hyeon
School of Computational Sciences, Korea Institute for Advanced Study, Seoul 02455, Korea

This script reproduces figures based on the hemibrain dataset that uses uPNs that
innervate all three neuropils
"""

import os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.cluster
import scipy.optimize
from collections import Counter
import copy

os.chdir(os.path.dirname(__file__))

PATH = r'./hemibrain/neuprint_PN_invte_all' # Path to .swc files
SEED = 1234

# Path to FAFB glomerulus label information
FAFB_glo_info = pd.read_csv('./1-s2.0-S0960982220308587-mmc4.csv') 

# Path to hemibrain glomerulus label information
glo_info = np.load('./hemibrain/neuprint_PNallinrvt_df.pkl', allow_pickle=True) 
uPNid = glo_info['bodyId']

fp = np.core.defchararray.add(np.array(uPNid).astype(str), '.swc')
fp = [os.path.join(PATH, f) for f in fp]

class MorphData():
    
    def __init__(self):
        self.morph_id = []
        self.morph_parent = []
        self.morph_dist = []
        self.neuron_id = []
        self.endP = []
        self.somaP = []
        self.MBdist = []
        self.MBdist_trk = []
        self.MBdist_per_n = []
        self.LHdist = []
        self.LHdist_trk = []
        self.LHdist_per_n = []
        self.ALdist = []
        self.ALdist_trk = []
        self.ALdist_per_n = []
    
class LengthData:
    length_total = np.empty(len(fp))
    length_branch = []
    length_direct = []
    length_MB = []
    length_LH = []
    length_AL = []
    length_MB_total = []
    length_LH_total = []
    length_AL_total = []
    
class BranchData:
    branchTrk = []
    branch_dist = []
    branchP = []
    MB_branchTrk = []
    MB_branchP = []
    MB_endP = []
    LH_branchTrk = []
    LH_branchP = []
    LH_endP = []
    AL_branchTrk = []
    AL_branchP = []
    AL_endP = []
    branchNum = np.empty(len(fp))

np.random.seed(SEED)

MorphData = MorphData()

r_d_20 = -20
r_rad_20 = np.radians(r_d_20)
r_20 = np.array([0, 1, 0])
r_vec_20 = r_rad_20 * r_20
rot20 = Rotation.from_rotvec(r_vec_20)

r_d_35 = 35
r_rad_35 = np.radians(r_d_35)
r_35 = np.array([0, 1, 0])
r_vec_35 = r_rad_35 * r_35
rot35 = Rotation.from_rotvec(r_vec_35)

r_d_40p = 40
r_rad_40p = np.radians(r_d_40p)
r_40p = np.array([0, 0, 1])
r_vec_40p = r_rad_40p * r_40p
rot40p = Rotation.from_rotvec(r_vec_40p)

r_d_50 = 50
r_rad_50 = np.radians(r_d_50)
r_50 = np.array([0, 0, 1])
r_vec_50 = r_rad_50 * r_50
rot50 = Rotation.from_rotvec(r_vec_50)

r_d_60 = -60
r_rad_60 = np.radians(r_d_60)
r_60 = np.array([0, 1, 0])
r_vec_60 = r_rad_60 * r_60
rot60 = Rotation.from_rotvec(r_vec_60)

r_d_50n = -50
r_rad_50n = np.radians(r_d_50n)
r_50n = np.array([0, 0, 1])
r_vec_50n = r_rad_50n * r_50n
rot50n = Rotation.from_rotvec(r_vec_50n)

for f in range(len(fp)):
    print(f, fp[f])
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    
    gen = np.genfromtxt(fp[f])
    df = pd.DataFrame(gen)
    df.iloc[:,0] = df.iloc[:, 0].astype(int)
    df.iloc[:,1] = df.iloc[:, 1].astype(int)
    df.iloc[:,6] = df.iloc[:, 6].astype(int)
    
    MorphData.neuron_id.append(os.path.basename(fp[f]).split('.')[0])
    
    scall = int(df.iloc[np.where(df[6] == -1)[0]].values[0][0])
    MorphData.somaP.append(scall)
    
    MorphData.morph_id.append(df[0].tolist())
    MorphData.morph_parent.append(df[6].tolist())
    MorphData.morph_dist.append(np.divide(np.array(df[[2,3,4]]), 1000).tolist())
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    BranchData.branchNum[f] = int(sum(i > 1 for i in ctrVal))
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]
    branchInd = branchInd[branchInd > 0]

    neu_branchTrk = []
    startid = []
    endid = []
    neu_indBranchTrk = []
    branch_dist_temp1 = []
    length_branch_temp = []
    indMorph_dist_temp1 = []
    indMDistLen_temp = []
    
    list_end = np.setdiff1d(MorphData.morph_id[f], MorphData.morph_parent[f])
    
    BranchData.branchP.append(branchInd.tolist())
    MorphData.endP.append(list_end)
    bPoint = np.append(branchInd, list_end)
    
    MBdist_per_n_temp = []
    LHdist_per_n_temp = []
    ALdist_per_n_temp = []
    length_MB_per_n = []
    length_LH_per_n = []
    length_AL_per_n = []
    MB_branchTrk_temp = []
    MB_branchP_temp = []
    LH_branchTrk_temp = []
    LH_branchP_temp = []
    AL_branchTrk_temp = []
    AL_branchP_temp = []
    MB_endP_temp = []
    LH_endP_temp = []
    AL_endP_temp = []
    
    for bp in range(len(bPoint)):
        if bPoint[bp] != scall:
            neu_branchTrk_temp = []
            branch_dist_temp2 = []
            dist = 0
            
            neu_branchTrk_temp.append(bPoint[bp])
            branch_dist_temp2.append(MorphData.morph_dist[f][MorphData.morph_id[f].index(bPoint[bp])])
            parentTrck = bPoint[bp]
            parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
            if parentTrck != -1:
                neu_branchTrk_temp.append(parentTrck)
                rhs = branch_dist_temp2[-1]
                lhs = MorphData.morph_dist[f][MorphData.morph_id[f].index(parentTrck)]
                branch_dist_temp2.append(lhs)
                dist += np.linalg.norm(np.subtract(rhs, lhs))
            while (parentTrck not in branchInd) and (parentTrck != -1):
                parentTrck = MorphData.morph_parent[f][MorphData.morph_id[f].index(parentTrck)]
                if parentTrck != -1:
                    neu_branchTrk_temp.append(parentTrck)
                    rhs = branch_dist_temp2[-1]
                    lhs = MorphData.morph_dist[f][MorphData.morph_id[f].index(parentTrck)]
                    branch_dist_temp2.append(lhs)
                    dist += np.linalg.norm(np.subtract(rhs, lhs))
                    
            if len(neu_branchTrk_temp) > 1:
                neu_branchTrk.append(neu_branchTrk_temp)
                startid.append(neu_branchTrk_temp[0])
                endid.append(neu_branchTrk_temp[-1])
                branch_dist_temp1.append(branch_dist_temp2)
                length_branch_temp.append(dist)
                
                # rotate -20 degrees
                branch_dist_temp2_rot20 = rot20.apply(branch_dist_temp2)
                # rotate -50 degrees
                branch_dist_temp2_rot50n = rot50n.apply(branch_dist_temp2)
                # rotate 35 degrees
                branch_dist_temp2_rot35 = rot35.apply(branch_dist_temp2)
                # rotate 40 degrees
                branch_dist_temp2_rot40p = rot40p.apply(branch_dist_temp2)
                # rotate 50 degrees
                branch_dist_temp2_rot50 = rot50.apply(branch_dist_temp2)
                # rotate -60 degrees
                branch_dist_temp2_rot60 = rot60.apply(branch_dist_temp2)
                
                if ((np.array(branch_dist_temp2_rot50)[:,0] > -6).all() and (np.array(branch_dist_temp2_rot50n)[:,0] < 23.5).all() and
                    (np.array(branch_dist_temp2_rot20)[:,1] < 24).all() and (np.array(branch_dist_temp2_rot60)[:,2] < 23).all()):
                    MorphData.MBdist.append(branch_dist_temp2)
                    MorphData.MBdist_trk.append(f)
                    MBdist_per_n_temp.append(branch_dist_temp2)
                    length_MB_per_n.append(dist)
                    MB_branchTrk_temp.append(neu_branchTrk_temp)
                    MB_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                    if bPoint[bp] in list_end:
                        MB_endP_temp.append(bPoint[bp])
                elif ((np.array(branch_dist_temp2_rot40p)[:,0] < -4).all() and (np.array(branch_dist_temp2_rot20)[:,1] < 24).all() and
                      (np.array(branch_dist_temp2_rot60)[:,2] < 16.5).all()):
                    MorphData.LHdist.append(branch_dist_temp2)
                    MorphData.LHdist_trk.append(f)
                    LHdist_per_n_temp.append(branch_dist_temp2)
                    length_LH_per_n.append(dist)
                    LH_branchTrk_temp.append(neu_branchTrk_temp)
                    LH_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                    if bPoint[bp] in list_end:
                        LH_endP_temp.append(bPoint[bp])
                elif ((np.array(branch_dist_temp2_rot35)[:,0] > 23).all() and (np.array(branch_dist_temp2_rot35)[:,0] < 40).all() and 
                      (np.array(branch_dist_temp2_rot20)[:,1] > 26).all() and (np.array(branch_dist_temp2_rot20)[:,2] > 20).all()):
                    MorphData.ALdist.append(branch_dist_temp2)
                    MorphData.ALdist_trk.append(f)
                    ALdist_per_n_temp.append(branch_dist_temp2)
                    length_AL_per_n.append(dist)
                    AL_branchTrk_temp.append(neu_branchTrk_temp)
                    AL_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                    if bPoint[bp] in list_end:
                        AL_endP_temp.append(bPoint[bp])
                
    BranchData.branchTrk.append(neu_branchTrk)
    BranchData.branch_dist.append(branch_dist_temp1)
    LengthData.length_branch.append(length_branch_temp)
    
    MorphData.MBdist_per_n.append(MBdist_per_n_temp)
    MorphData.LHdist_per_n.append(LHdist_per_n_temp)
    MorphData.ALdist_per_n.append(ALdist_per_n_temp)
    LengthData.length_MB.append(length_MB_per_n)
    LengthData.length_LH.append(length_LH_per_n)
    LengthData.length_AL.append(length_AL_per_n)
    BranchData.MB_branchTrk.append(MB_branchTrk_temp)
    BranchData.MB_branchP.append(np.unique([item for sublist in MB_branchP_temp for item in sublist]).tolist())
    BranchData.LH_branchTrk.append(LH_branchTrk_temp)
    BranchData.LH_branchP.append(np.unique([item for sublist in LH_branchP_temp for item in sublist]).tolist())
    BranchData.AL_branchTrk.append(AL_branchTrk_temp)
    BranchData.AL_branchP.append(np.unique([item for sublist in AL_branchP_temp for item in sublist]).tolist())
    BranchData.MB_endP.append(MB_endP_temp)
    BranchData.LH_endP.append(LH_endP_temp)
    BranchData.AL_endP.append(AL_endP_temp)

    
#%%

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

a = sorted(Counter(glo_list_neuron).items())
glo_len = [s[1] for s in a]

glo_idx = []

for i in range(len(glo_list)):
    glo_idx.append(list(np.where(glo_list_neuron == glo_list[i])[0]))

glo_lb = [sum(glo_len[0:i]) for i in range(len(glo_len)+1)]
glo_lbs = np.subtract(glo_lb, glo_lb[0])
glo_float = np.divide(glo_lbs, glo_lbs[-1])
glo_idx_flat = [item for sublist in glo_idx for item in sublist]

glo_lb_idx = []

for i in range(len(glo_lb)-1):
    glo_lb_idx.append(np.arange(glo_lb[i],glo_lb[i+1]))

morph_dist_MB = []
morph_dist_LH = []
morph_dist_AL = []

for i in range(len(glo_list)):
    morph_dist_MB_temp = []
    morph_dist_LH_temp = []
    morph_dist_AL_temp = []
    morph_dist_MB_bp_temp = []
    morph_dist_LH_bp_temp = []
    morph_dist_AL_bp_temp = []
    morph_dist_MB_ep_temp = []
    morph_dist_LH_ep_temp = []
    morph_dist_AL_ep_temp = []
    for j in range(len(glo_idx[i])):
        morph_dist_MB_temp2 = []
        morph_dist_LH_temp2 = []
        morph_dist_AL_temp2 = []
        morph_dist_MB_bp_temp2 = []
        morph_dist_LH_bp_temp2 = []
        morph_dist_AL_bp_temp2 = []
        morph_dist_MB_ep_temp2 = []
        morph_dist_LH_ep_temp2 = []
        morph_dist_AL_ep_temp2 = []
        
        for p in range(len(MorphData.morph_dist[glo_idx[i][j]])):
            
            # rotate -20 degrees
            branch_dist_temp2_rot20 = rot20.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate -50 degrees
            branch_dist_temp2_rot50n = rot50n.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate 35 degrees
            branch_dist_temp2_rot35 = rot35.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate 40 degrees
            branch_dist_temp2_rot40p = rot40p.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate 50 degrees
            branch_dist_temp2_rot50 = rot50.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate -60 degrees
            branch_dist_temp2_rot60 = rot60.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            
            if ((np.array(branch_dist_temp2_rot50)[0] > -6).all() and (np.array(branch_dist_temp2_rot50n)[0] < 23.5).all() and
                (np.array(branch_dist_temp2_rot20)[1] < 24).all() and (np.array(branch_dist_temp2_rot60)[2] < 23).all()):
                morph_dist_MB_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
                MB_endP_temp.append(bPoint[bp])
            elif ((np.array(branch_dist_temp2_rot40p)[0] < -4).all() and (np.array(branch_dist_temp2_rot20)[1] < 24).all() and
                  (np.array(branch_dist_temp2_rot60)[2] < 16.5).all()):
                morph_dist_LH_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(branch_dist_temp2_rot35)[0] > 23).all() and (np.array(branch_dist_temp2_rot35)[0] < 40).all() and 
                  (np.array(branch_dist_temp2_rot20)[1] > 26).all() and (np.array(branch_dist_temp2_rot20)[2] > 20).all()):    
                morph_dist_AL_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
 
        morph_dist_MB_temp.append(morph_dist_MB_temp2)
        morph_dist_LH_temp.append(morph_dist_LH_temp2)
        morph_dist_AL_temp.append(morph_dist_AL_temp2)
                
    morph_dist_MB.append(morph_dist_MB_temp)
    morph_dist_LH.append(morph_dist_LH_temp)
    morph_dist_AL.append(morph_dist_AL_temp)
    
cg = np.array(MorphData.MBdist_per_n, dtype=object)[glo_idx_flat]
lg = np.array(MorphData.LHdist_per_n, dtype=object)[glo_idx_flat]
ag = np.array(MorphData.ALdist_per_n, dtype=object)[glo_idx_flat]

cg = [item for sublist in cg for item in sublist]
lg = [item for sublist in lg for item in sublist]
ag = [item for sublist in ag for item in sublist]

MorphData.MBdist_flat_glo = [item for sublist in cg for item in sublist]
MorphData.LHdist_flat_glo = [item for sublist in lg for item in sublist]
MorphData.ALdist_flat_glo = [item for sublist in ag for item in sublist]


#%%

from scipy.spatial import ConvexHull

LengthData.length_MB_total = []
LengthData.length_LH_total = []
LengthData.length_AL_total = []

MBdist_per_n_flat = []
LHdist_per_n_flat = []
ALdist_per_n_flat = []

MBdist_per_n_count = []
LHdist_per_n_count = []
ALdist_per_n_count = []

un_MB = np.unique(MorphData.MBdist_trk)
un_LH = np.unique(MorphData.LHdist_trk)
un_AL = np.unique(MorphData.ALdist_trk)

for i in range(len(un_MB)):
    idx = np.where(np.array(MorphData.MBdist_trk) == un_MB[i])[0]
    tarval = np.array(MorphData.MBdist,dtype=object)[idx]
    MBdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
    sumval = np.sum(LengthData.length_MB[un_MB[i]])
    if MBdist_per_n_flat_t:
        MBdist_per_n_flat.append(MBdist_per_n_flat_t)
        MBdist_per_n_count.append(len(MBdist_per_n_flat_t))
        LengthData.length_MB_total.append(sumval)

for i in range(len(un_LH)):
    idx = np.where(np.array(MorphData.LHdist_trk) == un_LH[i])[0]
    tarval = np.array(MorphData.LHdist,dtype=object)[idx]
    LHdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
    sumval = np.sum(LengthData.length_LH[un_LH[i]])
    if LHdist_per_n_flat_t:
        LHdist_per_n_flat.append(LHdist_per_n_flat_t)
        LHdist_per_n_count.append(len(LHdist_per_n_flat_t))
        LengthData.length_LH_total.append(sumval)

for i in range(len(un_AL)):
    idx = np.where((MorphData.ALdist_trk) == un_AL[i])[0]
    tarval = np.array(MorphData.ALdist,dtype=object)[idx]
    ALdist_per_n_flat_t = [item for sublist in tarval for item in sublist]
    sumval = np.sum(LengthData.length_AL[un_AL[i]])
    if ALdist_per_n_flat_t:
        ALdist_per_n_flat.append(ALdist_per_n_flat_t)
        ALdist_per_n_count.append(len(ALdist_per_n_flat_t))
        LengthData.length_AL_total.append(sumval)

morph_dist_MB_CM = []
morph_dist_LH_CM = []
morph_dist_AL_CM = []

morph_dist_MB_std = []
morph_dist_LH_std = []
morph_dist_AL_std = []

for i in range(len(glo_idx)):
    morph_dist_MB_CM_temp = []
    morph_dist_LH_CM_temp = []
    morph_dist_AL_CM_temp = []
    
    morph_dist_MB_std_temp = []
    morph_dist_LH_std_temp = []
    morph_dist_AL_std_temp = []
    
    for j in range(len(glo_idx[i])):
        morph_dist_MB_CM_temp.append(np.average(np.array(morph_dist_MB[i][j]), axis=0))
        morph_dist_LH_CM_temp.append(np.average(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_AL_CM_temp.append(np.average(np.array(morph_dist_AL[i][j]), axis=0))
        
        morph_dist_MB_std_temp.append(np.std(np.array(morph_dist_MB[i][j]), axis=0))
        morph_dist_LH_std_temp.append(np.std(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_AL_std_temp.append(np.std(np.array(morph_dist_AL[i][j]), axis=0))
    
    morph_dist_MB_CM.append(morph_dist_MB_CM_temp)
    morph_dist_LH_CM.append(morph_dist_LH_CM_temp)
    morph_dist_AL_CM.append(morph_dist_AL_CM_temp)
    
    morph_dist_LH_std.append(morph_dist_LH_std_temp)
    morph_dist_MB_std.append(morph_dist_MB_std_temp)
    morph_dist_AL_std.append(morph_dist_AL_std_temp)
    
morph_dist_MB_flt = [item for sublist in morph_dist_MB for item in sublist]
morph_dist_MB_flat = [item for sublist in morph_dist_MB_flt for item in sublist]

mdMB_xmax = np.max(np.array(morph_dist_MB_flat)[:,0])
mdMB_xmin = np.min(np.array(morph_dist_MB_flat)[:,0])
mdMB_ymax = np.max(np.array(morph_dist_MB_flat)[:,1])
mdMB_ymin = np.min(np.array(morph_dist_MB_flat)[:,1])
mdMB_zmax = np.max(np.array(morph_dist_MB_flat)[:,2])
mdMB_zmin = np.min(np.array(morph_dist_MB_flat)[:,2])

morph_dist_LH_flt = [item for sublist in morph_dist_LH for item in sublist]
morph_dist_LH_flat = [item for sublist in morph_dist_LH_flt for item in sublist]

mdLH_xmax = np.max(np.array(morph_dist_LH_flat)[:,0])
mdLH_xmin = np.min(np.array(morph_dist_LH_flat)[:,0])
mdLH_ymax = np.max(np.array(morph_dist_LH_flat)[:,1])
mdLH_ymin = np.min(np.array(morph_dist_LH_flat)[:,1])
mdLH_zmax = np.max(np.array(morph_dist_LH_flat)[:,2])
mdLH_zmin = np.min(np.array(morph_dist_LH_flat)[:,2])

morph_dist_AL_flt = [item for sublist in morph_dist_AL for item in sublist]
morph_dist_AL_flat = [item for sublist in morph_dist_AL_flt for item in sublist]

mdAL_xmax = np.max(np.array(morph_dist_AL_flat)[:,0])
mdAL_xmin = np.min(np.array(morph_dist_AL_flat)[:,0])
mdAL_ymax = np.max(np.array(morph_dist_AL_flat)[:,1])
mdAL_ymin = np.min(np.array(morph_dist_AL_flat)[:,1])
mdAL_zmax = np.max(np.array(morph_dist_AL_flat)[:,2])
mdAL_zmin = np.min(np.array(morph_dist_AL_flat)[:,2])

hull_MB = ConvexHull(np.array(morph_dist_MB_flat))
MB_vol = hull_MB.volume
MB_area = hull_MB.area
MB_density_l = np.sum(LengthData.length_MB_total)/MB_vol

hull_LH = ConvexHull(np.array(morph_dist_LH_flat))
LH_vol = hull_LH.volume
LH_area = hull_LH.area
LH_density_l = np.sum(LengthData.length_LH_total)/LH_vol

hull_AL = ConvexHull(np.array(morph_dist_AL_flat))
AL_vol = hull_AL.volume
AL_area = hull_AL.area
AL_density_l = np.sum(LengthData.length_AL_total)/AL_vol

#%% Inter-PN distance calculation

# The script can re-calculate the inter-PN distances.
# Change `LOAD = False' do so.
# CAUTION! - THIS WILL TAKE A LONG TIME!
# Using the precomputed array is highly recommended
LOAD = True

if LOAD:
    morph_dist_MB_r_new = np.load(r'./hemibrain/morph_dist_MB_r_neuprint.npy')
    morph_dist_LH_r_new = np.load(r'./hemibrain/morph_dist_LH_r_neuprint.npy')
    morph_dist_AL_r_new = np.load(r'./hemibrain/morph_dist_AL_r_neuprint.npy')
else:
    morph_dist_MB_CM_flat = np.array([item for sublist in morph_dist_MB_CM for item in sublist])
    morph_dist_LH_CM_flat = np.array([item for sublist in morph_dist_LH_CM for item in sublist])
    morph_dist_AL_CM_flat = np.array([item for sublist in morph_dist_AL_CM for item in sublist])
    
    morph_dist_MB_r_new = np.zeros((len(morph_dist_MB_CM_flat), len(morph_dist_MB_CM_flat)))
    morph_dist_LH_r_new = np.zeros((len(morph_dist_LH_CM_flat), len(morph_dist_LH_CM_flat)))
    morph_dist_AL_r_new = np.zeros((len(morph_dist_AL_CM_flat), len(morph_dist_AL_CM_flat)))
    
    for i in range(len(morph_dist_MB_CM_flat)):
        for j in range(len(morph_dist_MB_CM_flat)):
            if i == j:
                morph_dist_MB_r_new[i][j] = 0
                morph_dist_LH_r_new[i][j] = 0
                morph_dist_AL_r_new[i][j] = 0
            elif morph_dist_MB_r_new[j][i] != 0:
                morph_dist_MB_r_new[i][j] = morph_dist_MB_r_new[j][i]
                morph_dist_LH_r_new[i][j] = morph_dist_LH_r_new[j][i]
                morph_dist_AL_r_new[i][j] = morph_dist_AL_r_new[j][i]
            else:
                morph_dist_MB_ed = scipy.spatial.distance.cdist(morph_dist_MB_flt[i], morph_dist_MB_flt[j])
                morph_dist_LH_ed = scipy.spatial.distance.cdist(morph_dist_LH_flt[i], morph_dist_LH_flt[j])
                morph_dist_AL_ed = scipy.spatial.distance.cdist(morph_dist_AL_flt[i], morph_dist_AL_flt[j])
                
                # NNmetric
                if len(morph_dist_MB_flt[i]) < len(morph_dist_MB_flt[j]):
                    N_MB = len(morph_dist_MB_flt[i])
                    dmin_MB = np.min(morph_dist_MB_ed, axis=1)
                elif len(morph_dist_MB_flt[i]) > len(morph_dist_MB_flt[j]):
                    N_MB = len(morph_dist_MB_flt[j])
                    dmin_MB = np.min(morph_dist_MB_ed, axis=0)
                else:
                    N_MB = len(morph_dist_MB_flt[i])
                    r1 = np.min(morph_dist_MB_ed, axis=0)
                    r2 = np.min(morph_dist_MB_ed, axis=1)
                    if np.sum(r1) < np.sum(r2):
                        dmin_MB = r1
                    else:
                        dmin_MB = r2
                
                if len(morph_dist_LH_flt[i]) < len(morph_dist_LH_flt[j]):
                    N_LH = len(morph_dist_LH_flt[i])
                    dmin_LH = np.min(morph_dist_LH_ed, axis=1)
                elif len(morph_dist_LH_flt[i]) > len(morph_dist_LH_flt[j]):
                    N_LH = len(morph_dist_LH_flt[j])
                    dmin_LH = np.min(morph_dist_LH_ed, axis=0)
                else:
                    N_LH = len(morph_dist_LH_flt[i])
                    r1 = np.min(morph_dist_LH_ed, axis=0)
                    r2 = np.min(morph_dist_LH_ed, axis=1)
                    if np.sum(r1) < np.sum(r2):
                        dmin_LH = r1
                    else:
                        dmin_LH = r2
                
                if len(morph_dist_AL_flt[i]) < len(morph_dist_AL_flt[j]):
                    N_AL = len(morph_dist_AL_flt[i])
                    dmin_AL = np.min(morph_dist_AL_ed, axis=1)
                elif len(morph_dist_AL_flt[i]) > len(morph_dist_AL_flt[j]):
                    N_AL = len(morph_dist_AL_flt[j])
                    dmin_AL = np.min(morph_dist_AL_ed, axis=0)
                else:
                    N_AL = len(morph_dist_AL_flt[i])
                    r1 = np.min(morph_dist_AL_ed, axis=0)
                    r2 = np.min(morph_dist_AL_ed, axis=1)
                    if np.sum(r1) < np.sum(r2):
                        dmin_AL = r1
                    else:
                        dmin_AL = r2
                
                morph_dist_MB_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_MB)), N_MB))
                morph_dist_LH_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_LH)), N_LH))
                morph_dist_AL_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_AL)), N_AL))
            
    np.save(r'./hemibrain/morph_dist_MB_r_neuprint.npy', morph_dist_MB_r_new)
    np.save(r'./hemibrain/morph_dist_LH_r_neuprint.npy', morph_dist_LH_r_new)
    np.save(r'./hemibrain/morph_dist_AL_r_neuprint.npy', morph_dist_AL_r_new)

morph_dist_MB_r_new = morph_dist_MB_r_new*8
morph_dist_LH_r_new = morph_dist_LH_r_new*8
morph_dist_AL_r_new = morph_dist_AL_r_new*8

MBdist_cluster_u_full_new = []
MBdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    MB_sq = morph_dist_MB_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    MB_sq_tri = MB_sq[np.triu_indices_from(MB_sq, k=1)]
    MB_nc = np.delete(morph_dist_MB_r_new[glo_lbs[i]:glo_lbs[i+1]], np.arange(glo_lbs[i], glo_lbs[i+1]))
        
    if len(MB_sq_tri) > 0:
        MBdist_cluster_u_full_new.append(MB_sq_tri)
    else:
        MBdist_cluster_u_full_new.append([])
    MBdist_noncluster_u_full_new.append(MB_nc.flatten())

MBdist_cluster_u_full_flat_new = [item for sublist in MBdist_cluster_u_full_new for item in sublist]
MBdist_noncluster_u_full_flat_new = [item for sublist in MBdist_noncluster_u_full_new for item in sublist]

LHdist_cluster_u_full_new = []
LHdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    LH_sq = morph_dist_LH_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    LH_sq_tri = LH_sq[np.triu_indices_from(LH_sq, k=1)]
    LH_nc = np.delete(morph_dist_LH_r_new[glo_lbs[i]:glo_lbs[i+1]], np.arange(glo_lbs[i], glo_lbs[i+1]))
        
    if len(LH_sq_tri) > 0:
        LHdist_cluster_u_full_new.append(LH_sq_tri)
    else:
        LHdist_cluster_u_full_new.append([])
    LHdist_noncluster_u_full_new.append(LH_nc.flatten())

LHdist_cluster_u_full_flat_new = [item for sublist in LHdist_cluster_u_full_new for item in sublist]
LHdist_noncluster_u_full_flat_new = [item for sublist in LHdist_noncluster_u_full_new for item in sublist]

ALdist_cluster_u_full_new = []
ALdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    AL_sq = morph_dist_AL_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    AL_sq_tri = AL_sq[np.triu_indices_from(AL_sq, k=1)]
    AL_nc = np.delete(morph_dist_AL_r_new[glo_lbs[i]:glo_lbs[i+1]], np.arange(glo_lbs[i], glo_lbs[i+1]))
    
    if len(AL_sq_tri) > 0:
        ALdist_cluster_u_full_new.append(AL_sq_tri)
    else:
        ALdist_cluster_u_full_new.append([])
    ALdist_noncluster_u_full_new.append(AL_nc.flatten())

ALdist_cluster_u_full_flat_new = [item for sublist in ALdist_cluster_u_full_new for item in sublist]
ALdist_noncluster_u_full_flat_new = [item for sublist in ALdist_noncluster_u_full_new for item in sublist]


print("MB cluster Mean: " + str(np.mean(MBdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(MBdist_cluster_u_full_flat_new)))
print("MB noncluster Mean: " + str(np.mean(MBdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(MBdist_noncluster_u_full_flat_new)))

print("LH cluster Mean: " + str(np.mean(LHdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(LHdist_cluster_u_full_flat_new)))
print("LH noncluster Mean: " + str(np.mean(LHdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(LHdist_noncluster_u_full_flat_new)))

print("AL cluster Mean: " + str(np.mean(ALdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(ALdist_cluster_u_full_flat_new)))
print("AL noncluster Mean: " + str(np.mean(ALdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(ALdist_noncluster_u_full_flat_new)))


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

#%% d_intra, d_inter, and lambda calculation

MBtest_cl = []
MBtest_ncl = []
MBtest_cl_std = []
MBtest_ncl_std = []
for i in range(len(MBdist_cluster_u_full_new)):
    MBtest_cl.append(np.mean(MBdist_cluster_u_full_new[i]))
    MBtest_cl_std.append(np.std(MBdist_cluster_u_full_new[i]))
for i in range(len(MBdist_noncluster_u_full_new)):
    MBtest_ncl.append(np.mean(MBdist_noncluster_u_full_new[i]))
    MBtest_ncl_std.append(np.std(MBdist_noncluster_u_full_new[i]))
    
LHtest_cl = []
LHtest_ncl = []
LHtest_cl_std = []
LHtest_ncl_std = []
for i in range(len(LHdist_cluster_u_full_new)):
    LHtest_cl.append(np.mean(LHdist_cluster_u_full_new[i]))
    LHtest_cl_std.append(np.std(LHdist_cluster_u_full_new[i]))
for i in range(len(LHdist_noncluster_u_full_new)):
    LHtest_ncl.append(np.mean(LHdist_noncluster_u_full_new[i]))
    LHtest_ncl_std.append(np.std(LHdist_noncluster_u_full_new[i]))

ALtest_cl = []
ALtest_ncl = []
ALtest_cl_std = []
ALtest_ncl_std = []
for i in range(len(ALdist_cluster_u_full_new)):
    ALtest_cl.append(np.mean(ALdist_cluster_u_full_new[i]))
    ALtest_cl_std.append(np.std(ALdist_cluster_u_full_new[i]))
for i in range(len(ALdist_noncluster_u_full_new)):
    ALtest_ncl.append(np.mean(ALdist_noncluster_u_full_new[i]))
    ALtest_ncl_std.append(np.std(ALdist_noncluster_u_full_new[i]))
    
MBtest_cl = np.nan_to_num(MBtest_cl)
MBtest_ncl = np.nan_to_num(MBtest_ncl)
LHtest_cl = np.nan_to_num(LHtest_cl)
LHtest_ncl = np.nan_to_num(LHtest_ncl)
ALtest_cl = np.nan_to_num(ALtest_cl)
ALtest_ncl = np.nan_to_num(ALtest_ncl)

MBtest_cl_std = np.nan_to_num(MBtest_cl_std)
MBtest_ncl_std = np.nan_to_num(MBtest_ncl_std)
LHtest_cl_std = np.nan_to_num(LHtest_cl_std)
LHtest_ncl_std = np.nan_to_num(LHtest_ncl_std)
ALtest_cl_std = np.nan_to_num(ALtest_cl_std)
ALtest_ncl_std = np.nan_to_num(ALtest_ncl_std)

ALtest_idx = np.where(np.array(ALtest_cl) >= 0)[0]
LHtest_idx = np.where(np.array(LHtest_cl) >= 0)[0]
MBtest_idx = np.where(np.array(MBtest_cl) >= 0)[0]

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

attavdict1 = {'DL2d': '#028e00', 'DL2v': '#028e00', 'VL1': '#028e00', 'VL2a': '#028e00', 'VM1': '#028e00', 'VM4': '#028e00', 'VC5': '#028e00',
             'DM1': '#7acb2f', 'DM4': '#7acb2f', 'DM5': '#7acb2f', 'DM6': '#7acb2f', 'VA4': '#7acb2f', 'VC2': '#7acb2f', 'VM7d': '#7acb2f',
             'DA3': '#00f700', 'DC1': '#00f700', 'DL1': '#00f700', 'VA3': '#00f700', 'VM2': '#00f700', 'VM5d': '#00f700', 'VM5v': '#00f700',  
             'DA4m': '#a3a3a3', 'VA7m': '#a3a3a3', 'VM7v': '#a3a3a3', 'VM6': '#a3a3a3', 
             'DM2': '#17d9f7', 'DP1l': '#17d9f7', 'DP1m': '#17d9f7', 'V': '#17d9f7', 'VA2': '#17d9f7', 'VC4': '#17d9f7', 'VL2p': '#17d9f7', 'VM3': '#17d9f7', 
             'D': '#f10000', 'DA2': '#f10000', 'DA4l': '#f10000', 'VC3': '#f10000', 'DC2': '#f10000', 'DC4': '#f10000', 'DL4': '#f10000', 'DL5': '#f10000', 'DM3': '#f10000',
             'VA6': '#e8f0be', 'VC1': '#e8f0be', 
             'VA5': '#b96d3d', 'VA7l': '#b96d3d', 
             'DA1': '#a200cb', 'DC3': '#a200cb', 'DL3': '#a200cb', 'VA1d': '#a200cb', 'VA1v': '#a200cb',
             'VP1d': '#ff40ff', 'VP1l': '#ff40ff', 'VP1m': '#ff40ff', 'VP2': '#ff40ff', 'VP3': '#ff40ff', 'VP4': '#ff40ff', 'VP5': '#ff40ff'}

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

attavdict3 = {'DL2d': '#025200', 'DL2v': '#025200', 'VL1': '#025200', 'VL2a': '#025200', 'VM1': '#025200', 'VM4': '#025200', 'VC5': '#025200',
             'DM1': '#3f8f2f', 'DM4': '#3f8f2f', 'DM5': '#3f8f2f', 'DM6': '#3f8f2f', 'VA4': '#3f8f2f', 'VC2': '#3f8f2f', 'VM7d': '#3f8f2f',
             'DA3': '#05a702', 'DC1': '#05a702', 'DL1': '#05a702', 'VA3': '#05a702', 'VM2': '#05a702', 'VM5d': '#05a702', 'VM5v': '#05a702',  
             'DA4m': '#676767', 'VA7m': '#676767', 'VM7v': '#676767', 'VM6': '#858585', 
             'DM2': '#17a0a7', 'DP1l': '#17a0a7', 'DP1m': '#17a0a7', 'V': '#17a0a7', 'VA2': '#17a0a7', 'VC4': '#17a0a7', 'VL2p': '#17a0a7', 'VM3': '#17a0a7', 
             'D': '#8d0000', 'DA2': '#8d0000', 'DA4l': '#8d0000', 'VC3': '#8d0000', 'DC2': '#8d0000', 'DC4': '#8d0000', 'DL4': '#8d0000', 'DL5': '#8d0000', 'DM3': '#8d0000',
             'VA6': '#b6b46e', 'VC1': '#b6b46e', 
             'VA5': '#592628', 'VA7l': '#592628', 
             'DA1': '#480071', 'DC3': '#480071', 'DL3': '#480071', 'VA1d': '#480071', 'VA1v': '#480071',
             'VP1d': '#c400c4', 'VP1l': '#c400c4', 'VP1m': '#c400c4', 'VP2': '#c400c4', 'VP3': '#c400c4', 'VP4': '#c400c4', 'VP5': '#c400c4'}

updatedxlabel = np.array(glo_list_new)[type_idx]

attavlist1 = []
attavlist2 = []
attavlist3 = []

for i in updatedxlabel:
    attavlist1.append(attavdict1[i])
    attavlist2.append(attavdict2[i])
    attavlist3.append(attavdict3[i])

type_idx = np.flip(type_idx)
updatedxlabel = np.flip(updatedxlabel)

attavlist1 = np.flip(attavlist1)
attavlist2 = np.flip(attavlist2)
attavlist3 = np.flip(attavlist3)

#%% Figure 5A

fig, ax = plt.subplots(figsize=(6,6))
labels = ['AL', 'MB calyx', 'LH']
x = np.arange(0, len(labels)+0.1, 1.5)
width = .3

cmeans = [np.mean(ALtest_cl[np.nonzero(ALtest_cl)]), 
          np.mean(MBtest_cl[np.nonzero(MBtest_cl)]), 
          np.mean(LHtest_cl[np.nonzero(LHtest_cl)])]
cerr = [np.std(ALtest_cl[np.nonzero(ALtest_cl)]),
        np.std(MBtest_cl[np.nonzero(MBtest_cl)]), 
        np.std(LHtest_cl[np.nonzero(LHtest_cl)])]
ncmeans = [np.mean(ALtest_ncl), 
           np.mean(MBtest_ncl), 
           np.mean(LHtest_ncl)]
ncerr = [np.std(ALtest_ncl),
         np.std(MBtest_ncl), 
         np.std(LHtest_ncl)]

lamb = [np.mean((ALtest_cl/ALtest_ncl)[np.nonzero(ALtest_cl)]), 
        np.mean((MBtest_cl/MBtest_ncl)[np.nonzero(MBtest_cl)]), 
        np.mean((LHtest_cl/LHtest_ncl)[np.nonzero(LHtest_cl)])]

lamberr = [np.std((ALtest_cl/ALtest_ncl)[np.nonzero(ALtest_cl)]), 
           np.std((MBtest_cl/MBtest_ncl)[np.nonzero(MBtest_cl)]), 
           np.std((LHtest_cl/LHtest_ncl)[np.nonzero(LHtest_cl)])]

ax2 = ax.twinx()
ax.scatter(np.repeat(x[0] - width-0.015, len(np.nonzero(ALtest_cl)[0])) + np.random.rand(len(np.nonzero(ALtest_cl)[0]))*(width/2)-(width/4),
           ALtest_cl[np.nonzero(ALtest_cl)], color='tab:blue', marker='.', s=25, label=r'$\bar{d}_{{\rm intra}}$')
ax.scatter(np.repeat(x[0], len(ALtest_ncl)) + np.random.rand(len(ALtest_ncl))*(width/2)-(width/4),
           ALtest_ncl, color='tab:orange', marker='.', s=25, label=r'$\bar{d}_{{\rm inter}}$')
ax2.scatter(np.repeat(x[0] + width+0.015, len(np.nonzero(ALtest_cl)[0])) + np.random.rand(len(np.nonzero(ALtest_cl)[0]))*(width/2)-(width/4),
           (ALtest_cl/ALtest_ncl)[np.nonzero(ALtest_cl)], color='tab:red', marker='.', s=25, label='$\lambda$')

ax.scatter(np.repeat(x[1] - width-0.015, len(np.nonzero(MBtest_cl)[0])) + np.random.rand(len(np.nonzero(MBtest_cl)[0]))*(width/2)-(width/4),
           MBtest_cl[np.nonzero(MBtest_cl)], color='tab:blue', marker='.', s=25)
ax.scatter(np.repeat(x[1], len(MBtest_ncl)) + np.random.rand(len(MBtest_ncl))*(width/2)-(width/4),
           MBtest_ncl, color='tab:orange', marker='.', s=25)
ax2.scatter(np.repeat(x[1] + width+0.015, len(np.nonzero(MBtest_cl)[0])) + np.random.rand(len(np.nonzero(MBtest_cl)[0]))*(width/2)-(width/4),
           (MBtest_cl/MBtest_ncl)[np.nonzero(MBtest_cl)], color='tab:red', marker='.', s=25)

ax.scatter(np.repeat(x[2] - width-0.015, len(np.nonzero(LHtest_cl)[0])) + np.random.rand(len(np.nonzero(LHtest_cl)[0]))*(width/2)-(width/4),
           LHtest_cl[np.nonzero(LHtest_cl)], color='tab:blue', marker='.', s=25)
ax.scatter(np.repeat(x[2], len(LHtest_ncl)) + np.random.rand(len(LHtest_ncl))*(width/2)-(width/4),
           LHtest_ncl, color='tab:orange', marker='.', s=25)
ax2.scatter(np.repeat(x[2] + width+0.015, len(np.nonzero(LHtest_cl)[0])) + np.random.rand(len(np.nonzero(LHtest_cl)[0]))*(width/2)-(width/4),
           (LHtest_cl/LHtest_ncl)[np.nonzero(LHtest_cl)], color='tab:red', marker='.', s=25)

ax.errorbar(x - width-0.015, cmeans, yerr=cerr, ls='none', color='k', capsize=5, marker='_', markersize=15)
ax.errorbar(x, ncmeans, yerr=ncerr, ls='none', color='k', capsize=5, marker='_', markersize=15)
ax2.errorbar(x + width+0.015, lamb, yerr=lamberr, ls='none', color='k', capsize=5, marker='_', markersize=15)

ax2.set_ylim(0, 1)
ax.set_ylabel(r'$\bar{d}_{{\rm intra}}$, $\bar{d}_{{\rm inter}}$ ($\mu$m)', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=1, fontsize=15)
plt.tight_layout()
plt.show()

#%% Figure 6B, Figure 5-Figure supplement 1

fig, ax = plt.subplots(1, 3, figsize=(8,12))
x = np.arange(len(MBtest_idx))
width = .275

ax[0].barh(x + width, ALtest_cl[type_idx], 
          width,
          capsize=5, label='Identical Glomerulus', color=np.array(attavlist1), alpha=0.5)
ax[0].barh(x , MBtest_cl[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist2), alpha=0.75)
ax[0].barh(x - width , LHtest_cl[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist3), alpha=1)
ax[0].set_yticks(x)
ax[0].set_title('$\\bar{d}_{intra,X}$', fontsize=25)
ax[0].set_yticklabels([])
ax[0].set_xticks(np.array([0, 25]))
ax[0].tick_params(axis="x", labelsize=15)
ax[0].set_ylim(x[0] - 1, x[-1] + 1)
ax[0].set_yticklabels(updatedxlabel, rotation=0, fontsize=15)
[t.set_color(i) for (i,t) in zip(np.flip(list(attavdict2.values())), ax[0].yaxis.get_ticklabels())]

ax[1].barh(x + width, ALtest_ncl[type_idx],
          width,
          capsize=5, label='Identical Glomerulus', color=np.array(attavlist1), alpha=0.5)
ax[1].barh(x, MBtest_ncl[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist2), alpha=0.75)
ax[1].barh(x - width, LHtest_ncl[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist3), alpha=1.)
ax[1].set_yticks(x)
ax[1].set_title('$\\bar{d}_{inter,X}$', fontsize=25)
ax[1].set_yticklabels([])
ax[1].set_xticks(np.array([0, 25]))
ax[1].tick_params(axis="x", labelsize=15)
ax[1].set_ylim(x[0] - 1, x[-1] + 1)

ax[2].barh(x + width, np.divide(ALtest_cl, ALtest_ncl)[type_idx], 
          width,
          capsize=5, label='Identical Glomerulus', color=np.array(attavlist1), alpha=0.5)
ax[2].barh(x, np.divide(MBtest_cl, MBtest_ncl)[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist2), alpha=0.75)
ax[2].barh(x - width, np.divide(LHtest_cl, LHtest_ncl)[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist3), alpha=1.)
ax[2].set_yticks(x)
ax[2].set_title('$\\lambda_{X}$', fontsize=25)
ax[2].set_xticks(np.array([0, 0.5, 1]))
ax[2].set_yticklabels([])
ax[2].tick_params(axis="x", labelsize=15)
ax[2].set_ylim(x[0] - 1, x[-1] + 1)
plt.tight_layout()
plt.show()

#%% Figure 7-Figure supplement 1

n = np.nonzero(np.divide(LHtest_cl, LHtest_ncl)[type_idx])

comb = pd.DataFrame({'AL': np.divide(ALtest_cl, ALtest_ncl)[type_idx][n], 
                     'MB': np.divide(MBtest_cl, MBtest_ncl)[type_idx][n], 
                     'LH': np.divide(LHtest_cl, LHtest_ncl)[type_idx][n]})

fig, ax = plt.subplots(figsize=(4,4))
for i in range(len(type_idx)):
    if np.divide(LHtest_cl, LHtest_ncl)[type_idx[i]] != 0:
        plt.scatter(np.divide(LHtest_cl, LHtest_ncl)[type_idx[i]],
                    np.divide(ALtest_cl, ALtest_ncl)[type_idx[i]], 
                    color=attavdict2[updatedxlabel[i]])

coef = np.polyfit(comb['LH'], comb['AL'], 1)
poly1d_fn = np.poly1d(coef)

ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel(r'LH, $\lambda_{X}$', fontsize=15)
ax.set_ylabel(r'AL, $\lambda_{X}$', fontsize=15)
plt.show()


fig, ax = plt.subplots(figsize=(4,4))
for i in range(len(type_idx)):
    if np.divide(MBtest_cl, MBtest_ncl)[type_idx[i]] != 0:
        plt.scatter(np.divide(MBtest_cl, MBtest_ncl)[type_idx[i]],
                    np.divide(ALtest_cl, ALtest_ncl)[type_idx[i]], 
                    color=attavdict2[updatedxlabel[i]])

coef = np.polyfit(comb['MB'], comb['AL'], 1)
poly1d_fn = np.poly1d(coef)

ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel(r'MB calyx, $\lambda_{X}$', fontsize=15)
ax.set_ylabel(r'AL, $\lambda_{X}$', fontsize=15)
plt.show()

fig, ax = plt.subplots(figsize=(4,4))
for i in range(len(type_idx)):
    if np.divide(MBtest_cl, MBtest_ncl)[type_idx[i]] != 0:
        plt.scatter(np.divide(MBtest_cl, MBtest_ncl)[type_idx[i]],
                    np.divide(LHtest_cl, LHtest_ncl)[type_idx[i]], 
                    color=attavdict2[updatedxlabel[i]])
        
coef = np.polyfit(comb['MB'], comb['LH'], 1)
poly1d_fn = np.poly1d(coef)
plt.plot(np.arange(0.1, 1.5, 0.1), poly1d_fn(np.arange(0.1, 1.5, 0.1)), color='k', ls='--')

plt.text(0.8, 0.85, '$r=0.677$\n$p<0.0001$', color='k', fontsize=11)
ax.set_ylim(0, 1.5)
ax.set_xlim(0, 1.5)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel(r'MB calyx, $\lambda_{X}$', fontsize=15)
ax.set_ylabel(r'LH, $\lambda_{X}$', fontsize=15)
plt.show()


#%% Spatial proximity-based clustering

L_AL_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_AL_r_new), method='complete', optimal_ordering=True)
L_MB_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_MB_r_new), method='complete', optimal_ordering=True)
L_LH_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_LH_r_new), method='complete', optimal_ordering=True)


#%% Tree-cutting using dynamic cut tree hybrid method

from dynamicTreeCut import cutreeHybrid

ind_AL_dist = cutreeHybrid(L_AL_new_ind, scipy.spatial.distance.squareform(morph_dist_AL_r_new), minClusterSize=1)['labels']
ind_LH_dist = cutreeHybrid(L_LH_new_ind, scipy.spatial.distance.squareform(morph_dist_LH_r_new), minClusterSize=4)['labels']
ind_MB_dist = cutreeHybrid(L_MB_new_ind, scipy.spatial.distance.squareform(morph_dist_MB_r_new), minClusterSize=4)['labels']

ind_MB_dist = ind_MB_dist.astype(object)
ind_LH_dist = ind_LH_dist.astype(object)

# Reorganize the cluster label
ind_MB_dist[np.where(ind_MB_dist == 13)] = '1'
ind_MB_dist[np.where(ind_MB_dist == 11)] = '2'
ind_MB_dist[np.where(ind_MB_dist == 8)] = '3'
ind_MB_dist[np.where(ind_MB_dist == 7)] = '4'
ind_MB_dist[np.where(ind_MB_dist == 9)] = '5'
ind_MB_dist[np.where(ind_MB_dist == 1)] = '6'
ind_MB_dist[np.where(ind_MB_dist == 3)] = '7'
ind_MB_dist[np.where(ind_MB_dist == 10)] = '8'
ind_MB_dist[np.where(ind_MB_dist == 5)] = '9'
ind_MB_dist[np.where(ind_MB_dist == 2)] = '10'
ind_MB_dist[np.where(ind_MB_dist == 6)] = '11'
ind_MB_dist[np.where(ind_MB_dist == 4)] = '12'
ind_MB_dist[np.where(ind_MB_dist == 12)] = '13'
ind_MB_dist[np.where(ind_MB_dist == 14)] = '14'

ind_MB_dist[np.where(ind_MB_dist == '1')] = 1
ind_MB_dist[np.where(ind_MB_dist == '2')] = 2
ind_MB_dist[np.where(ind_MB_dist == '3')] = 3
ind_MB_dist[np.where(ind_MB_dist == '4')] = 4
ind_MB_dist[np.where(ind_MB_dist == '5')] = 5
ind_MB_dist[np.where(ind_MB_dist == '6')] = 6
ind_MB_dist[np.where(ind_MB_dist == '7')] = 7
ind_MB_dist[np.where(ind_MB_dist == '8')] = 8
ind_MB_dist[np.where(ind_MB_dist == '9')] = 9
ind_MB_dist[np.where(ind_MB_dist == '10')] = 10
ind_MB_dist[np.where(ind_MB_dist == '11')] = 11
ind_MB_dist[np.where(ind_MB_dist == '12')] = 12
ind_MB_dist[np.where(ind_MB_dist == '13')] = 13
ind_MB_dist[np.where(ind_MB_dist == '14')] = 14

ind_LH_dist[np.where(ind_LH_dist == 9)] = '1'
ind_LH_dist[np.where(ind_LH_dist == 11)] = '2'
ind_LH_dist[np.where(ind_LH_dist == 7)] = '3'
ind_LH_dist[np.where(ind_LH_dist == 4)] = '4'
ind_LH_dist[np.where(ind_LH_dist == 8)] = '5'
ind_LH_dist[np.where(ind_LH_dist == 2)] = '6'
ind_LH_dist[np.where(ind_LH_dist == 13)] = '7'
ind_LH_dist[np.where(ind_LH_dist == 6)] = '8'
ind_LH_dist[np.where(ind_LH_dist == 10)] = '9'
ind_LH_dist[np.where(ind_LH_dist == 3)] = '10'
ind_LH_dist[np.where(ind_LH_dist == 5)] = '11'
ind_LH_dist[np.where(ind_LH_dist == 12)] = '12'
ind_LH_dist[np.where(ind_LH_dist == 1)] = '13'

ind_LH_dist[np.where(ind_LH_dist == '1')] = 1
ind_LH_dist[np.where(ind_LH_dist == '2')] = 2
ind_LH_dist[np.where(ind_LH_dist == '3')] = 3
ind_LH_dist[np.where(ind_LH_dist == '4')] = 4
ind_LH_dist[np.where(ind_LH_dist == '5')] = 5
ind_LH_dist[np.where(ind_LH_dist == '6')] = 6
ind_LH_dist[np.where(ind_LH_dist == '7')] = 7
ind_LH_dist[np.where(ind_LH_dist == '8')] = 8
ind_LH_dist[np.where(ind_LH_dist == '9')] = 9
ind_LH_dist[np.where(ind_LH_dist == '10')] = 10
ind_LH_dist[np.where(ind_LH_dist == '11')] = 11
ind_LH_dist[np.where(ind_LH_dist == '12')] = 12
ind_LH_dist[np.where(ind_LH_dist == '13')] = 13


#%% Pearson's chisquare test

def cramers_v(contingency_matrix):
    chi_val, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_matrix)
    n = contingency_matrix.sum().sum()
    phi2 = chi_val/n
    r,k = contingency_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return chi_val, np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))), p_val

def generate_fix_sum_random_vec(limit, num_elem):
    v = np.zeros(num_elem)
    for i in range(limit):
        p = np.random.randint(num_elem)
        v[p] += 1
    
    return v

a2 = pd.crosstab(glo_list_neuron_new[glo_idx_flat], ind_MB_dist)
a3 = pd.crosstab(glo_list_neuron_new[glo_idx_flat], ind_LH_dist)

print("The output is in order of: chi-square value, Cramer's V, and p-value")
print('Glomerular Labels vs C^MB')
print(cramers_v(a2))
print('Glomerular Labels vs C^LH')
print(cramers_v(a3))

print('C^MB vs C^LH')
a0 = pd.crosstab(ind_LH_dist, ind_MB_dist)
print(cramers_v(a0))

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

print('Monte Carlo chi-square value: Glomerular Labels vs C^MB (mean, std)')
print(np.mean(p2), np.std(p2))
print('Monte Carlo chi-square value: Glomerular Labels vs C^LH (mean, std)')
print(np.mean(p3), np.std(p3))

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

for i in glo_list_neuron_new[glo_idx_flat]:
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

a5 = pd.crosstab(grp1, ind_MB_dist)
a6 = pd.crosstab(grp1, ind_LH_dist)

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

print('Odor Type vs C^MB')
print(cramers_v(a5))
print('Odor Type vs C^LH')
print(cramers_v(a6))

print('Monte Carlo chi-square value: Odor Type vs C^MB (mean, std)')
print(np.mean(p5), np.std(p5))
print('Monte Carlo chi-square value: Odor Type vs C^LH (mean, std)')
print(np.mean(p6), np.std(p6))

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

for i in glo_list_neuron_new[glo_idx_flat]:
    if odor_dict2[i] == '#d62728':
        grp2.append(1)
    elif odor_dict2[i] == '#2ca02c':
        grp2.append(2)
    else:
        grp2.append(3)

grp2 = np.array(grp2)


a8 = pd.crosstab(grp2, ind_MB_dist)
a9 = pd.crosstab(grp2, ind_LH_dist)

orig8 = np.array(a8)

p8 = []

for i in range(1000):
    shu8 = np.zeros(np.shape(orig8), dtype=int)
    for j in range(len(orig8)):
        shu8[j] = generate_fix_sum_random_vec(np.sum(orig8[j]), len(orig8[j]))
    while len(np.where(np.sum(shu8, axis=0) == 0)[0]) > 0:
        shu8 = np.zeros(np.shape(orig8), dtype=int)
        for j in range(len(orig8)):
            shu8[j] = generate_fix_sum_random_vec(np.sum(orig8[j]), len(orig8[j]))
    shu8 = pd.DataFrame(shu8)
    shu8.index = a8.index
    shu8.columns = a8.columns
    a,b,c = cramers_v(shu8)
    
    p8.append(a)

orig9 = np.array(a9)

p9 = []

for i in range(1000):
    shu9 = np.zeros(np.shape(orig9), dtype=int)
    for j in range(len(orig9)):
        shu9[j] = generate_fix_sum_random_vec(np.sum(orig9[j]), len(orig9[j]))
    while len(np.where(np.sum(shu9, axis=0) == 0)[0]) > 0:
        shu9 = np.zeros(np.shape(orig9), dtype=int)
        for j in range(len(orig9)):
            shu9[j] = generate_fix_sum_random_vec(np.sum(orig9[j]), len(orig9[j]))
    shu9 = pd.DataFrame(shu9)
    shu9.index = a9.index
    shu9.columns = a9.columns
    a,b,c = cramers_v(shu9)
    
    p9.append(a)

print('Odor valence vs C^MB')
print(cramers_v(a8))
print('Odor valence vs C^LH')
print(cramers_v(a9))

print('Monte Carlo chi-square value: Odor valence vs C^MB (mean, std)')
print(np.mean(p8), np.std(p8))
print('Monte Carlo chi-square value: Odor valence vs C^LH (mean, std)')
print(np.mean(p9), np.std(p9))

#%% Mutual information study

mi_sample_MB_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_MB_dist))), len(glo_idx_flat))
    b = np.random.choice(np.arange(len(np.unique(ind_LH_dist))), len(glo_idx_flat))
    mi_sample_MB_LH_d.append(sklearn.metrics.mutual_info_score(a, b))

mi_sample_MB_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_MB_dist))), len(glo_idx_flat))
    mi_sample_MB_d.append(sklearn.metrics.mutual_info_score(glo_list_neuron_new[glo_idx_flat], a))

mi_sample_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_LH_dist))), len(glo_idx_flat))
    mi_sample_LH_d.append(sklearn.metrics.mutual_info_score(glo_list_neuron_new[glo_idx_flat], a))

print('Glomerular Labels vs C^MB')
print(sklearn.metrics.mutual_info_score(glo_list_neuron_new[glo_idx_flat], ind_MB_dist))
print('Glomerular Labels vs C^LH')
print(sklearn.metrics.mutual_info_score(glo_list_neuron_new[glo_idx_flat], ind_LH_dist))

print('C^MB vs C^LH')
print(sklearn.metrics.mutual_info_score(ind_LH_dist, ind_MB_dist))

print('Randomly sampled mean, std - Glomerular Labels vs C^MB')
print(np.mean(mi_sample_MB_LH_d), np.std(mi_sample_MB_LH_d))
print('Randomly sampled mean, std - Glomerular Labels vs C^LH')
print(np.mean(mi_sample_MB_d), np.std(mi_sample_MB_d))
print('Randomly sampled mean, std - C^MB vs C^LH')
print(np.mean(mi_sample_LH_d), np.std(mi_sample_LH_d))

mi_sample_MB_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_MB_dist))), len(glo_idx_flat))
    mi_sample_MB_d.append(sklearn.metrics.mutual_info_score(grp1, a))

mi_sample_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_LH_dist))), len(glo_idx_flat))
    mi_sample_LH_d.append(sklearn.metrics.mutual_info_score(grp1, a))

print('Odor Type vs C^MB')
print(sklearn.metrics.mutual_info_score(grp1, ind_MB_dist))
print('Odor Type vs C^LH')
print(sklearn.metrics.mutual_info_score(grp1, ind_LH_dist))

print('Randomly sampled mean, std - Odor Type vs C^MB')
print(np.mean(mi_sample_MB_d), np.std(mi_sample_MB_d))
print('Randomly sampled mean, std - Odor Type vs C^LH')
print(np.mean(mi_sample_LH_d), np.std(mi_sample_LH_d))

mi_sample_MB_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_MB_dist))), len(glo_idx_flat))
    mi_sample_MB_d.append(sklearn.metrics.mutual_info_score(grp2, a))

mi_sample_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(len(np.unique(ind_LH_dist))), len(glo_idx_flat))
    mi_sample_LH_d.append(sklearn.metrics.mutual_info_score(grp2, a))

print('Odor valence vs C^MB')
print(sklearn.metrics.mutual_info_score(grp2, ind_MB_dist))
print('Odor valence vs C^LH')
print(sklearn.metrics.mutual_info_score(grp2, ind_LH_dist))

print('Randomly sampled mean, std - Odor valence vs C^MB')
print(np.mean(mi_sample_MB_d), np.std(mi_sample_MB_d))
print('Randomly sampled mean, std - Odor valence vs C^LH')
print(np.mean(mi_sample_LH_d), np.std(mi_sample_LH_d))

#%% Figure 8-Figure supplement 1

odor = ['VM4', 'VC5', 'DL2v', 'DL2d', 'VL1', 'VM1', 'VL2a', 'VM7d', 'DM5',
       'DM6', 'DM1', 'DM4', 'VA4', 'VC2', 'VA3', 'VM5d', 'VM5v', 'DL1',
       'DA3', 'VM2', 'DC1', 'VM6', 'VA7m', 'VM7v', 'DA4m', 'VC4', 
       'DM2', 'VM3', 'V', 'DP1l', 'DP1m', 'VA2', 'VL2p', 'VC3', 'DC2', 'DA2',
       'D', 'DA4l', 'DC4', 'DL4', 'DL5', 'DM3', 'VA6', 'VC1', 'VA5',
       'VA7l', 'VA1v', 'DC3', 'DA1', 'VA1d', 'DL3', 'VP1m', 'VP5', 'VP1d',
       'VP1l', 'VP2', 'VP3', 'VP4']

columns_AL_new = []

for i in range(len(odor)):
    columns_AL_new.append(list(glo_list_new).index(odor[i]))

corrpoint = [0, 17, 29, 45, 54, 67, 85, 87, 91, 108, 120]
glist = ['VM4', 'DM5', 'VM5d', 'VA7m', 'V', 'DC2', 'VA6', 'VA5', 'VA1v', 'VP2']

glo_list_cluster = np.array(glo_list_new)[columns_AL_new]
glo_len_cluster = np.array(glo_len)[columns_AL_new]

glo_idx_cluster_dist = []
for i in range(len(glo_list_new)):
    taridx = glo_idx[np.argwhere(np.array(glo_list_new)[columns_AL_new][i] == np.array(glo_list_new))[0][0]]
    temp = []
    for j in taridx:
        temp.append(glo_idx_flat.index(j))
    glo_idx_cluster_dist.append(temp)

glo_idx_cluster_dist_flat = [item for sublist in glo_idx_cluster_dist for item in sublist]

ct_AL_glo_temp_dist = np.zeros((len(np.unique(ind_AL_dist)), len(glo_idx_flat)))
ct_LH_glo_temp_dist = np.zeros((len(np.unique(ind_LH_dist)), len(glo_idx_flat)))
ct_MB_glo_temp_dist = np.zeros((len(np.unique(ind_MB_dist)), len(glo_idx_flat)))

ix = 0

for i in range(len(glo_idx)):
    for j in range(len(glo_idx[i])):
        ct_AL_glo_temp_dist[ind_AL_dist[ix]-1][ix] = 1
        ct_LH_glo_temp_dist[ind_LH_dist[ix]-1][ix] = 1
        ct_MB_glo_temp_dist[ind_MB_dist[ix]-1][ix] = 1
        ix += 1

ct_AL_glo_dist = []
ct_LH_glo_dist = []
ct_MB_glo_dist = []

ct_AL_glo_dist.append(np.array(ct_AL_glo_temp_dist)[:,glo_idx_cluster_dist_flat])
ct_LH_glo_dist.append(np.array(ct_LH_glo_temp_dist)[:,glo_idx_cluster_dist_flat])
ct_MB_glo_dist.append(np.array(ct_MB_glo_temp_dist)[:,glo_idx_cluster_dist_flat])

glo_len_cluster = np.array(glo_len)[columns_AL_new]
glo_lb_cluster = [sum(glo_len_cluster[0:i]) for i in range(len(glo_len_cluster)+1)]
glo_lb_cluster_s = np.subtract(glo_lb_cluster, glo_lb_cluster[0])
glo_float_cluster = np.divide(glo_lb_cluster_s, glo_lb_cluster_s[-1])

fig = plt.figure(figsize=(12,0.15*len(np.unique(ind_LH_dist))))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax1.set_xticks(np.arange(len(glo_list_neuron_new))+0.45, minor=False)
ax1.set_yticks(np.arange(len(np.unique(ind_LH_dist)))-0.5, minor=False)
ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.25)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
for i in range(len(corrpoint)-1):
    cmap = matplotlib.colors.ListedColormap(['w', odor_dict[glist[i]]])
    a = copy.deepcopy(ct_LH_glo_dist[0])
    a[:,:corrpoint[i]] = None
    a[:,corrpoint[i+1]:] = None
    im = plt.imshow(a, cmap=cmap, aspect='auto')
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
ax2.set_xticks(glo_float_cluster)
ax3.set_yticks(np.arange(np.max(ind_LH_dist)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(np.max(ind_LH_dist)+1) + 0.5)))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add('$C_{', 
                                                                             np.char.add(np.arange(1,len(np.unique(ind_LH_dist))+1).astype(str),'}^{LH}$'))))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=6, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
for i in range(len(glo_list_cluster)):
    ax2.text(((glo_float_cluster[1:] + glo_float_cluster[:-1])/2-0.003)[i], -2.5, 
              glo_list_cluster[i], rotation=90, fontsize=6, color=odor_dict[glo_list_cluster[i]])
plt.show()


fig = plt.figure(figsize=(12,0.15*len(np.unique(ind_MB_dist))))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax1.set_xticks(np.arange(len(glo_list_neuron_new))+0.45, minor=False)
ax1.set_yticks(np.arange(len(np.unique(ind_MB_dist)))-0.5, minor=False)
ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.25)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
for i in range(len(corrpoint)-1):
    cmap = matplotlib.colors.ListedColormap(['w', odor_dict[glist[i]]])
    a = copy.deepcopy(ct_MB_glo_dist[0])
    a[:,:corrpoint[i]] = None
    a[:,corrpoint[i+1]:] = None
    im = plt.imshow(a, cmap=cmap, aspect='auto')
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
ax2.set_xticks(glo_float_cluster)
ax3.set_yticks(np.arange(np.max(ind_MB_dist)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(np.max(ind_MB_dist)+1) + 0.5)))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add('$C_{', 
                                                                             np.char.add(np.arange(1,len(np.unique(ind_MB_dist))+1).astype(str),'}^{MB}$'))))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=6, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
for i in range(len(glo_list_cluster)):
    ax2.text(((glo_float_cluster[1:] + glo_float_cluster[:-1])/2-0.003)[i], -2.5, 
               glo_list_cluster[i], rotation=90, fontsize=6, color=odor_dict[glo_list_cluster[i]])
plt.show()

