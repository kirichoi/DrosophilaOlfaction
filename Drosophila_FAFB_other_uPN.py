# -*- coding: utf-8 -*-
"""
Olfactory responses of Drosophila are encoded in the organization of projection neurons

Kiri Choi, Won Kyu Kim, Changbong Hyeon
School of Computational Sciences, Korea Institute for Advanced Study, Seoul 02455, Korea

This script reproduces figures based on the FAFB dataset that uses uPNs that
does not innervate all three neuropils
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.cluster
import scipy.optimize
from collections import Counter
import copy

os.chdir(os.path.dirname(__file__))

PATH = r'./FAFB/FAFB_swc' # Path to .swc files
SEED = 1234

glo_info = pd.read_csv('./1-s2.0-S0960982220308587-mmc4.csv') # Path to glomerulus label information
glo_info = glo_info.drop(66) # neuron id 1356477 nonexistent
uPNididx = np.where(glo_info['PN_type'] == 'uPN')[0]
mPNididx = np.where(glo_info['PN_type'] == 'mPN')[0]
uPNid = glo_info['skeleton_id'].iloc[uPNididx]
all_innerv_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 
                  19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34,
                  35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 
                  52, 53, 55, 56, 57, 58, 59, 60, 61, 63, 66, 67, 68, 69, 70, 
                  72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 
                  89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 
                  103, 104, 105, 106, 107, 111, 113, 114, 115, 116, 118, 119, 
                  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                  132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
                  148, 149, 151, 154, 155, 156, 158]
uPNididx_old = uPNididx[all_innerv_idx]
uPNid_old = glo_info['skeleton_id'].iloc[uPNididx_old]

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

r_d_10 = -10
r_rad_10 = np.radians(r_d_10)
r_10 = np.array([0, 1, 0])
r_vec_10 = r_rad_10 * r_10
rot10 = Rotation.from_rotvec(r_vec_10)

r_d_15 = -15
r_rad_15 = np.radians(r_d_15)
r_15 = np.array([0, 1, 0])
r_vec_15 = r_rad_15 * r_15
rot15 = Rotation.from_rotvec(r_vec_15)

r_d_25 = -25
r_rad_25 = np.radians(r_d_25)
r_25 = np.array([0, 1, 0])
r_vec_25 = r_rad_25 * r_25
rot25 = Rotation.from_rotvec(r_vec_25)

r_d_20 = -20
r_rad_20 = np.radians(r_d_20)
r_20 = np.array([0, 1, 0])
r_vec_20 = r_rad_20 * r_20
rot20 = Rotation.from_rotvec(r_vec_20)

r_d_30 = -30
r_rad_30 = np.radians(r_d_30)
r_30 = np.array([0, 1, 0])
r_vec_30 = r_rad_30 * r_30
rot30 = Rotation.from_rotvec(r_vec_30)

r_d_40 = -40
r_rad_40 = np.radians(r_d_40)
r_40 = np.array([0, 1, 0])
r_vec_40 = r_rad_40 * r_40
rot40 = Rotation.from_rotvec(r_vec_40)


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
    MorphData.morph_dist.append(np.divide(np.array(df[[2,3,4]]), 1000).tolist()) # Scale
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    BranchData.branchNum[f] = int(sum(i > 1 for i in ctrVal))
    branchInd = np.array(ctrKey)[np.where(np.array(ctrVal) > 1)[0]]

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
                
                # rotate -10 degrees
                branch_dist_temp2_rot10 = rot10.apply(branch_dist_temp2)
                # rotate -15 degrees
                branch_dist_temp2_rot15 = rot15.apply(branch_dist_temp2)
                # rotate -20 degrees
                branch_dist_temp2_rot20 = rot20.apply(branch_dist_temp2)
                # rotate -25 degrees
                branch_dist_temp2_rot25 = rot25.apply(branch_dist_temp2)
                # rotate -30 degrees
                branch_dist_temp2_rot30 = rot30.apply(branch_dist_temp2)
                
                if ((np.array(branch_dist_temp2_rot20)[:,0] > 315).all() and (np.array(branch_dist_temp2_rot25)[:,0] < 353).all() and
                    (np.array(branch_dist_temp2_rot20)[:,1] > 110).all() and (np.array(branch_dist_temp2_rot20)[:,1] < 170).all() and
                    (np.array(branch_dist_temp2_rot30)[:,2] > 360).all() and (np.array(branch_dist_temp2_rot30)[:,2] < 450).all()):
                    if f != 158:
                        MorphData.MBdist.append(branch_dist_temp2)
                        MorphData.MBdist_trk.append(f)
                        MBdist_per_n_temp.append(branch_dist_temp2)
                        length_MB_per_n.append(dist)
                        MB_branchTrk_temp.append(neu_branchTrk_temp)
                        MB_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                elif ((np.array(branch_dist_temp2_rot15)[:,0] < 350).all() and (np.array(branch_dist_temp2_rot20)[:,1] > 110).all() and
                      (np.array(branch_dist_temp2_rot20)[:,1] < 180).all() and (np.array(branch_dist_temp2_rot20)[:,2] > 240).all() and
                      (np.array(branch_dist_temp2_rot30)[:,2] < 375).all()):
                    MorphData.LHdist.append(branch_dist_temp2)
                    MorphData.LHdist_trk.append(f)
                    LHdist_per_n_temp.append(branch_dist_temp2)
                    length_LH_per_n.append(dist)
                    LH_branchTrk_temp.append(neu_branchTrk_temp)
                    LH_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                elif ((np.array(branch_dist_temp2_rot25)[:,0] > 345).all() and (np.array(branch_dist_temp2_rot20)[:,0] < 485).all() and 
                      (np.array(branch_dist_temp2_rot20)[:,1] > 170).all() and (np.array(branch_dist_temp2_rot20)[:,1] < 310).all() and
                      (np.array(branch_dist_temp2_rot10)[:,2] < 180).all()):
                    MorphData.ALdist.append(branch_dist_temp2)
                    MorphData.ALdist_trk.append(f)
                    ALdist_per_n_temp.append(branch_dist_temp2)
                    length_AL_per_n.append(dist)
                    AL_branchTrk_temp.append(neu_branchTrk_temp)
                    AL_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                
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
    

glo_idx = []
glo_idx_old = []

glo_list = np.unique(glo_info['top_glomerulus'].iloc[uPNididx])
glo_list_neuron = np.array(glo_info['top_glomerulus'].iloc[uPNididx])

glo_list_o = np.unique(glo_info['top_glomerulus'].iloc[uPNididx_old])
glo_list_neuron_o = np.array(glo_info['top_glomerulus'].iloc[uPNididx_old])

glo_len = []
glo_len_old = []

for i in range(len(glo_list)):
    a = list(np.where(glo_list_neuron[uPNididx] == glo_list[i])[0])
    if len(a) > 0:
        glo_idx.append(uPNididx[a])
        glo_len.append(len(a))
    
for i in range(len(glo_list)):
    a = list(np.where(glo_list_neuron[uPNididx_old] == glo_list[i])[0])
    if len(a) > 0:
        glo_idx_old.append(uPNididx_old[a])
        glo_len_old.append(len(a))
        
glo_lb = [sum(glo_len[0:i]) for i in range(len(glo_len)+1)]
glo_lbs = np.subtract(glo_lb, glo_lb[0])
glo_float = np.divide(glo_lbs, glo_lbs[-1])
glo_idx_flat = [item for sublist in glo_idx for item in sublist]

glo_lb_old = [sum(glo_len_old[0:i]) for i in range(len(glo_len_old)+1)]
glo_lbs_old = np.subtract(glo_lb_old, glo_lb_old[0])
glo_float_old = np.divide(glo_lbs_old, glo_lbs_old[-1])
glo_idx_flat_old = [item for sublist in glo_idx_old for item in sublist]

glo_lb_idx = []

for i in range(len(glo_lb)-1):
    glo_lb_idx.append(np.arange(glo_lb[i],glo_lb[i+1]))

morph_dist_MB = []
morph_dist_LH = []
morph_dist_AL = []

glo_idx_MB = []
glo_idx_LH = []
glo_idx_AL = []

for i in range(len(glo_list)):
    morph_dist_MB_temp = []
    morph_dist_LH_temp = []
    morph_dist_AL_temp = []
    glo_idx_MB_temp = [] 
    glo_idx_LH_temp = [] 
    glo_idx_AL_temp = [] 
    
    for j in range(len(glo_idx[i])):
        morph_dist_MB_temp2 = []
        morph_dist_LH_temp2 = []
        morph_dist_AL_temp2 = []
        
        for p in range(len(MorphData.morph_dist[glo_idx[i][j]])):
            
            # rotate -10 degrees
            branch_dist_temp2_rot10 = rot10.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate -15 degrees
            branch_dist_temp2_rot15 = rot15.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate -20 degrees
            branch_dist_temp2_rot20 = rot20.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate -25 degrees
            branch_dist_temp2_rot25 = rot25.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            # rotate -30 degrees
            branch_dist_temp2_rot30 = rot30.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))

            if ((np.array(branch_dist_temp2_rot20)[0] > 315).all() and (np.array(branch_dist_temp2_rot25)[0] < 353).all() and
                (np.array(branch_dist_temp2_rot20)[1] > 110).all() and (np.array(branch_dist_temp2_rot20)[1] < 170).all() and
                (np.array(branch_dist_temp2_rot30)[2] > 360).all() and (np.array(branch_dist_temp2_rot30)[2] < 450).all()):
                morph_dist_MB_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(branch_dist_temp2_rot15)[0] < 350).all() and (np.array(branch_dist_temp2_rot20)[1] > 110).all() and
                  (np.array(branch_dist_temp2_rot20)[1] < 180).all() and (np.array(branch_dist_temp2_rot20)[2] > 240).all() and
                  (np.array(branch_dist_temp2_rot30)[2] < 375).all()):
                morph_dist_LH_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(branch_dist_temp2_rot25)[0] > 345).all() and (np.array(branch_dist_temp2_rot20)[0] < 485).all() and 
                  (np.array(branch_dist_temp2_rot20)[1] > 170).all() and (np.array(branch_dist_temp2_rot20)[1] < 310).all() and
                  (np.array(branch_dist_temp2_rot10)[2] < 180).all()):
                morph_dist_AL_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
        
        if len(morph_dist_MB_temp2) > 0:
            if glo_idx[i][j] != 158:
                morph_dist_MB_temp.append(morph_dist_MB_temp2)
                glo_idx_MB_temp.append(glo_idx[i][j])
        
        if len(morph_dist_LH_temp2) > 0:
            morph_dist_LH_temp.append(morph_dist_LH_temp2)
            glo_idx_LH_temp.append(glo_idx[i][j])
        
        if len(morph_dist_AL_temp2) > 0:
            morph_dist_AL_temp.append(morph_dist_AL_temp2)
            glo_idx_AL_temp.append(glo_idx[i][j])
            
    morph_dist_MB.append(morph_dist_MB_temp)
    morph_dist_LH.append(morph_dist_LH_temp)
    morph_dist_AL.append(morph_dist_AL_temp)
    glo_idx_MB.append(glo_idx_MB_temp)
    glo_idx_LH.append(glo_idx_LH_temp)
    glo_idx_AL.append(glo_idx_AL_temp)

glo_idx_flat_AL = [item for sublist in glo_idx_AL for item in sublist]
glo_idx_flat_MB = [item for sublist in glo_idx_MB for item in sublist]
glo_idx_flat_LH = [item for sublist in glo_idx_LH for item in sublist]

ag = np.array(MorphData.ALdist_per_n, dtype=object)[glo_idx_flat_AL]
cg = np.array(MorphData.MBdist_per_n, dtype=object)[glo_idx_flat_MB]
lg = np.array(MorphData.LHdist_per_n, dtype=object)[glo_idx_flat_LH]

ag = [item for sublist in ag for item in sublist]
cg = [item for sublist in cg for item in sublist]
lg = [item for sublist in lg for item in sublist]

MorphData.ALdist_flat_glo = [item for sublist in ag for item in sublist]
MorphData.MBdist_flat_glo = [item for sublist in cg for item in sublist]
MorphData.LHdist_flat_glo = [item for sublist in lg for item in sublist]


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

for i in range(len(glo_idx_MB)):
    morph_dist_MB_CM_temp = []
    morph_dist_MB_std_temp = []
    
    for j in range(len(glo_idx_MB[i])):
        morph_dist_MB_CM_temp.append(np.average(np.array(morph_dist_MB[i][j]), axis=0))
        morph_dist_MB_std_temp.append(np.std(np.array(morph_dist_MB[i][j]), axis=0))
    
    morph_dist_MB_CM.append(morph_dist_MB_CM_temp)
    morph_dist_MB_std.append(morph_dist_MB_std_temp)
    

for i in range(len(glo_idx_LH)):
    morph_dist_LH_CM_temp = []
    morph_dist_LH_std_temp = []
    
    for j in range(len(glo_idx_LH[i])):
        morph_dist_LH_CM_temp.append(np.average(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_LH_std_temp.append(np.std(np.array(morph_dist_LH[i][j]), axis=0))

    morph_dist_LH_CM.append(morph_dist_LH_CM_temp)
    morph_dist_LH_std.append(morph_dist_LH_std_temp)
    
    
for i in range(len(glo_idx_AL)):
    morph_dist_AL_CM_temp = []
    morph_dist_AL_std_temp = []
    
    for j in range(len(glo_idx_AL[i])):
        morph_dist_AL_CM_temp.append(np.average(np.array(morph_dist_AL[i][j]), axis=0))
        morph_dist_AL_std_temp.append(np.std(np.array(morph_dist_AL[i][j]), axis=0))
    
    morph_dist_AL_CM.append(morph_dist_AL_CM_temp)
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
    morph_dist_MB_r_new = np.load(r'./FAFB/morph_dist_MB_r_FAFB_uPN.npy')
    morph_dist_LH_r_new = np.load(r'./FAFB/morph_dist_LH_r_FAFB_uPN.npy')
    morph_dist_AL_r_new = np.load(r'./FAFB/morph_dist_AL_r_FAFB_uPN.npy')
else:    
    morph_dist_MB_CM_flat = np.array([item for sublist in morph_dist_MB_CM for item in sublist])
    morph_dist_LH_CM_flat = np.array([item for sublist in morph_dist_LH_CM for item in sublist])
    morph_dist_AL_CM_flat = np.array([item for sublist in morph_dist_AL_CM for item in sublist])
    
    morph_dist_AL_r_new = np.zeros((len(morph_dist_AL_CM_flat), len(morph_dist_AL_CM_flat)))
    morph_dist_MB_r_new = np.zeros((len(morph_dist_MB_CM_flat), len(morph_dist_MB_CM_flat)))
    morph_dist_LH_r_new = np.zeros((len(morph_dist_LH_CM_flat), len(morph_dist_LH_CM_flat)))
    
    for i in range(len(morph_dist_AL_CM_flat)):
        for j in range(len(morph_dist_AL_CM_flat)):
            if i == j:
                morph_dist_AL_r_new[i][j] = 0
            elif morph_dist_AL_r_new[j][i] != 0:
                morph_dist_AL_r_new[i][j] = morph_dist_AL_r_new[j][i]
            else:
                morph_dist_AL_ed = scipy.spatial.distance.cdist(morph_dist_AL_flt[i], morph_dist_AL_flt[j])
                
                # NNmetric
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
                
                morph_dist_AL_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_AL)), N_AL))
        
    for i in range(len(morph_dist_MB_CM_flat)):
        for j in range(len(morph_dist_MB_CM_flat)):
            if i == j:
                morph_dist_MB_r_new[i][j] = 0
            elif morph_dist_MB_r_new[j][i] != 0:
                morph_dist_MB_r_new[i][j] = morph_dist_MB_r_new[j][i]
            else:
                morph_dist_MB_ed = scipy.spatial.distance.cdist(morph_dist_MB_flt[i], morph_dist_MB_flt[j])
                
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
                
                morph_dist_MB_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_MB)), N_MB))
                
    for i in range(len(morph_dist_LH_CM_flat)):
        for j in range(len(morph_dist_LH_CM_flat)):
            if i == j:
                morph_dist_LH_r_new[i][j] = 0
            elif morph_dist_LH_r_new[j][i] != 0:
                morph_dist_LH_r_new[i][j] = morph_dist_LH_r_new[j][i]
            else:
                morph_dist_LH_ed = scipy.spatial.distance.cdist(morph_dist_LH_flt[i], morph_dist_LH_flt[j])
                
                # NNmetric
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
                        
                morph_dist_LH_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_LH)), N_LH))
    
    np.save(r'./FAFB/morph_dist_AL_r_FAFB_uPN.npy', morph_dist_AL_r_new)
    np.save(r'./FAFB/morph_dist_MB_r_FAFB_uPN.npy', morph_dist_MB_r_new)
    np.save(r'./FAFB/morph_dist_LH_r_FAFB_uPN.npy', morph_dist_LH_r_new)
    
morph_dist_MB_r_old = np.load(r'./FAFB/morph_dist_MB_r_FAFB.npy')
morph_dist_LH_r_old = np.load(r'./FAFB/morph_dist_LH_r_FAFB.npy')
morph_dist_AL_r_old = np.load(r'./FAFB/morph_dist_AL_r_FAFB.npy')


#%% Updated glomerulus label

glo_list_neuron_new = copy.deepcopy(glo_list_neuron)
glo_list_new = copy.deepcopy(glo_list)
glo_list_neuron_old = copy.deepcopy(glo_list_neuron_o)
glo_list_old = copy.deepcopy(glo_list_o)

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

vc3m = np.where(glo_list_neuron_old == 'VC3m')
vc3l = np.where(glo_list_neuron_old == 'VC3l')
vc5 = np.where(glo_list_neuron_old == 'VC5')

glo_list_neuron_old[vc3m] = 'VC5'
glo_list_neuron_old[vc3l] = 'VC3'
glo_list_neuron_old[vc5] = 'VM6'

vc3m = np.where(glo_list_old == 'VC3m')
vc3l = np.where(glo_list_old == 'VC3l')
vc5 = np.where(glo_list_old == 'VC5')

glo_list_old[vc3m] = 'VC5'
glo_list_old[vc3l] = 'VC3'
glo_list_old[vc5] = 'VM6'

#%% Spatial proximity-based clustering

L_AL_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_AL_r_new), method='complete', optimal_ordering=True)
L_MB_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_MB_r_new), method='complete', optimal_ordering=True)
L_LH_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_LH_r_new), method='complete', optimal_ordering=True)

#%% Tree-cutting using dynamic cut tree hybrid method

from dynamicTreeCut import cutreeHybrid

ind_AL_dist = cutreeHybrid(L_AL_new_ind, scipy.spatial.distance.squareform(morph_dist_AL_r_new), minClusterSize=1)['labels']
ind_MB_dist = cutreeHybrid(L_MB_new_ind, scipy.spatial.distance.squareform(morph_dist_MB_r_new), minClusterSize=4, cutHeight=33)['labels']
ind_LH_dist = cutreeHybrid(L_LH_new_ind, scipy.spatial.distance.squareform(morph_dist_LH_r_new), minClusterSize=4)['labels']

ind_MB_dist = ind_MB_dist.astype(object)
ind_LH_dist = ind_LH_dist.astype(object)

# Reorganize the cluster label
ind_MB_dist[np.where(ind_MB_dist == 9)] = '1'
ind_MB_dist[np.where(ind_MB_dist == 5)] = '2'
ind_MB_dist[np.where(ind_MB_dist == 8)] = '3'
ind_MB_dist[np.where(ind_MB_dist == 7)] = '4'
ind_MB_dist[np.where(ind_MB_dist == 6)] = '5'
ind_MB_dist[np.where(ind_MB_dist == 1)] = '6'
ind_MB_dist[np.where(ind_MB_dist == 10)] = '7'
ind_MB_dist[np.where(ind_MB_dist == 3)] = '8'
ind_MB_dist[np.where(ind_MB_dist == 2)] = '9'
ind_MB_dist[np.where(ind_MB_dist == 4)] = '10'

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

ind_LH_dist[np.where(ind_LH_dist == 6)] = '1'
ind_LH_dist[np.where(ind_LH_dist == 8)] = '2'
ind_LH_dist[np.where(ind_LH_dist == 1)] = '3'
ind_LH_dist[np.where(ind_LH_dist == 5)] = '4'
ind_LH_dist[np.where(ind_LH_dist == 3)] = '5'
ind_LH_dist[np.where(ind_LH_dist == 2)] = '6'
ind_LH_dist[np.where(ind_LH_dist == 7)] = '7'
ind_LH_dist[np.where(ind_LH_dist == 10)] = '8'
ind_LH_dist[np.where(ind_LH_dist == 9)] = '9'
ind_LH_dist[np.where(ind_LH_dist == 14)] = '10'
ind_LH_dist[np.where(ind_LH_dist == 11)] = '11'
ind_LH_dist[np.where(ind_LH_dist == 4)] = '12'
ind_LH_dist[np.where(ind_LH_dist == 15)] = '13'
ind_LH_dist[np.where(ind_LH_dist == 12)] = '14'
ind_LH_dist[np.where(ind_LH_dist == 13)] = '15'

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
ind_LH_dist[np.where(ind_LH_dist == '14')] = 14
ind_LH_dist[np.where(ind_LH_dist == '15')] = 15

glo_len_MB = []

for i in range(len(glo_idx_MB)):
    glo_len_MB.append(len(glo_idx_MB[i]))

glo_len_LH = []

for i in range(len(glo_idx_LH)):
    glo_len_LH.append(len(glo_idx_LH[i]))

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


trans = np.array(glo_info['transmitter'])
trans_AL = trans[glo_idx_flat_AL]
trans_MB = trans[glo_idx_flat_MB]
trans_LH = trans[glo_idx_flat_LH]

trans_dict = {'ACh': '#2ca02c', 'GABA': '#d62728', 'Octopamine': '#ff7f0e', 'unknown': '#858585'}

#%% Figure S10

AL_innerv_idx = list(set(glo_idx_flat_AL) - set(all_innerv_idx))
MB_innerv_idx = list(set(glo_idx_flat_MB) - set(all_innerv_idx))
LH_innerv_idx = list(set(glo_idx_flat_LH) - set(all_innerv_idx))

LH_innerv_idx_in_d = []

for i in LH_innerv_idx:
    LH_innerv_idx_in_d.append(glo_idx_flat_LH.index(i))

morph_dist_LH_r_new_innerv = morph_dist_LH_r_new[LH_innerv_idx_in_d][:,LH_innerv_idx_in_d]

new_glo = np.array(glo_list_neuron_new)[np.array(glo_idx_flat_LH)[LH_innerv_idx_in_d]]
new_glo_unique = np.unique(new_glo)

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

LHtest_cl = []
LHtest_ncl = []
for i in range(len(LHdist_cluster_u_full_new)):
    LHtest_cl.append(np.mean(LHdist_cluster_u_full_new[i]))
for i in range(len(LHdist_noncluster_u_full_new)):
    LHtest_ncl.append(np.mean(LHdist_noncluster_u_full_new[i]))

LHtest_cl = np.nan_to_num(LHtest_cl)
LHtest_ncl = np.nan_to_num(LHtest_ncl)

LHdist_cluster_u_part_new = []
LHdist_noncluster_u_part_new = []

for i in range(len(glo_list_old)):
    LH_sq = morph_dist_LH_r_old[glo_lbs_old[i]:glo_lbs_old[i+1],glo_lbs_old[i]:glo_lbs_old[i+1]]
    LH_sq_tri = LH_sq[np.triu_indices_from(LH_sq, k=1)]
    LH_nc = np.delete(morph_dist_LH_r_old[glo_lbs_old[i]:glo_lbs_old[i+1]], np.arange(glo_lbs_old[i], glo_lbs_old[i+1]))
        
    if len(LH_sq_tri) > 0:
        LHdist_cluster_u_part_new.append(LH_sq_tri)
    else:
        LHdist_cluster_u_part_new.append([])
    LHdist_noncluster_u_part_new.append(LH_nc.flatten())

LHdist_cluster_u_part_flat_new = [item for sublist in LHdist_cluster_u_part_new for item in sublist]
LHdist_noncluster_u_part_flat_new = [item for sublist in LHdist_noncluster_u_part_new for item in sublist]

LHtest_cl_old = []
LHtest_ncl_old = []
for i in range(len(LHdist_cluster_u_part_new)):
    LHtest_cl_old.append(np.mean(LHdist_cluster_u_part_new[i]))
for i in range(len(LHdist_noncluster_u_part_new)):
    LHtest_ncl_old.append(np.mean(LHdist_noncluster_u_part_new[i]))

LHtest_cl_old = np.nan_to_num(LHtest_cl_old)
LHtest_ncl_old = np.nan_to_num(LHtest_ncl_old)


lnew = []
lold = []

for i in new_glo_unique:
    lnew.append(np.where(glo_list_new == i)[0][0])
    if i != 'VP3':
        lold.append(np.where(glo_list_old == i)[0][0])


fig, ax = plt.subplots(figsize=(6,6))
labels = ['Original 15\n homotypes', '27 uPNs added']
x = np.arange(0, len(labels)+0.1, 1.5)
width = .3

cmeans = [np.mean(LHtest_cl_old[lold][np.nonzero(LHtest_cl_old[lold])]),
          np.mean(LHtest_cl[lnew][np.nonzero(LHtest_cl[lnew])])]
cerr = [np.std(LHtest_cl_old[lold][np.nonzero(LHtest_cl_old[lold])]),
        np.std(LHtest_cl[lnew][np.nonzero(LHtest_cl[lnew])])]
ncmeans = [np.mean(LHtest_ncl_old[lold]),
           np.mean(LHtest_ncl[lnew])]
ncerr = [np.std(LHtest_ncl_old[lold]),
         np.std(LHtest_ncl[lnew])]

lamb = [np.mean((LHtest_cl_old[lold]/LHtest_ncl_old[lold])[np.nonzero(LHtest_cl_old[lold])]),
        np.mean((LHtest_cl[lnew]/LHtest_ncl[lnew])[np.nonzero(LHtest_cl[lnew])])]

lamberr = [np.std((LHtest_cl_old[lold]/LHtest_ncl_old[lold])[np.nonzero(LHtest_cl_old[lold])]),
           np.std((LHtest_cl[lnew]/LHtest_ncl[lnew])[np.nonzero(LHtest_cl[lnew])])]

ax2 = ax.twinx()

ax.scatter(np.repeat(x[1] - width-0.015, len(np.nonzero(LHtest_cl[lnew])[0])) + np.random.rand(len(np.nonzero(LHtest_cl[lnew])[0]))*(width/2)-(width/4),
           LHtest_cl[lnew][np.nonzero(LHtest_cl[lnew])], color='tab:blue', marker='.', s=25, label=r'$\bar{d}_{{\rm intra}}$')
ax.scatter(np.repeat(x[1], len(LHtest_ncl[lnew])) + np.random.rand(len(LHtest_ncl[lnew]))*(width/2)-(width/4),
           LHtest_ncl[lnew], color='tab:orange', marker='.', s=25, label=r'$\bar{d}_{{\rm inter}}$')
ax2.scatter(np.repeat(x[1] + width+0.015, len(np.nonzero(LHtest_cl[lnew])[0])) + np.random.rand(len(np.nonzero(LHtest_cl[lnew])[0]))*(width/2)-(width/4),
           (LHtest_cl[lnew]/LHtest_ncl[lnew])[np.nonzero(LHtest_cl[lnew])], color='tab:red', marker='.', s=25, label='$\lambda$')

ax.scatter(np.repeat(x[0] - width-0.015, len(np.nonzero(LHtest_cl_old[lold])[0])) + np.random.rand(len(np.nonzero(LHtest_cl_old[lold])[0]))*(width/2)-(width/4),
           LHtest_cl_old[lold][np.nonzero(LHtest_cl_old[lold])], color='tab:blue', marker='.', s=25)
ax.scatter(np.repeat(x[0], len(LHtest_ncl_old[lold])) + np.random.rand(len(LHtest_ncl_old[lold]))*(width/2)-(width/4),
           LHtest_ncl_old[lold], color='tab:orange', marker='.', s=25)
ax2.scatter(np.repeat(x[0] + width+0.015, len(np.nonzero(LHtest_cl_old[lold])[0])) + np.random.rand(len(np.nonzero(LHtest_cl_old[lold])[0]))*(width/2)-(width/4),
           (LHtest_cl_old[lold]/LHtest_ncl_old[lold])[np.nonzero(LHtest_cl_old[lold])], color='tab:red', marker='.', s=25)

ax.errorbar(x - width-0.015, cmeans, yerr=cerr, ls='none', color='k', capsize=5, marker='_', markersize=15)
ax.errorbar(x, ncmeans, yerr=ncerr, ls='none', color='k', capsize=5, marker='_', markersize=15)
ax2.errorbar(x + width+0.015, lamb, yerr=lamberr, ls='none', color='k', capsize=5, marker='_', markersize=15)

ax.set_ylim(-0.8564008082882784, 33.87873722683592)
ax2.set_ylim(0, 1)
ax.set_ylabel(r'$\bar{d}_{{\rm intra}}$, $\bar{d}_{{\rm inter}}$ ($\mu$m)', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2, fontsize=15)
plt.tight_layout()
plt.show()


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

a3 = pd.crosstab(glo_list_neuron_new[glo_idx_flat_LH], ind_LH_dist)

print("The output is in order of: chi-square value, Cramer's V, and p-value")
print('Glomerular Labels vs C^LH')
print(cramers_v(a3))

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

for i in glo_list_neuron_new[glo_idx_flat_LH]:
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

a6 = pd.crosstab(grp1, ind_LH_dist)

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

print('Odor Type vs C^LH')
print(cramers_v(a6))

print('Monte Carlo chi-square value: Odor Type vs C^LH (mean, std)')
print(np.mean(p6), np.std(p6))

odor_dict2 = {'DM6': '#d62728', 'VL2a': '#d62728', 'V': '#d62728', 'DL5': '#d62728', 'DM2': '#d62728', 'VM3': '#d62728',
            'DP1m': '#d62728', 'VL2p': '#d62728', 'DM3': '#d62728', 'VA3': '#2ca02c',
            'VA6': '#d62728', 'DM5': '#d62728', 'DL1': '#d62728', 'D': '#d62728',
            'DC1': '#d62728', 'DC2': '#d62728', 'VA7l': '#d62728', 'VA5': '#d62728',
            'DC3': '#d62728', 'DA2': '#d62728', 'DL4': '#d62728', 'DC4': '#d62728', 
            'DA4l': '#d62728', 'VC3': '#d62728','VA7m': '#d62728', 'DA4m': '#d62728', 'VM7d': '#2ca02c',
            'VA2': '#2ca02c', 'DM1': '#2ca02c', 'DM4': '#2ca02c', 'VM5v': '#2ca02c', 
            'VC2': '#2ca02c', 'VM2': '#2ca02c', 'VM5d': '#2ca02c',
            'DA3': '#2ca02c', 'VM4': '#2ca02c', 'VM1': '#2ca02c', 
            'VC1': '#2ca02c', 'VA1v': '#2ca02c', 'DA1': '#2ca02c', 'DL3': '#2ca02c',
            'VM7v': '#2ca02c', 'DP1l': '#000000', 'VC4': '#000000', 'VA4': '#000000', 
            'DL2d': '#000000', 'DL2v': '#000000', 'VC5': '#000000', 'VL1': '#000000', 'VA1d': '#000000',
            'VP1d': '#000000', 'VP1l': '#000000', 
            'VP1m': '#000000', 'VP2': '#000000', 'VP3': '#000000', 'VP4': '#000000', 'VP5': '#000000', 'VM6': '#000000'}

grp2 = []

for i in glo_list_neuron_new[glo_idx_flat_LH]:
    if odor_dict2[i] == '#d62728':
        grp2.append(1)
    elif odor_dict2[i] == '#2ca02c':
        grp2.append(2)
    else:
        grp2.append(3)

grp2 = np.array(grp2)

a9 = pd.crosstab(grp2, ind_LH_dist)

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

print('Odor valence vs C^LH')
print(cramers_v(a9))

print('Monte Carlo chi-square value: Odor valence vs C^LH (mean, std)')
print(np.mean(p9), np.std(p9))


#%% Visualization of LH clusters - Figure S9

# Choose the cluster number
c_n = 1

# Choose the view, 't' = top, 'f' = front
view = 't'

cidx = np.where(ind_LH_dist == c_n)[0]

idx_cluster = np.array(glo_idx_flat)[cidx]

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))

if view == 'f':
    vertcoor = np.array([[410.145, 136.235, 178.44 ],
                       [405.098, 155.704, 160.24 ],
                       [389.172, 165.974, 150.8  ],
                       [386.589, 166.984, 153.56 ],
                       [366.047, 173.932, 173.24 ],
                       [358.978, 175.413, 170.12 ],
                       [348.093, 175.295, 167.88 ],
                       [342.632, 171.318, 170.04 ],
                       [336.133, 166.405, 174.64 ],
                       [336.094, 166.353, 174.4  ],
                       [326.14 , 152.957, 173.84 ],
                       [324.989, 149.594, 176.96 ],
                       [324.651, 148.377, 178.48 ],
                       [324.566, 147.925, 178.24 ],
                       [323.874, 144.09 , 177.2  ],
                       [322.508, 135.391, 177.72 ],
                       [324.556, 131.489, 176.84 ],
                       [324.616, 131.375, 177.   ],
                       [330.235, 124.759, 183.44 ],
                       [337.854, 114.789, 169.56 ],
                       [338.823, 114.076, 169.56 ],
                       [351.567, 110.514, 176.32 ],
                       [365.639, 110.02 , 158.64 ],
                       [372.079, 110.011, 166.08 ],
                       [374.664, 110.009, 156.8  ],
                       [387.446, 110.006, 174.44 ],
                       [387.642, 110.006, 174.56 ],
                       [392.763, 110.177, 149.   ],
                       [395.588, 111.536, 150.96 ],
                       [401.842, 116.162, 147.68 ],
                       [403.037, 117.139, 153.56 ],
                       [405.206, 119.284, 160.8  ],
                       [409.584, 125.032, 179.16 ],
                       [409.792, 126.614, 177.12 ],
                       [410.099, 131.324, 178.28 ],
                       [410.106, 131.842, 178.68 ],
                       [410.145, 136.235, 178.44 ]])
    ax.plot(vertcoor[:,0], 
            vertcoor[:,1], 
            np.repeat(145, len(vertcoor)),
            color='k',
            lw=5)
else:
    vertcoor = np.array([[407.578, 137.245, 193.92 ],
                         [366.748, 141.212, 200.16 ],
                         [356.96 , 146.069, 201.48 ],
                         [344.433, 144.612, 199.44 ],
                         [342.463, 145.847, 198.68 ],
                         [332.547, 133.409, 193.72 ],
                         [330.423, 148.619, 192.24 ],
                         [328.313, 149.362, 189.   ],
                         [325.643, 148.824, 183.92 ],
                         [324.566, 147.925, 178.24 ],
                         [324.547, 147.626, 177.92 ],
                         [324.662, 146.765, 172.32 ],
                         [324.962, 145.319, 169.52 ],
                         [325.039, 145.374, 169.36 ],
                         [325.183, 145.425, 169.12 ],
                         [327.321, 138.327, 166.32 ],
                         [329.789, 137.013, 163.4  ],
                         [335.801, 133.013, 156.6  ],
                         [339.373, 136.254, 152.6  ],
                         [350.583, 120.18 , 143.76 ],
                         [384.953, 124.606, 154.72 ],
                         [389.226, 135.121, 157.44 ],
                         [398.503, 130.537, 164.04 ],
                         [407.578, 137.245, 193.92 ]])
    ax.plot(vertcoor[:,0], 
            np.repeat(175, len(vertcoor)),
            vertcoor[:,2], 
            color='k',
            lw=5)

n = 0
for i in glo_idx_flat:
    if i in idx_cluster:
        for j in range(len(MorphData.LHdist_per_n[i])):
            listOfPoints = MorphData.LHdist_per_n[i][j]
            for f in range(len(listOfPoints)-1):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], 
                          color=odor_dict[glo_list_neuron_new[glo_idx_flat][cidx][n]], lw=1.)
        n += 1
ax.axis('off')

if view == 'f':
    ax.view_init(elev=90., azim=-90)
else:
    ax.view_init(elev=0., azim=-90)

ax.set_xlim(320, 420)
ax.set_ylim(200, 100)
ax.set_zlim(130, 230)
ax.dist = 7
plt.show()


#%% Visualization of LH clusters comparing homotypes with and without 27 uPNs - Figure S11

# choose the homotype
g = 'DA1'

# Choose the view, 't' = top, 'f' = front
view = 't'

try:
    assert(g in new_glo_unique)
except:
    raise Exception('Enter a valid homotype from ' + str(new_glo_unique))

innerv_dict = {'old': '#b3b3b3', 'new': '#000000'}

idx = np.where(glo_list_new == g)[0][0]

idx_cluster = glo_idx_LH[idx]

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))

if view == 'f':
    vertcoor = np.array([[410.145, 136.235, 178.44 ],
                       [405.098, 155.704, 160.24 ],
                       [389.172, 165.974, 150.8  ],
                       [386.589, 166.984, 153.56 ],
                       [366.047, 173.932, 173.24 ],
                       [358.978, 175.413, 170.12 ],
                       [348.093, 175.295, 167.88 ],
                       [342.632, 171.318, 170.04 ],
                       [336.133, 166.405, 174.64 ],
                       [336.094, 166.353, 174.4  ],
                       [326.14 , 152.957, 173.84 ],
                       [324.989, 149.594, 176.96 ],
                       [324.651, 148.377, 178.48 ],
                       [324.566, 147.925, 178.24 ],
                       [323.874, 144.09 , 177.2  ],
                       [322.508, 135.391, 177.72 ],
                       [324.556, 131.489, 176.84 ],
                       [324.616, 131.375, 177.   ],
                       [330.235, 124.759, 183.44 ],
                       [337.854, 114.789, 169.56 ],
                       [338.823, 114.076, 169.56 ],
                       [351.567, 110.514, 176.32 ],
                       [365.639, 110.02 , 158.64 ],
                       [372.079, 110.011, 166.08 ],
                       [374.664, 110.009, 156.8  ],
                       [387.446, 110.006, 174.44 ],
                       [387.642, 110.006, 174.56 ],
                       [392.763, 110.177, 149.   ],
                       [395.588, 111.536, 150.96 ],
                       [401.842, 116.162, 147.68 ],
                       [403.037, 117.139, 153.56 ],
                       [405.206, 119.284, 160.8  ],
                       [409.584, 125.032, 179.16 ],
                       [409.792, 126.614, 177.12 ],
                       [410.099, 131.324, 178.28 ],
                       [410.106, 131.842, 178.68 ],
                       [410.145, 136.235, 178.44 ]])
    ax.plot(vertcoor[:,0], 
            vertcoor[:,1], 
            np.repeat(145, len(vertcoor)),
            color='k',
            lw=5)
else:
    vertcoor = np.array([[407.578, 137.245, 193.92 ],
                         [366.748, 141.212, 200.16 ],
                         [356.96 , 146.069, 201.48 ],
                         [344.433, 144.612, 199.44 ],
                         [342.463, 145.847, 198.68 ],
                         [332.547, 133.409, 193.72 ],
                         [330.423, 148.619, 192.24 ],
                         [328.313, 149.362, 189.   ],
                         [325.643, 148.824, 183.92 ],
                         [324.566, 147.925, 178.24 ],
                         [324.547, 147.626, 177.92 ],
                         [324.662, 146.765, 172.32 ],
                         [324.962, 145.319, 169.52 ],
                         [325.039, 145.374, 169.36 ],
                         [325.183, 145.425, 169.12 ],
                         [327.321, 138.327, 166.32 ],
                         [329.789, 137.013, 163.4  ],
                         [335.801, 133.013, 156.6  ],
                         [339.373, 136.254, 152.6  ],
                         [350.583, 120.18 , 143.76 ],
                         [384.953, 124.606, 154.72 ],
                         [389.226, 135.121, 157.44 ],
                         [398.503, 130.537, 164.04 ],
                         [407.578, 137.245, 193.92 ]])
    ax.plot(vertcoor[:,0], 
            np.repeat(175, len(vertcoor)),
            vertcoor[:,2], 
            color='k',
            lw=5)
    
    
for i in idx_cluster:
    if i in LH_innerv_idx:
        color = innerv_dict['new']
    else:
        color = innerv_dict['old']
        
    if i == 135:
        listOfPoints = morph_dist_LH[52][0]
        for f in range(len(listOfPoints)-1):
            morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
        
            ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], 
                      color=color, lw=1.)
    else:
        for j in range(len(MorphData.LHdist_per_n[i])):
            listOfPoints = MorphData.LHdist_per_n[i][j]
            for f in range(len(listOfPoints)-1):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], 
                          color=color, lw=1.)
ax.axis('off')

if view == 'f':
    ax.view_init(elev=90., azim=-90)
else:
    ax.view_init(elev=0., azim=-90)

ax.set_xlim(320, 420)
ax.set_ylim(200, 100)
ax.set_zlim(130, 230)
ax.dist = 7
plt.show()


