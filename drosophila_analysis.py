# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:35:46 2022

@author: user
"""

import os
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
from scipy.spatial.transform import Rotation
import pandas as pd
import scipy.optimize
from collections import Counter
import copy

os.chdir(os.path.dirname(__file__))

class Parameter:

    PATH = r'./TEMCA2/Skels connectome_mod'
    
    RUN = True
    SAVE = False
    PLOT = True
    RN = '1'
    
    SEED = 1234
    
    outputdir = './output_TEMCA2/RN_' + str(RN)

fp = [f for f in os.listdir(Parameter.PATH) if os.path.isfile(os.path.join(Parameter.PATH, f))]
fp = [os.path.join(Parameter.PATH, f) for f in fp]

fp.pop(17)

class MorphData():
    
    def __init__(self):
        self.morph_id = []
        self.morph_parent = []
        self.morph_dist = []
        self.neuron_id = []
        self.endP = []
        self.somaP = []
        self.calyxdist = []
        self.calyxdist_trk = []
        self.calyxdist_per_n = []
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
    length_calyx = []
    length_LH = []
    length_AL = []
    length_calyx_total = []
    length_LH_total = []
    length_AL_total = []
    
class BranchData:
    branchTrk = []
    branch_dist = []
    branchP = []
    calyx_branchTrk = []
    calyx_branchP = []
    calyx_endP = []
    LH_branchTrk = []
    LH_branchP = []
    LH_endP = []
    AL_branchTrk = []
    AL_branchP = []
    AL_endP = []
    branchNum = np.empty(len(fp))

np.random.seed(Parameter.SEED)

MorphData = MorphData()

r_d_x = -10
r_rad_x = np.radians(r_d_x)
r_x = np.array([0, 1, 0])
r_vec_x = r_rad_x * r_x
rotx = Rotation.from_rotvec(r_vec_x)

r_d_y = -25
r_rad_y = np.radians(r_d_y)
r_y = np.array([0, 1, 0])
r_vec_y = r_rad_y * r_y
roty = Rotation.from_rotvec(r_vec_y)

r_d_z = -40
r_rad_z = np.radians(r_d_z)
r_z = np.array([0, 1, 0])
r_vec_z = r_rad_z * r_z
rotz = Rotation.from_rotvec(r_vec_z)

for f in range(len(fp)):
    print(f, fp[f])
    morph_neu_id = []
    morph_neu_parent = []
    morph_neu_prox = []
    morph_neu_dist = []
    
    df = pd.read_csv(fp[f], delimiter=' ', header=None)
    
    MorphData.neuron_id.append(os.path.basename(fp[f]).split('.')[0])
    
    scall = int(df.iloc[np.where(df[6] == -1)[0]].values[0][0])
    MorphData.somaP.append(scall)
    
    MorphData.morph_id.append(df[0].tolist())
    MorphData.morph_parent.append(df[6].tolist())
    MorphData.morph_dist.append(np.divide(np.array(df[[2,3,4]]), 1000).tolist()) # Scale
    ctr = Counter(df[6].tolist())
    ctrVal = list(ctr.values())
    ctrKey = list(ctr.keys())
    BranchData.branchNum[f] = sum(i > 1 for i in ctrVal)
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
    
    calyxdist_per_n_temp = []
    LHdist_per_n_temp = []
    ALdist_per_n_temp = []
    length_calyx_per_n = []
    length_LH_per_n = []
    length_AL_per_n = []
    calyx_branchTrk_temp = []
    calyx_branchP_temp = []
    LH_branchTrk_temp = []
    LH_branchP_temp = []
    AL_branchTrk_temp = []
    AL_branchP_temp = []
    calyx_endP_temp = []
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
                
                # rotate -25 degrees on y-axis
                branch_dist_temp2_rot = roty.apply(branch_dist_temp2)
                
                # rotate -35 degrees on x-axis
                branch_dist_temp2_rot2 = rotx.apply(branch_dist_temp2)
                
                # rotate 50 degrees on z-axis
                branch_dist_temp2_rot3 = rotz.apply(branch_dist_temp2)
                
                if ((np.array(branch_dist_temp2_rot)[:,0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[:,0] < 426.14).all() and
                    (np.array(branch_dist_temp2_rot)[:,1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and
                    (np.array(branch_dist_temp2_rot3)[:,2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[:,2] < 496.22).all()):
                    MorphData.calyxdist.append(branch_dist_temp2)
                    MorphData.calyxdist_trk.append(f)
                    calyxdist_per_n_temp.append(branch_dist_temp2)
                    length_calyx_per_n.append(dist)
                    calyx_branchTrk_temp.append(neu_branchTrk_temp)
                    calyx_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                    if bPoint[bp] in list_end:
                        calyx_endP_temp.append(bPoint[bp])
                elif ((np.array(branch_dist_temp2_rot)[:,0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[:,1] > 176.68).all() and
                      (np.array(branch_dist_temp2_rot)[:,1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[:,2] > 286.78).all() and
                      (np.array(branch_dist_temp2_rot)[:,2] < 343.93).all()):
                    MorphData.LHdist.append(branch_dist_temp2)
                    MorphData.LHdist_trk.append(f)
                    LHdist_per_n_temp.append(branch_dist_temp2)
                    length_LH_per_n.append(dist)
                    LH_branchTrk_temp.append(neu_branchTrk_temp)
                    LH_branchP_temp.append(list(set(neu_branchTrk_temp) & set(branchInd)))
                    if bPoint[bp] in list_end:
                        LH_endP_temp.append(bPoint[bp])
                elif ((np.array(branch_dist_temp2_rot)[:,0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[:,0] < 533.42).all() and 
                      (np.array(branch_dist_temp2_rot)[:,1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[:,1] < 363.12).all() and
                      (np.array(branch_dist_temp2_rot2)[:,2] < 180.77).all()):
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
    
    MorphData.calyxdist_per_n.append(calyxdist_per_n_temp)
    MorphData.LHdist_per_n.append(LHdist_per_n_temp)
    MorphData.ALdist_per_n.append(ALdist_per_n_temp)
    LengthData.length_calyx.append(length_calyx_per_n)
    LengthData.length_LH.append(length_LH_per_n)
    LengthData.length_AL.append(length_AL_per_n)
    BranchData.calyx_branchTrk.append(calyx_branchTrk_temp)
    BranchData.calyx_branchP.append(np.unique([item for sublist in calyx_branchP_temp for item in sublist]).tolist())
    BranchData.LH_branchTrk.append(LH_branchTrk_temp)
    BranchData.LH_branchP.append(np.unique([item for sublist in LH_branchP_temp for item in sublist]).tolist())
    BranchData.AL_branchTrk.append(AL_branchTrk_temp)
    BranchData.AL_branchP.append(np.unique([item for sublist in AL_branchP_temp for item in sublist]).tolist())
    BranchData.calyx_endP.append(calyx_endP_temp)
    BranchData.LH_endP.append(LH_endP_temp)
    BranchData.AL_endP.append(AL_endP_temp)
    
#%%

glo_info = pd.read_excel(os.path.join(Parameter.PATH, '../all_skeletons_type_list_180919.xls'))

glo_list = []
glo_idx = []

for f in range(len(MorphData.neuron_id)):
    idx = np.where(glo_info.skid == int(MorphData.neuron_id[f]))[0][0]
    if 'glomerulus' in glo_info['old neuron name'][idx]:
        if glo_info['type'][idx] != 'unknown glomerulus': # One neuron in this glomerulus that does not project to LH
            if glo_info['type'][idx] == 'DP1l, VL2p': # Neuron with both DP1l and VL2p label
                glo_name = 'VL2p' # Neuron seems to have more similar spetrum as VL2p
            else:
                glo_name = glo_info['type'][idx]
                
            if glo_name in glo_list:
                glo_idx[glo_list.index(glo_name)].append(f)
            else:
                glo_list.append(glo_name)
                glo_idx.append([f])

glo_len = [len(arr) for arr in glo_idx]
glo_lb = [sum(glo_len[0:i]) for i in range(len(glo_len)+1)]
glo_lbs = np.subtract(glo_lb, glo_lb[0])
glo_float = np.divide(glo_lbs, glo_lbs[-1])
glo_idx_flat = [item for sublist in glo_idx for item in sublist]
glo_idx_flat.sort()

glo_list_neuron = np.repeat(glo_list, glo_len)
glo_lb_idx = []

for i in range(len(glo_lb)-1):
    glo_lb_idx.append(np.arange(glo_lb[i],glo_lb[i+1]))

morph_dist_calyx = []
morph_dist_LH = []
morph_dist_AL = []

for i in range(len(glo_list)):
    morph_dist_calyx_temp = []
    morph_dist_LH_temp = []
    morph_dist_AL_temp = []
    morph_dist_calyx_bp_temp = []
    morph_dist_LH_bp_temp = []
    morph_dist_AL_bp_temp = []
    morph_dist_calyx_ep_temp = []
    morph_dist_LH_ep_temp = []
    morph_dist_AL_ep_temp = []
    for j in range(len(glo_idx[i])):
        morph_dist_calyx_temp2 = []
        morph_dist_LH_temp2 = []
        morph_dist_AL_temp2 = []
        morph_dist_calyx_bp_temp2 = []
        morph_dist_LH_bp_temp2 = []
        morph_dist_AL_bp_temp2 = []
        morph_dist_calyx_ep_temp2 = []
        morph_dist_LH_ep_temp2 = []
        morph_dist_AL_ep_temp2 = []
        
        for p in range(len(MorphData.morph_dist[glo_idx[i][j]])):
            
            branch_dist_temp2_rot = roty.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            branch_dist_temp2_rot2 = rotx.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))
            branch_dist_temp2_rot3 = rotz.apply(np.array(MorphData.morph_dist[glo_idx[i][j]][p]))

            if ((np.array(branch_dist_temp2_rot)[0] > 353.95).all() and (np.array(branch_dist_temp2_rot)[0] < 426.14).all() and
                (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and
                (np.array(branch_dist_temp2_rot3)[2] > 434.08).all() and (np.array(branch_dist_temp2_rot3)[2] < 496.22).all()):
                morph_dist_calyx_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(branch_dist_temp2_rot)[0] < 353.95).all() and (np.array(branch_dist_temp2_rot)[1] > 176.68).all() and
                  (np.array(branch_dist_temp2_rot)[1] < 272.91).all() and (np.array(branch_dist_temp2_rot)[2] > 286.78).all() and
                  (np.array(branch_dist_temp2_rot)[2] < 343.93).all()):
                morph_dist_LH_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
            elif ((np.array(branch_dist_temp2_rot)[0] > 426.14).all() and (np.array(branch_dist_temp2_rot)[0] < 533.42).all() and 
                  (np.array(branch_dist_temp2_rot)[1] > 272.91).all() and (np.array(branch_dist_temp2_rot)[1] < 363.12).all() and
                  (np.array(branch_dist_temp2_rot2)[2] < 180.77).all()):
                morph_dist_AL_temp2.append(MorphData.morph_dist[glo_idx[i][j]][p])
        
        morph_dist_calyx_temp.append(morph_dist_calyx_temp2)
        morph_dist_LH_temp.append(morph_dist_LH_temp2)
        morph_dist_AL_temp.append(morph_dist_AL_temp2)
                
    morph_dist_calyx.append(morph_dist_calyx_temp)
    morph_dist_LH.append(morph_dist_LH_temp)
    morph_dist_AL.append(morph_dist_AL_temp)
    
cg = np.array(MorphData.calyxdist_per_n, dtype=object)[glo_idx_flat]    
lg = np.array(MorphData.LHdist_per_n, dtype=object)[glo_idx_flat]    
ag = np.array(MorphData.ALdist_per_n, dtype=object)[glo_idx_flat]    

cg = [item for sublist in cg for item in sublist]
lg = [item for sublist in lg for item in sublist]
ag = [item for sublist in ag for item in sublist]

MorphData.calyxdist_flat_glo = [item for sublist in cg for item in sublist]
MorphData.LHdist_flat_glo = [item for sublist in lg for item in sublist]
MorphData.ALdist_flat_glo = [item for sublist in ag for item in sublist]

#%%

morph_dist_calyx_CM = []
morph_dist_LH_CM = []
morph_dist_AL_CM = []

morph_dist_calyx_std = []
morph_dist_LH_std = []
morph_dist_AL_std = []

for i in range(len(morph_dist_AL)):
    morph_dist_calyx_CM_temp = []
    morph_dist_LH_CM_temp = []
    morph_dist_AL_CM_temp = []
    
    morph_dist_calyx_std_temp = []
    morph_dist_LH_std_temp = []
    morph_dist_AL_std_temp = []
    
    for j in range(len(morph_dist_AL[i])):
        morph_dist_calyx_CM_temp.append(np.average(np.array(morph_dist_calyx[i][j]), axis=0))
        morph_dist_LH_CM_temp.append(np.average(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_AL_CM_temp.append(np.average(np.array(morph_dist_AL[i][j]), axis=0))
        
        morph_dist_calyx_std_temp.append(np.std(np.array(morph_dist_calyx[i][j]), axis=0))
        morph_dist_LH_std_temp.append(np.std(np.array(morph_dist_LH[i][j]), axis=0))
        morph_dist_AL_std_temp.append(np.std(np.array(morph_dist_AL[i][j]), axis=0))
    
    morph_dist_calyx_CM.append(morph_dist_calyx_CM_temp)
    morph_dist_LH_CM.append(morph_dist_LH_CM_temp)
    morph_dist_AL_CM.append(morph_dist_AL_CM_temp)
    
    morph_dist_LH_std.append(morph_dist_LH_std_temp)
    morph_dist_calyx_std.append(morph_dist_calyx_std_temp)
    morph_dist_AL_std.append(morph_dist_AL_std_temp)
    
from scipy.spatial import ConvexHull

morph_dist_calyx_flt = [item for sublist in morph_dist_calyx for item in sublist]
morph_dist_calyx_flat = [item for sublist in morph_dist_calyx_flt for item in sublist]

mdcalyx_xmax = np.max(np.array(morph_dist_calyx_flat)[:,0])
mdcalyx_xmin = np.min(np.array(morph_dist_calyx_flat)[:,0])
mdcalyx_ymax = np.max(np.array(morph_dist_calyx_flat)[:,1])
mdcalyx_ymin = np.min(np.array(morph_dist_calyx_flat)[:,1])
mdcalyx_zmax = np.max(np.array(morph_dist_calyx_flat)[:,2])
mdcalyx_zmin = np.min(np.array(morph_dist_calyx_flat)[:,2])

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

hull_calyx = ConvexHull(np.array(morph_dist_calyx_flat))
calyx_vol = hull_calyx.volume
calyx_area = hull_calyx.area
calyx_density_l = np.sum(LengthData.length_calyx_total)/calyx_vol

hull_LH = ConvexHull(np.array(morph_dist_LH_flat))
LH_vol = hull_LH.volume
LH_area = hull_LH.area
LH_density_l = np.sum(LengthData.length_LH_total)/LH_vol

hull_AL = ConvexHull(np.array(morph_dist_AL_flat))
AL_vol = hull_AL.volume
AL_area = hull_AL.area
AL_density_l = np.sum(LengthData.length_AL_total)/AL_vol

#%%

LOAD = True

if LOAD:
    morph_dist_calyx_r_new = np.load(r'./morph_dist_calyx_r_new.npy')
    morph_dist_LH_r_new = np.load(r'./morph_dist_LH_r_new.npy')
    morph_dist_AL_r_new = np.load(r'./morph_dist_AL_r_new.npy')
else:    
    morph_dist_calyx_CM_flat = np.array([item for sublist in morph_dist_calyx_CM for item in sublist])
    morph_dist_LH_CM_flat = np.array([item for sublist in morph_dist_LH_CM for item in sublist])
    morph_dist_AL_CM_flat = np.array([item for sublist in morph_dist_AL_CM for item in sublist])
    
    morph_dist_calyx_r_new = np.zeros((len(morph_dist_calyx_CM_flat), len(morph_dist_calyx_CM_flat)))
    morph_dist_LH_r_new = np.zeros((len(morph_dist_LH_CM_flat), len(morph_dist_LH_CM_flat)))
    morph_dist_AL_r_new = np.zeros((len(morph_dist_AL_CM_flat), len(morph_dist_AL_CM_flat)))
    
    for i in range(len(morph_dist_calyx_CM_flat)):
        for j in range(len(morph_dist_calyx_CM_flat)):
            morph_dist_calyx_ed = scipy.spatial.distance.cdist(morph_dist_calyx_flt[i], morph_dist_calyx_flt[j])
            morph_dist_LH_ed = scipy.spatial.distance.cdist(morph_dist_LH_flt[i], morph_dist_LH_flt[j])
            morph_dist_AL_ed = scipy.spatial.distance.cdist(morph_dist_AL_flt[i], morph_dist_AL_flt[j])
            
            # NNmetric
            if len(morph_dist_calyx_flt[i]) < len(morph_dist_calyx_flt[j]):
                N_calyx = len(morph_dist_calyx_flt[i])
                dmin_calyx = np.min(morph_dist_calyx_ed, axis=1)
            elif len(morph_dist_calyx_flt[i]) > len(morph_dist_calyx_flt[j]):
                N_calyx = len(morph_dist_calyx_flt[j])
                dmin_calyx = np.min(morph_dist_calyx_ed, axis=0)
            else:
                N_calyx = len(morph_dist_calyx_flt[i])
                r1 = np.min(morph_dist_calyx_ed, axis=0)
                r2 = np.min(morph_dist_calyx_ed, axis=1)
                if np.sum(r1) < np.sum(r2):
                    dmin_calyx = r1
                else:
                    dmin_calyx = r2
            
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
            
            morph_dist_calyx_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_calyx)), N_calyx))
            morph_dist_LH_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_LH)), N_LH))
            morph_dist_AL_r_new[i][j] = np.sqrt(np.divide(np.sum(np.square(dmin_AL)), N_AL))

calyxdist_cluster_u_full_new = []
calyxdist_noncluster_u_full_new = []

for i in range(len(glo_list)):
    calyx_sq = morph_dist_calyx_r_new[glo_lbs[i]:glo_lbs[i+1],glo_lbs[i]:glo_lbs[i+1]]
    calyx_sq_tri = calyx_sq[np.triu_indices_from(calyx_sq, k=1)]
    calyx_nc = np.delete(morph_dist_calyx_r_new[glo_lbs[i]:glo_lbs[i+1]], np.arange(glo_lbs[i], glo_lbs[i+1]))
        
    if len(calyx_sq_tri) > 0:
        calyxdist_cluster_u_full_new.append(calyx_sq_tri)
    else:
        calyxdist_cluster_u_full_new.append([])
    calyxdist_noncluster_u_full_new.append(calyx_nc.flatten())

calyxdist_cluster_u_full_flat_new = [item for sublist in calyxdist_cluster_u_full_new for item in sublist]
calyxdist_noncluster_u_full_flat_new = [item for sublist in calyxdist_noncluster_u_full_new for item in sublist]

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


print("Calyx cluster Mean: " + str(np.mean(calyxdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(calyxdist_cluster_u_full_flat_new)))
print("Calyx noncluster Mean: " + str(np.mean(calyxdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(calyxdist_noncluster_u_full_flat_new)))

print("LH cluster Mean: " + str(np.mean(LHdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(LHdist_cluster_u_full_flat_new)))
print("LH noncluster Mean: " + str(np.mean(LHdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(LHdist_noncluster_u_full_flat_new)))

print("AL cluster Mean: " + str(np.mean(ALdist_cluster_u_full_flat_new)) + ", STD: " + str(np.std(ALdist_cluster_u_full_flat_new)))
print("AL noncluster Mean: " + str(np.mean(ALdist_noncluster_u_full_flat_new)) + ", STD: " + str(np.std(ALdist_noncluster_u_full_flat_new)))

#%%

fig, ax = plt.subplots(figsize=(6,6))
labels = ['AL', 'MB calyx', 'LH']
x = np.arange(len(labels))
width = .3

cmeans = [np.mean(ALdist_cluster_u_full_flat_new), 
          np.mean(calyxdist_cluster_u_full_flat_new), 
          np.mean(LHdist_cluster_u_full_flat_new)]
cerr = [np.std(ALdist_cluster_u_full_flat_new),
        np.std(calyxdist_cluster_u_full_flat_new), 
        np.std(LHdist_cluster_u_full_flat_new)]
ncmeans = [np.mean(ALdist_noncluster_u_full_flat_new), 
           np.mean(calyxdist_noncluster_u_full_flat_new), 
           np.mean(LHdist_noncluster_u_full_flat_new)]
ncerr = [np.std(ALdist_noncluster_u_full_flat_new),
         np.std(calyxdist_noncluster_u_full_flat_new), 
         np.std(LHdist_noncluster_u_full_flat_new)]

lamb = [np.mean(ALdist_cluster_u_full_flat_new)/np.mean(ALdist_noncluster_u_full_flat_new), 
        np.mean(calyxdist_cluster_u_full_flat_new)/np.mean(calyxdist_noncluster_u_full_flat_new), 
        np.mean(LHdist_cluster_u_full_flat_new)/np.mean(LHdist_noncluster_u_full_flat_new)]

lamberr = [np.sqrt(np.square(cerr[0]/cmeans[0]) + np.square(ncerr[0]/ncmeans[0]))*lamb[0], 
           np.sqrt(np.square(cerr[1]/cmeans[1]) + np.square(ncerr[1]/ncmeans[1]))*lamb[1],
           np.sqrt(np.square(cerr[2]/cmeans[2]) + np.square(ncerr[2]/ncmeans[2]))*lamb[2]]

bar1 = ax.bar(x - width, cmeans, width, yerr=cerr, capsize=5, label=r'$\bar{d}_{{\rm intra}}$', color='tab:blue')
bar2 = ax.bar(x, ncmeans, width, yerr=ncerr, capsize=5, label=r'$\bar{d}_{{\rm inter}}$', color='tab:orange')
ax2 = ax.twinx()
bar3 = ax2.bar(x + width, lamb, width, yerr=lamberr, capsize=5, label='$\lambda$', color='tab:red')
ax2.set_ylim(0, 1)
ax.set_ylabel('Distance ($\mu$m)', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=17)
ax.tick_params(axis="y", labelsize=15)
ax2.tick_params(axis="y", labelsize=15)
ax2.set_ylabel('Ratio', fontsize=17)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=1, fontsize=15)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(5, 9))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.hist(ALdist_cluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax1.hist(ALdist_noncluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax1.set_ylim(0, 0.4)
ax1.set_ylabel('AL', fontsize=15)
ax1.legend(['Identical Glomerulus', 'Different Glomeruli'], fontsize=13)
ax2.hist(calyxdist_cluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax2.hist(calyxdist_noncluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax2.set_ylim(0, 0.4)
ax2.set_ylabel('MB calyx', fontsize=15)
ax3.hist(LHdist_cluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax3.hist(LHdist_noncluster_u_full_flat_new, bins=20, alpha=0.5, density=True)
ax3.set_ylim(0, 0.4)
ax3.set_ylabel('LH', fontsize=15)
ax3.set_xlabel(r'Distance $(\mu m)$', fontsize=15)
plt.tight_layout()
plt.show()

#%%

calyxtest_cl = []
calyxtest_ncl = []
calyxtest_cl_std = []
calyxtest_ncl_std = []
for i in range(len(calyxdist_cluster_u_full_new)):
    calyxtest_cl.append(np.mean(calyxdist_cluster_u_full_new[i]))
    calyxtest_cl_std.append(np.std(calyxdist_cluster_u_full_new[i]))
for i in range(len(calyxdist_noncluster_u_full_new)):
    calyxtest_ncl.append(np.mean(calyxdist_noncluster_u_full_new[i]))
    calyxtest_ncl_std.append(np.std(calyxdist_noncluster_u_full_new[i]))
    
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
    
calyxtest_cl = np.nan_to_num(calyxtest_cl)
calyxtest_ncl = np.nan_to_num(calyxtest_ncl)
LHtest_cl = np.nan_to_num(LHtest_cl)
LHtest_ncl = np.nan_to_num(LHtest_ncl)
ALtest_cl = np.nan_to_num(ALtest_cl)
ALtest_ncl = np.nan_to_num(ALtest_ncl)

calyxtest_cl_std = np.nan_to_num(calyxtest_cl_std)
calyxtest_ncl_std = np.nan_to_num(calyxtest_ncl_std)
LHtest_cl_std = np.nan_to_num(LHtest_cl_std)
LHtest_ncl_std = np.nan_to_num(LHtest_ncl_std)
ALtest_cl_std = np.nan_to_num(ALtest_cl_std)
ALtest_ncl_std = np.nan_to_num(ALtest_ncl_std)

ALtest_idx = np.where(np.array(ALtest_cl) >= 0)[0]
LHtest_idx = np.where(np.array(LHtest_cl) >= 0)[0]
calyxtest_idx = np.where(np.array(calyxtest_cl) >= 0)[0]


type_idx = [17, 21, 26, 9, 48, 6, 
            10, 44, 37, 27, 36, 16, 39, 
            30, 2, 15, 45, 1, 42, 50, 
            8, 13, 19, 4, 32, 34, 
            5, 12, 43, 33, 23, 22, 49, 14,
            46, 20, 3, 38, 40, 18, 35, 25,
            0, 7,
            11, 47,
            41, 24, 28, 31, 29]

attavdict1 = {'DL2d': '#028e00', 'DL2v': '#028e00', 'VL1': '#028e00', 'VL2a': '#028e00', 'VM1': '#028e00', 'VM4': '#028e00', 
             'DM1': '#7acb2f', 'DM4': '#7acb2f', 'DM5': '#7acb2f', 'DM6': '#7acb2f', 'VA4': '#7acb2f', 'VC2': '#7acb2f', 'VM7d': '#7acb2f',
             'DA3': '#00f700', 'DC1': '#00f700', 'DL1': '#00f700', 'VA3': '#00f700', 'VM2': '#00f700', 'VM5d': '#00f700', 'VM5v': '#00f700',  
             'DA4m': '#a3a3a3', 'VA7m': '#a3a3a3', 'VC3l': '#a3a3a3', 'VC3m': '#a3a3a3', 'VM6': '#a3a3a3', 'VM7v': '#a3a3a3', 
             'DM2': '#17d9f7', 'DP1l': '#17d9f7', 'DP1m': '#17d9f7', 'V': '#17d9f7', 'VA2': '#17d9f7', 'VC4': '#17d9f7', 'VL2p': '#17d9f7', 'VM3': '#17d9f7', 
             'D': '#f10000', 'DA2': '#f10000', 'DA4l': '#f10000', 'DC2': '#f10000', 'DC4': '#f10000', 'DL4': '#f10000', 'DL5': '#f10000', 'DM3': '#f10000',
             'VA6': '#e8f0be', 'VC1': '#e8f0be', 
             'VA5': '#b96d3d', 'VA7l': '#b96d3d', 
             'DA1': '#a200cb', 'DC3': '#a200cb', 'DL3': '#a200cb', 'VA1d': '#a200cb', 'VA1v': '#a200cb'}

attavdict2 = {'DL2d': '#027000', 'DL2v': '#027000', 'VL1': '#027000', 'VL2a': '#027000', 'VM1': '#027000', 'VM4': '#027000', 
             'DM1': '#5dad2f', 'DM4': '#5dad2f', 'DM5': '#5dad2f', 'DM6': '#5dad2f', 'VA4': '#5dad2f', 'VC2': '#5dad2f', 'VM7d': '#5dad2f',
             'DA3': '#05cf02', 'DC1': '#05cf02', 'DL1': '#05cf02', 'VA3': '#05cf02', 'VM2': '#05cf02', 'VM5d': '#05cf02', 'VM5v': '#05cf02',  
             'DA4m': '#858585', 'VA7m': '#858585', 'VC3l': '#858585', 'VC3m': '#858585', 'VM6': '#858585', 'VM7v': '#858585', 
             'DM2': '#17becf', 'DP1l': '#17becf', 'DP1m': '#17becf', 'V': '#17becf', 'VA2': '#17becf', 'VC4': '#17becf', 'VL2p': '#17becf', 'VM3': '#17becf', 
             'D': '#bf0000', 'DA2': '#bf0000', 'DA4l': '#bf0000', 'DC2': '#bf0000', 'DC4': '#bf0000', 'DL4': '#bf0000', 'DL5': '#bf0000', 'DM3': '#bf0000',
             'VA6': '#d4d296', 'VC1': '#d4d296', 
             'VA5': '#91451f', 'VA7l': '#91451f', 
             'DA1': '#700099', 'DC3': '#700099', 'DL3': '#700099', 'VA1d': '#700099', 'VA1v': '#700099'}

attavdict3 = {'DL2d': '#025200', 'DL2v': '#025200', 'VL1': '#025200', 'VL2a': '#025200', 'VM1': '#025200', 'VM4': '#025200', 
             'DM1': '#3f8f2f', 'DM4': '#3f8f2f', 'DM5': '#3f8f2f', 'DM6': '#3f8f2f', 'VA4': '#3f8f2f', 'VC2': '#3f8f2f', 'VM7d': '#3f8f2f',
             'DA3': '#05a702', 'DC1': '#05a702', 'DL1': '#05a702', 'VA3': '#05a702', 'VM2': '#05a702', 'VM5d': '#05a702', 'VM5v': '#05a702',  
             'DA4m': '#676767', 'VA7m': '#676767', 'VC3l': '#676767', 'VC3m': '#676767', 'VM6': '#676767', 'VM7v': '#676767', 
             'DM2': '#17a0a7', 'DP1l': '#17a0a7', 'DP1m': '#17a0a7', 'V': '#17a0a7', 'VA2': '#17a0a7', 'VC4': '#17a0a7', 'VL2p': '#17a0a7', 'VM3': '#17a0a7', 
             'D': '#8d0000', 'DA2': '#8d0000', 'DA4l': '#8d0000', 'DC2': '#8d0000', 'DC4': '#8d0000', 'DL4': '#8d0000', 'DL5': '#8d0000', 'DM3': '#8d0000',
             'VA6': '#b6b46e', 'VC1': '#b6b46e', 
             'VA5': '#592628', 'VA7l': '#592628', 
             'DA1': '#480071', 'DC3': '#480071', 'DL3': '#480071', 'VA1d': '#480071', 'VA1v': '#480071'}

updatedxlabel = np.array(glo_list)[type_idx]

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


fig, ax = plt.subplots(1, 3, figsize=(8,12))
x = np.arange(len(calyxtest_idx))
width = .275

ax[0].barh(x + width, ALtest_cl[type_idx], 
          width,
          capsize=5, label='Identical Glomerulus', color=np.array(attavlist1), alpha=0.5)
ax[0].barh(x , calyxtest_cl[type_idx], 
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
ax[1].barh(x, calyxtest_ncl[type_idx], 
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
ax[2].barh(x, np.divide(calyxtest_cl, calyxtest_ncl)[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist2), alpha=0.75)
ax[2].barh(x - width, np.divide(LHtest_cl, LHtest_ncl)[type_idx], 
          width,
          capsize=5, label='Different Glomeruli', color=np.array(attavlist3), alpha=1.)
ax[2].set_yticks(x)
ax[2].set_title('$\\lambda_{X}$', fontsize=25)
ax[2].set_xticks(np.array([0, 0.5, 1, 1.5]))
ax[2].set_yticklabels([])
ax[2].tick_params(axis="x", labelsize=15)
ax[2].set_ylim(x[0] - 1, x[-1] + 1)
plt.tight_layout()
plt.show()

n = np.nonzero(np.divide(LHtest_cl, LHtest_ncl)[type_idx])

comb = pd.DataFrame({'AL': np.divide(ALtest_cl, ALtest_ncl)[type_idx][n], 
                     'calyx': np.divide(calyxtest_cl, calyxtest_ncl)[type_idx][n], 
                     'LH': np.divide(LHtest_cl, LHtest_ncl)[type_idx][n]})

comb_decay = pd.DataFrame({'AL': np.divide(ALtest_cl, ALtest_ncl)[[17, 21, 26, 9, 48]],
                     'calyx': np.divide(calyxtest_cl, calyxtest_ncl)[[17, 21, 26, 9, 48]],
                     'LH': np.divide(LHtest_cl, LHtest_ncl)[[17, 21, 26, 9, 48]]})

comb_pheromone = pd.DataFrame({'AL': np.divide(ALtest_cl, ALtest_ncl)[[41, 24, 28, 31, 29]],
                     'calyx': np.divide(calyxtest_cl, calyxtest_ncl)[[41, 24, 28, 31, 29]],
                     'LH': np.divide(LHtest_cl, LHtest_ncl)[[41, 24, 28, 31, 29]]})

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
    if np.divide(calyxtest_cl, calyxtest_ncl)[type_idx[i]] != 0:
        plt.scatter(np.divide(calyxtest_cl, calyxtest_ncl)[type_idx[i]],
                    np.divide(ALtest_cl, ALtest_ncl)[type_idx[i]], 
                    color=attavdict2[updatedxlabel[i]])

coef = np.polyfit(comb['calyx'], comb['AL'], 1)
poly1d_fn = np.poly1d(coef)

ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.5)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel(r'MB calyx, $\lambda_{X}$', fontsize=15)
ax.set_ylabel(r'AL, $\lambda_{X}$', fontsize=15)
plt.show()


fig, ax = plt.subplots(figsize=(4,4))
for i in range(len(type_idx)):
    if np.divide(calyxtest_cl, calyxtest_ncl)[type_idx[i]] != 0:
        plt.scatter(np.divide(calyxtest_cl, calyxtest_ncl)[type_idx[i]],
                    np.divide(LHtest_cl, LHtest_ncl)[type_idx[i]], 
                    color=attavdict2[updatedxlabel[i]])
        
coef = np.polyfit(comb_pheromone['calyx'], comb_pheromone['LH'], 1)
poly1d_fn = np.poly1d(coef)
plt.plot(np.arange(0.2, 1., 0.1), poly1d_fn(np.arange(0.2, 1., 0.1)), attavdict2['DL3'], ls='--')
coef = np.polyfit(comb_decay['calyx'], comb_decay['LH'], 1)
poly1d_fn = np.poly1d(coef)
plt.plot(np.arange(0.1, 1., 0.1), poly1d_fn(np.arange(0.1, 1., 0.1)), attavdict2['VM4'], ls='--')
coef = np.polyfit(comb['calyx'], comb['LH'], 1)
poly1d_fn = np.poly1d(coef)

plt.text(0.9, 0.05, '$r=-0.997$\n$p<0.001$', color=attavdict2['DL3'], fontsize=11)
plt.text(0.6, 0.85, '$r=0.969$\n$p<0.01$', color=attavdict2['VM4'], fontsize=11)
ax.set_ylim(0, 1.5)
ax.set_xlim(0, 1.5)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel(r'MB calyx, $\lambda_{X}$', fontsize=15)
ax.set_ylabel(r'LH, $\lambda_{X}$', fontsize=15)
plt.show()

#%%

L_AL_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_AL_r_new), method='complete', optimal_ordering=True)
L_calyx_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_calyx_r_new), method='complete', optimal_ordering=True)
L_LH_new_ind = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.squareform(morph_dist_LH_r_new), method='complete', optimal_ordering=True)

glo_idx_flat_unsrt = [item for sublist in glo_idx for item in sublist]

fig, ax = plt.subplots(figsize=(20, 3))
R_AL_new = scipy.cluster.hierarchy.dendrogram(L_AL_new_ind,
                                        orientation='top',
                                        labels=glo_list_neuron,
                                        distance_sort='ascending',
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=9.5)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(20, 3))
R_calyx_new = scipy.cluster.hierarchy.dendrogram(L_calyx_new_ind,
                                        orientation='top',
                                        labels=glo_list_neuron,
                                        distance_sort='ascending',
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=16)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()

fig, ax = plt.subplots(figsize=(20, 3))
R_LH_new = scipy.cluster.hierarchy.dendrogram(L_LH_new_ind,
                                        orientation='top',
                                        labels=glo_list_neuron,
                                        distance_sort='ascending',
                                        show_leaf_counts=False,
                                        leaf_font_size=10,
                                        color_threshold=25)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
plt.show()

AL_t = []
LH_t = []
calyx_t = []

for k in np.arange(2, 70):
    ind_AL = scipy.cluster.hierarchy.fcluster(L_AL_new_ind, k, 'maxclust')
    AL_t.append(sklearn.metrics.silhouette_score(morph_dist_AL_r_new, ind_AL, metric="precomputed"))
    ind_calyx = scipy.cluster.hierarchy.fcluster(L_calyx_new_ind, k, 'maxclust')
    calyx_t.append(sklearn.metrics.silhouette_score(morph_dist_calyx_r_new, ind_calyx, metric="precomputed"))
    ind_LH = scipy.cluster.hierarchy.fcluster(L_LH_new_ind, k, 'maxclust')
    LH_t.append(sklearn.metrics.silhouette_score(morph_dist_LH_r_new, ind_LH, metric="precomputed"))

fig, ax = plt.subplots(figsize=(4, 3))
plt.plot(np.arange(2, 70), AL_t, color='tab:blue')
plt.plot(np.arange(2, 70), calyx_t, color='tab:orange')
plt.plot(np.arange(2, 70), LH_t, color='tab:green')
plt.legend(['AL', 'MB calyx', 'LH'], loc='center right', fontsize=10)
plt.ylabel('Average Silhouette Coefficients', fontsize=12)
plt.xlabel('Number of Clusters', fontsize=12)
plt.show()

print(np.argmax(AL_t))
print(np.argmax(calyx_t))
print(np.argmax(LH_t))

morph_dist_calyx_r_new_df = pd.DataFrame(morph_dist_calyx_r_new)

morph_dist_LH_r_new_df = pd.DataFrame(morph_dist_LH_r_new)

morph_dist_AL_r_new_df = pd.DataFrame(morph_dist_AL_r_new)

morph_dist_calyx_r_new_df = morph_dist_calyx_r_new_df.reindex(R_calyx_new['leaves'], axis=0)
morph_dist_calyx_r_new_df = morph_dist_calyx_r_new_df.reindex(R_calyx_new['leaves'], axis=1)

morph_dist_LH_r_new_df = morph_dist_LH_r_new_df.reindex(R_LH_new['leaves'], axis=0)
morph_dist_LH_r_new_df = morph_dist_LH_r_new_df.reindex(R_LH_new['leaves'], axis=1)

morph_dist_AL_r_new_df = morph_dist_AL_r_new_df.reindex(R_AL_new['leaves'], axis=0)
morph_dist_AL_r_new_df = morph_dist_AL_r_new_df.reindex(R_AL_new['leaves'], axis=1)

fig = plt.figure(figsize=(12,12))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_AL_r_new_df, cmap='plasma_r')
ax1.set_xticks([]) 
ax1.set_yticks([]) 
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
ax2.set_xticks(np.arange(len(ind_AL)+1))
ax3.set_yticks(np.arange(len(ind_AL)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(ind_AL))[:] + np.arange(len(ind_AL))[:])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(ind_AL))[:] + np.arange(len(ind_AL))[:])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_neuron[R_AL_new['leaves']]))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_neuron[R_AL_new['leaves']]))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
cbar = plt.colorbar(im, fraction=0.045)
cbar.ax.tick_params(labelsize=15)
plt.show()


fig = plt.figure(figsize=(12,12))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_calyx_r_new_df, cmap='plasma_r')
ax1.set_xticks([]) 
ax1.set_yticks([]) 
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
ax2.set_xticks(np.arange(len(ind_AL)+1))
ax3.set_yticks(np.arange(len(ind_AL)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(ind_AL))[:] + np.arange(len(ind_AL))[:])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(ind_AL))[:] + np.arange(len(ind_AL))[:])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_neuron[R_calyx_new['leaves']]))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_neuron[R_calyx_new['leaves']]))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
cbar = plt.colorbar(im, fraction=0.045)
cbar.ax.tick_params(labelsize=15)
plt.show()


fig = plt.figure(figsize=(12,12))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
im = plt.imshow(morph_dist_LH_r_new_df, cmap='plasma_r')
ax1.set_xticks([]) 
ax1.set_yticks([]) 
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
ax2.set_xticks(np.arange(len(ind_AL)+1))
ax3.set_yticks(np.arange(len(ind_AL)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(ind_AL))[:] + np.arange(len(ind_AL))[:])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(len(ind_AL))[:] + np.arange(len(ind_AL))[:])/2))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_neuron[R_LH_new['leaves']]))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_neuron[R_LH_new['leaves']]))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=8, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=8, rotation_mode='default')
cbar = plt.colorbar(im, fraction=0.045)
cbar.ax.tick_params(labelsize=15)
plt.show()

#%%

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi_val, p_val, dof, expected = scipy.stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi_val/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return chi_val, np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))), p_val

ind_AL_dist = scipy.cluster.hierarchy.fcluster(L_AL_new_ind, 39, 'maxclust')
ind_calyx_dist = scipy.cluster.hierarchy.fcluster(L_calyx_new_ind, 3, 'maxclust')
ind_LH_dist = scipy.cluster.hierarchy.fcluster(L_LH_new_ind, 4, 'maxclust')

print('glo_list vs Cd')
print(cramers_v(glo_list_neuron, ind_AL_dist))
print(cramers_v(glo_list_neuron, ind_calyx_dist))
print(cramers_v(glo_list_neuron, ind_LH_dist))

print('CAL vs Cd')
print(cramers_v(ind_AL_dist, ind_calyx_dist))
print(cramers_v(ind_AL_dist, ind_LH_dist))
print(cramers_v(ind_LH_dist, ind_calyx_dist))


odor_dict = {'DL2d': '#027000', 'DL2v': '#027000', 'VL1': '#027000', 'VL2a': '#027000', 'VM1': '#027000', 'VM4': '#027000', 
             'DM1': '#5dad2f', 'DM4': '#5dad2f', 'DM5': '#5dad2f', 'DM6': '#5dad2f', 'VA4': '#5dad2f', 'VC2': '#5dad2f', 'VM7d': '#5dad2f',
             'DA3': '#05cf02', 'DC1': '#05cf02', 'DL1': '#05cf02', 'VA3': '#05cf02', 'VM2': '#05cf02', 'VM5d': '#05cf02', 'VM5v': '#05cf02',  
             'DA4m': '#858585', 'VA7m': '#858585', 'VC3l': '#858585', 'VC3m': '#858585', 'VM6': '#858585', 'VM7v': '#858585', 
             'DM2': '#17becf', 'DP1l': '#17becf', 'DP1m': '#17becf', 'V': '#17becf', 'VA2': '#17becf', 'VC4': '#17becf', 'VL2p': '#17becf', 'VM3': '#17becf', 
             'D': '#bf0000', 'DA2': '#bf0000', 'DA4l': '#bf0000', 'DC2': '#bf0000', 'DC4': '#bf0000', 'DL4': '#bf0000', 'DL5': '#bf0000', 'DM3': '#bf0000',
             'VA6': '#d4d296', 'VC1': '#d4d296', 
             'VA5': '#91451f', 'VA7l': '#91451f', 
             'DA1': '#700099', 'DC3': '#700099', 'DL3': '#700099', 'VA1d': '#700099', 'VA1v': '#700099'}

grp1 = []

for i in glo_list_neuron:
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
    else:
        grp1.append(9)

grp1 = np.array(grp1)

print('odor type vs Cdist')
print(cramers_v(grp1, ind_AL_dist))
print(cramers_v(grp1, ind_calyx_dist))
print(cramers_v(grp1, ind_LH_dist))

odor_dict2 = {'DM6': '#d62728', 'VL2a': '#d62728', 'V': '#d62728', 'DL5': '#d62728', 'DM2': '#d62728', 'VM3': '#d62728',
            'DP1m': '#d62728', 'VL2p': '#d62728', 'DM3': '#d62728', 'VA3': '#2ca02c',
            'VA6': '#d62728', 'DM5': '#d62728', 'DL1': '#d62728', 'D': '#000000',
            'DC1': '#d62728', 'DC2': '#d62728', 'VA7l': '#d62728', 'VA5': '#d62728',
            'DC3': '#d62728', 'DA2': '#d62728', 'DL4': '#d62728', 'DC4': '#d62728', 
            'DA4l': '#d62728', 'VA7m': '#d62728', 'DA4m': '#d62728', 'VM7d': '#2ca02c',
            'VA2': '#2ca02c', 'DM1': '#2ca02c', 'DM4': '#2ca02c', 'VM5v': '#000000', 
            'VC2': '#2ca02c', 'VM2': '#2ca02c', 'VM5d': '#2ca02c',
            'DA3': '#2ca02c', 'VM4': '#2ca02c', 'VM1': '#2ca02c', 
            'VC1': '#2ca02c', 'VA1v': '#2ca02c', 'DA1': '#2ca02c', 'DL3': '#2ca02c',
            'VM7v': '#2ca02c', 'DP1l': '#000000', 'VC4': '#000000', 'VA4': '#000000', 
            'DL2d': '#000000', 'DL2v': '#000000', 'VL1': '#000000', 'VA1d': '#000000',
            'VC3l': '#000000', 'VC3m': '#000000', 'VM6': '#000000'}

grp2 = []

for i in glo_list_neuron:
    if odor_dict2[i] == '#d62728':
        grp2.append(1)
    elif odor_dict2[i] == '#2ca02c':
        grp2.append(2)
    else:
        grp2.append(3)

grp2 = np.array(grp2)

print('odor val vs Cdist')
print(cramers_v(grp2, ind_AL_dist))
print(cramers_v(grp2, ind_calyx_dist))
print(cramers_v(grp2, ind_LH_dist))

#%%

mi_sample_AL_d = []

for i in range(1000):
    a = np.random.choice(np.arange(39), 111)
    mi_sample_AL_d.append(sklearn.metrics.mutual_info_score(glo_list_neuron, a))

mi_sample_calyx_d = []

for i in range(1000):
    a = np.random.choice(np.arange(3), 111)
    mi_sample_calyx_d.append(sklearn.metrics.mutual_info_score(glo_list_neuron, a))

mi_sample_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(4), 111)
    mi_sample_LH_d.append(sklearn.metrics.mutual_info_score(glo_list_neuron, a))

print('glo_list vs Cd')
print(sklearn.metrics.mutual_info_score(glo_list_neuron, ind_AL_dist))
print(sklearn.metrics.mutual_info_score(glo_list_neuron, ind_calyx_dist))
print(sklearn.metrics.mutual_info_score(glo_list_neuron, ind_LH_dist))

print('CAL vs Cd')
print(sklearn.metrics.mutual_info_score(ind_AL_dist, ind_calyx_dist))
print(sklearn.metrics.mutual_info_score(ind_AL_dist, ind_LH_dist))
print(sklearn.metrics.mutual_info_score(ind_LH_dist, ind_calyx_dist))

print('d')
print(np.mean(mi_sample_AL_d), np.std(mi_sample_AL_d))
print(np.mean(mi_sample_calyx_d), np.std(mi_sample_calyx_d))
print(np.mean(mi_sample_LH_d), np.std(mi_sample_LH_d))

mi_sample_AL_d = []

for i in range(1000):
    a = np.random.choice(np.arange(39), 111)
    mi_sample_AL_d.append(sklearn.metrics.mutual_info_score(grp1, a))

mi_sample_calyx_d = []

for i in range(1000):
    a = np.random.choice(np.arange(3), 111)
    mi_sample_calyx_d.append(sklearn.metrics.mutual_info_score(grp1, a))

mi_sample_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(4), 111)
    mi_sample_LH_d.append(sklearn.metrics.mutual_info_score(grp1, a))


print('odor type vs Cdist')
print(sklearn.metrics.mutual_info_score(grp1, ind_AL_dist))
print(sklearn.metrics.mutual_info_score(grp1, ind_calyx_dist))
print(sklearn.metrics.mutual_info_score(grp1, ind_LH_dist))

print('d')
print(np.mean(mi_sample_AL_d), np.std(mi_sample_AL_d))
print(np.mean(mi_sample_calyx_d), np.std(mi_sample_calyx_d))
print(np.mean(mi_sample_LH_d), np.std(mi_sample_LH_d))

mi_sample_AL_d = []

for i in range(1000):
    a = np.random.choice(np.arange(39), 111)
    mi_sample_AL_d.append(sklearn.metrics.mutual_info_score(grp2, a))

mi_sample_calyx_d = []

for i in range(1000):
    a = np.random.choice(np.arange(3), 111)
    mi_sample_calyx_d.append(sklearn.metrics.mutual_info_score(grp2, a))

mi_sample_LH_d = []

for i in range(1000):
    a = np.random.choice(np.arange(4), 111)
    mi_sample_LH_d.append(sklearn.metrics.mutual_info_score(grp2, a))

print('odor val vs Cdist')
print(sklearn.metrics.mutual_info_score(grp2, ind_AL_dist))
print(sklearn.metrics.mutual_info_score(grp2, ind_calyx_dist))
print(sklearn.metrics.mutual_info_score(grp2, ind_LH_dist))

print('d')
print(np.mean(mi_sample_AL_d), np.std(mi_sample_AL_d))
print(np.mean(mi_sample_calyx_d), np.std(mi_sample_calyx_d))
print(np.mean(mi_sample_LH_d), np.std(mi_sample_LH_d))

#%% distance clustering LH

c_n = 3
view = 't'

cidx = np.where(ind_LH_dist == c_n)[0]

idx_cluster = np.array(glo_idx_flat_unsrt)[cidx]

tri_LH = []
for i in range(len(hull_LH.simplices)):
    tt = []
    for j in range(len(hull_LH.simplices[i])):
        tt.append(np.where(hull_LH.vertices == hull_LH.simplices[i][j])[0][0])
    tri_LH.append(tuple(tt))

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))

if view == 'f':
    hull_LH_temp = ConvexHull(np.array(morph_dist_LH_flat)[:,:2])
    vert = np.append(hull_LH_temp.vertices, hull_LH_temp.vertices[0])
    ax.plot(np.array(morph_dist_LH_flat)[vert][:,0], 
            np.array(morph_dist_LH_flat)[vert][:,1], 
            150,
            color='k',
            lw=5)
else:
    hull_LH_temp = ConvexHull(np.array(morph_dist_LH_flat)[:,[0,2]])
    vert = np.append(hull_LH_temp.vertices, hull_LH_temp.vertices[0])
    ax.plot(np.array(morph_dist_LH_flat)[vert][:,0], 
            np.repeat(230, len(vert)),
            np.array(morph_dist_LH_flat)[vert][:,2], 
            color='k',
            lw=5)

n = 0
for i in glo_idx_flat_unsrt:
    if i in idx_cluster:
        for j in range(len(MorphData.LHdist_per_n[i])):
            listOfPoints = MorphData.LHdist_per_n[i][j]
            for f in range(len(listOfPoints)-1):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], 
                          color=odor_dict[glo_list_neuron[cidx][n]], lw=1.)
        n += 1
ax.axis('off')

if view == 'f':
    ax.view_init(elev=90., azim=-90)
else:
    ax.view_init(elev=0., azim=-90)

ax.set_xlim(380, 480)
ax.set_ylim(280, 180)
ax.set_zlim(100, 200)
ax.dist = 7
plt.show()

#%% distance clustering calyx

c_n = 2
view = 't'

cidx = np.where(ind_calyx_dist == c_n)[0]

idx_cluster = np.array(glo_idx_flat_unsrt)[cidx]

tri_calyx = []
for i in range(len(hull_calyx.simplices)):
    tt = []
    for j in range(len(hull_calyx.simplices[i])):
        tt.append(np.where(hull_calyx.vertices == hull_calyx.simplices[i][j])[0][0])
    tri_calyx.append(tuple(tt))

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.set_box_aspect((1,1,1))

if view == 'f':
    hull_calyx_temp = ConvexHull(np.array(morph_dist_calyx_flat)[:,:2])
    vert = np.append(hull_calyx_temp.vertices, hull_calyx_temp.vertices[0])
    ax.plot(np.array(morph_dist_calyx_flat)[vert][:,0], 
            np.array(morph_dist_calyx_flat)[vert][:,1], 
            150,
            color='k',
            lw=5)
else:
    hull_calyx_temp = ConvexHull(np.array(morph_dist_calyx_flat)[:,[0,2]])
    vert = np.append(hull_calyx_temp.vertices, hull_calyx_temp.vertices[0])
    ax.plot(np.array(morph_dist_calyx_flat)[vert][:,0], 
            np.repeat(230, len(vert)),
            np.array(morph_dist_calyx_flat)[vert][:,2], 
            color='k',
            lw=5)

n = 0
for i in glo_idx_flat_unsrt:
    if i in idx_cluster:
        for j in range(len(MorphData.calyxdist_per_n[i])):
            listOfPoints = MorphData.calyxdist_per_n[i][j]
            for f in range(len(listOfPoints)-1):
                morph_line = np.vstack((listOfPoints[f], listOfPoints[f+1]))
            
                ax.plot3D(morph_line[:,0], morph_line[:,1], morph_line[:,2], 
                          color=odor_dict[glo_list_neuron[cidx][n]], lw=1.)
        n += 1
ax.axis('off')

if view == 'f':
    ax.view_init(elev=90., azim=-90)
else:
    ax.view_init(elev=0., azim=-90)

ax.set_xlim(470, 550)
ax.set_ylim(260, 180)
ax.set_zlim(140, 220)
ax.dist = 7
plt.show()

#%%

odor = ['VM4', 'VM1', 'DL2d', 'DL2v', 'VL1', 'VL2a', 
        'DM5', 'DM6', 'VM7d', 'DM1', 'DM4', 'VA4', 'VC2',
        'VM5d', 'DL1', 'VM5v', 'DA3', 'VA3', 'VM2', 'DC1',
        'VC3m', 'VC3l', 'VM7v', 'VA7m', 'DA4m', 'VM6',
        'V', 'VL2p', 'DM2', 'VC4', 'VM3', 'DP1l', 'DP1m', 'VA2',
        'DC2', 'D', 'DA2', 'DA4l', 'DC4', 'DL4', 'DL5', 'DM3',
        'VA6', 'VC1',
        'VA5', 'VA7l',
        'VA1v', 'DA1', 'DC3', 'VA1d', 'DL3']

columns_AL_new = []

for i in range(len(odor)):
    columns_AL_new.append(glo_list.index(odor[i]))

corrpoint = [0, 15, 27, 44, 57, 71, 86, 88, 92, 111]
glist = ['VM4', 'DM5', 'VM5d', 'VC3m', 'V', 'DC2', 'VA6', 'VA5', 'VA1v']

glo_list_cluster = np.array(glo_list)[columns_AL_new]
glo_len_cluster = np.array(glo_len)[columns_AL_new]

ind_AL_dist = scipy.cluster.hierarchy.fcluster(L_AL_new_ind, 39, 'maxclust')
ind_calyx_dist = scipy.cluster.hierarchy.fcluster(L_calyx_new_ind, 3, 'maxclust')
ind_LH_dist = scipy.cluster.hierarchy.fcluster(L_LH_new_ind, 4, 'maxclust')

ind_LH_dist[np.where(ind_LH_dist == 3)] = 5
ind_LH_dist[np.where(ind_LH_dist == 4)] = 3
ind_LH_dist[np.where(ind_LH_dist == 5)] = 4

glo_idx_cluster_dist = []
for i in range(len(glo_list)):
    taridx = glo_idx[np.argwhere(np.array(glo_list)[columns_AL_new][i] == np.array(glo_list))[0][0]]
    temp = []
    for j in taridx:
        temp.append(glo_idx_flat_unsrt.index(j))
    glo_idx_cluster_dist.append(temp)

glo_idx_cluster_dist_flat = [item for sublist in glo_idx_cluster_dist for item in sublist]

ct_AL_glo_temp_dist = np.zeros((len(np.unique(ind_AL_dist)), len(glo_idx_flat)))
ct_LH_glo_temp_dist = np.zeros((len(np.unique(ind_LH_dist)), len(glo_idx_flat)))
ct_calyx_glo_temp_dist = np.zeros((len(np.unique(ind_calyx_dist)), len(glo_idx_flat)))

ix = 0

for i in range(len(glo_idx)):
    for j in range(len(glo_idx[i])):
        ct_AL_glo_temp_dist[ind_AL_dist[ix]-1][ix] = 1
        ct_LH_glo_temp_dist[ind_LH_dist[ix]-1][ix] = 1
        ct_calyx_glo_temp_dist[ind_calyx_dist[ix]-1][ix] = 1
        ix += 1

ct_AL_glo_dist = []
ct_LH_glo_dist = []
ct_calyx_glo_dist = []

ct_AL_glo_dist.append(np.array(ct_AL_glo_temp_dist)[:,glo_idx_cluster_dist_flat])
ct_LH_glo_dist.append(np.array(ct_LH_glo_temp_dist)[:,glo_idx_cluster_dist_flat])
ct_calyx_glo_dist.append(np.array(ct_calyx_glo_temp_dist)[:,glo_idx_cluster_dist_flat])

glo_len_cluster = np.array(glo_len)[columns_AL_new]
glo_lb_cluster = [sum(glo_len_cluster[0:i]) for i in range(len(glo_len_cluster)+1)]
glo_lb_cluster_s = np.subtract(glo_lb_cluster, glo_lb_cluster[0])
glo_float_cluster = np.divide(glo_lb_cluster_s, glo_lb_cluster_s[-1])


fig = plt.figure(figsize=(10,0.15*len(np.unique(ind_AL_dist))))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax1.set_xticks(np.arange(111)+0.45, minor=False)
ax1.set_yticks(np.arange(39)-0.5, minor=False)
ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.25)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
for i in range(len(corrpoint)-1):
    cmap = matplotlib.colors.ListedColormap(['w', odor_dict[glist[i]]])
    a = copy.deepcopy(ct_AL_glo_dist[0])
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
ax3.set_yticks(np.arange(np.max(ind_AL_dist)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(np.max(ind_AL_dist)+1) + 0.5)))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(np.core.defchararray.add('$C_{', np.char.add(np.arange(1,40).astype(str),'}^{AL}$'))))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=6, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
for i in range(len(glo_list_cluster)):
    ax2.text(((glo_float_cluster[1:] + glo_float_cluster[:-1])/2-0.003)[i], -2.5, 
              glo_list_cluster[i], rotation=90, fontsize=6, color=odor_dict[glo_list_cluster[i]])
plt.show()


fig = plt.figure(figsize=(10,0.15*len(np.unique(ind_LH_dist))))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax1.set_xticks(np.arange(111)+0.45, minor=False)
ax1.set_yticks(np.arange(4)-0.5, minor=False)
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
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(['$C_{1}^{LH}$', '$C_{2}^{LH}$', '$C_{3}^{LH}$', '$C_{4}^{LH}$']))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=6, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
for i in range(len(glo_list_cluster)):
    ax2.text(((glo_float_cluster[1:] + glo_float_cluster[:-1])/2-0.003)[i], -2.5, 
              glo_list_cluster[i], rotation=90, fontsize=6, color=odor_dict[glo_list_cluster[i]])
plt.show()


fig = plt.figure(figsize=(10,0.15*len(np.unique(ind_calyx_dist))))
ax1 = SubplotHost(fig, 111)
fig.add_subplot(ax1)
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax1.set_xticks(np.arange(111)+0.45, minor=False)
ax1.set_yticks(np.arange(3)-0.5, minor=False)
ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.25)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
for i in range(len(corrpoint)-1):
    cmap = matplotlib.colors.ListedColormap(['w', odor_dict[glist[i]]])
    a = copy.deepcopy(ct_calyx_glo_dist[0])
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
ax3.set_yticks(np.arange(np.max(ind_calyx_dist)+1))
ax3.invert_yaxis()
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax3.yaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator((glo_float_cluster[1:] + glo_float_cluster[:-1])/2))
ax3.yaxis.set_minor_locator(ticker.FixedLocator((np.arange(np.max(ind_calyx_dist)+1) + 0.5)))
# ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(glo_list_cluster))
ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(['$C_{1}^{MB}$', '$C_{2}^{MB}$', '$C_{3}^{MB}$']))
ax2.axis["top"].minor_ticklabels.set(rotation=-90, fontsize=6, rotation_mode='default')
ax3.axis["left"].minor_ticklabels.set(fontsize=6, rotation_mode='default')
for i in range(len(glo_list_cluster)):
    ax2.text(((glo_float_cluster[1:] + glo_float_cluster[:-1])/2-0.003)[i], -2.5, 
               glo_list_cluster[i], rotation=90, fontsize=6, color=odor_dict[glo_list_cluster[i]])
plt.show()
