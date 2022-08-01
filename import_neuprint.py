# -*- coding: utf-8 -*-
"""
Olfactory responses of Drosophila are encoded in the organization of projection neurons

Kiri Choi, Won Kyu Kim, Changbong Hyeon
School of Computational Sciences, Korea Institute for Advanced Study, Seoul 02455, Korea

This script queries uPNs from the hemibrain dataset that innervate all three
neuropils
"""

import pandas as pd
from neuprint import Client, fetch_adjacencies
from neuprint import NeuronCriteria
from neuprint import fetch_neurons
import time

# The script can re-query the uPNs and connectivity data from the hemibrain dataset.
# Change `QUERY = True' do so.
# CAUTION! - THIS WILL TAKE A LONG TIME!
# Using the pickled files are highly recommended
QUERY = False

# ENTER YOUR PERSONAL TOKEN HERE ##############################################
TOKEN = ('')
###############################################################################

c = Client('neuprint.janelia.org', 
           dataset='hemibrain:v1.2.1', 
           token=TOKEN)

criteria_PNallinrvt = NeuronCriteria(type='^.*_(.*PN)$', regex=True, 
                                     inputRois=['AL(R)'], outputRois=['CA(R)', 'LH(R)'], 
                                     status='Traced', cropped=False)

criteria_KC = NeuronCriteria(type='^KC.*', regex=True, 
                             inputRois=['CA(R)'], status='Traced', cropped=False)

criteria_LH = NeuronCriteria(type='^LH.*', regex=True, 
                             inputRois=['LH(R)'], status='Traced', cropped=False)


#%% Basic neuron information

if QUERY:
    PNallinrvt_df, PNallinrvt_roi_df = fetch_neurons(criteria_PNallinrvt)
    PNallinrvt_df.to_pickle(r'./hemibrain/neuprint_PNallinrvt_df.pkl')
else:
    PNallinrvt_df = pd.read_pickle(r'./hemibrain/neuprint_PNallinrvt_df.pkl')

PNallinrvtbid = list(PNallinrvt_df['bodyId'])
PNallinrvtinstance = list(PNallinrvt_df['instance'])
PNallinrvttype = list(PNallinrvt_df['type'])

#%% Query connectivity data

if QUERY:
    KC_df, KC_roi_df = fetch_neurons(criteria_KC)
    
    LH_df, LH_roi_df = fetch_neurons(criteria_LH)
    
    KCbid = list(KC_df['bodyId'])
    KCinstance = list(KC_df['instance'])
    KCtype = list(KC_df['type'])
    
    LHbid = list(LH_df['bodyId'])
    LHinstance = list(LH_df['instance'])
    LHtype = list(LH_df['type'])
    
    KCneuron_df, KCconn_df = fetch_adjacencies(PNallinrvt_df['bodyId'], KC_df['bodyId'], rois=['CA(R)'], min_roi_weight=3)
    LHneuron_df, LHconn_df = fetch_adjacencies(PNallinrvt_df['bodyId'], LH_df['bodyId'], rois=['LH(R)'], min_roi_weight=3)
    
    KCneuron_df.to_pickle(r'./hemibrain/neuron_PNKC_df.pkl')
    KCconn_df.to_pickle(r'./hemibrain/conn_PNKC_df.pkl')
    
    LHneuron_df.to_pickle(r'./hemibrain/neuron_PNLH_df.pkl')
    LHconn_df.to_pickle(r'./hemibrain/conn_PNLH_df.pkl')

#%% Query skeletal reconstructions

if QUERY:
    for i in PNallinrvtbid:
        print(i)
        skel = c.fetch_skeleton(i, format='swc', export_path=r'./hemibrain/neuprint_PN_invte_all/' + str(i) + '.swc')
        time.sleep(5)

    
    