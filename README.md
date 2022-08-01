# DrosophilaOlfaction

A repository containinng the complete set of data to reproduce the results presented in the manuscript titled `Olfactory responses of Drosophila are encoded in the organization of projection neurons'

Copyright 2022 Kiri Choi

## Introduction

The scripts utilize two datasets: the FAFB dataset by [Bates et al. 2020](https://www.sciencedirect.com/science/article/pii/S0960982220308587) and the hemibrain dataset by [Scheffer et al. 2020](https://elifesciences.org/articles/57443).
A part of these datasets is reproduced in this repository. For the datasets, we would like to credit all the original authors.
Below is a short description of what each file contains:

- Drosophila_FAFB.py: the Python script for the data analysis and figure reproduction based on uPNs that innervate all three neuropils in the FAFB dataset
- Drosophila_FAFB_other_uPN.py: the Python script for the data analysis and figure reproduction based on uPNs that does not innervate all three neuropils in the FAFB dataset
- Drosophila_labeled_line.py: the Python script for the data analysis and figure reproduction of the labeled line study
- Drosophila_neuprint.py: the Python script for the data analysis and figure reproduction based on uPNs that innervate all three neuropils in the hemibrain dataset
- import_neuprint.py: the Python script for querying the neurons we used in the study from the neuPrint database
- FAFB_summary.xlsx: a summary of the uPNs we used that are from the FAFB dataset
- FAFB_summary.xlsx: a summary of the uPNs we used that are from the hemibrain dataset
- FAFB
    - FAFB_swc: contains the neuron reconstructions in .swc format
    - morph_dist_(AL/MB/LH)_r_FAFB(.csv/.npy): precomputed distance matrics of uPNs innervating all three neuropils
    - morph_dist_(AL/MB/LH)_r_FAFB_uPN(.csv/.npy): precomputed distance matrics of all uPNs innervating each neuropil
    - NBLAST_(AL/MB/LH)_FAFB(.csv/.npy): precomputed NBLAST distance matrics of uPNs innervating all three neuropils
- hemibrain
    - neuprint_PN_invte_all: contains the neuron reconstructions in .swc format
    - morph_dist_(AL/MB/LH)_r_neuprint(.csv/.npy): precomputed distance matrics of uPNs innervating all three neuropils
    - neuprint_PNallinrvt_df.pkl: a pickled instance of Pandas DataFrame containing information about all queried neurons
    - conn_(PNKC/PNLH)_df.pkl: a pickled instance of Pandas DataFrame containing connectivity information
    - neuron_(PNKC/PNLH)_df.pkl: a pickled instance of Pandas DataFrame containing information about third-order neurons