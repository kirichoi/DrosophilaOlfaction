# DrosophilaOlfaction

Python scripts for analyzing *Drosophila* olfactory projection neuron reconstructions and reproducing the figures.
The original dataset is available from [Zheng et al. 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30787-6) and the [Github repo](https://github.com/bocklab/temca2data/tree/master/geometry_analysis/data).
EM-reconstructed neuron skeleton data in .swc format and glomerulus labels will be needed to run the script.

- drosophila_analysis.py: the Python script for the data analysis and figure reproduction
- Skels connectome_mod: contains artifect-corrected version of *Drosophila* olfactory projection neuron reconstructions originally published by Zheng et al. 2018.
- all_skeletons_type_list_180919.xls: contains glomerulus label information originally published by Zheng et al. 2018.
- morph_dist_AL_r_new.npy: numpy array containing raw inter-PN distances in AL
- morph_dist_calyx_r_new.npy: numpy array containing raw inter-PN distances in MB calyx
- morph_dist_LH_r_new.npy: numpy array containing raw inter-PN distances in LH
