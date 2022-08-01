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

There are few other necessary files to run the code that are not included in this repository that should be acquired separately.

- Morphological reconstruction files (.swc): All reconstructs are publically available from the above links. Use ALLEN_DB.py to query mouse V1 neurons.
- For the mouse V1, the supplementary file `1-s2.0-S0960982220308587-mmc4.csv`, which contains the neuron IDs that Gouwens et al. have used, is necessary for the comparison between F(q)-based and morphometry-based clustering results. This file can be found [here](https://www.nature.com/articles/s41593-019-0417-0).