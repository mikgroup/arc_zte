# Arc-ZTE
This repository contains code to compute custom Arc-ZTE segment trajectories for any desired arc angle and to reproduce some figures in paper "Arc-ZTE: Incoherent temporal sampling for flexible, dynamic, quiet Zero-TE MRI using continuously-slewed gradients". 

## Computing Arc-ZTE trajectories
To compute a custom Arc-ZTE trajectory, the optimization to select per-TR twist angles can be run with the provided script using a call like:

`python gen_arczte_traj.py --arc_angle 53 --spokes_per_seg 384 --num_segs 172`

This script will save the rotation matrices each TR in the designed segment to a .txt file and the trajectory coordinates for the segment as a .npy file. Paths can be specified using the arguments `--out_rotmat_txt_path` and `--out_coords_npy_path` respectively. 

The folder `rot_txt_files` contains the rotation matrices .txt files we used for our tests shown in the paper (Figure 3 and 4). These files can be used directly for a scanner implementation to rotate the arc spoke gradients every TR and create gradient waveforms for a continuous segment. 

To create the different segments of the trajectory, we used golden angles to rotate this single segment in 3D; the rotation matrices we used are listed in `rot_txt_files/seg_golden3d_rotMats.txt`.

#### Comparison radial ZTE trajectories

Code for calculating AZTEK trajectories can be found in [`this Github repo`](https://github.com/BioMaps-MRI/AZTEK) based on [`this paper`](https://pubmed.ncbi.nlm.nih.gov/32936490/): "AZTEK: Adaptive Zero TE K-space trajectories. Tanguy Boucneau, Brice Fernandez, Florent Besson, Anne Menini, Florian Wiesinger, Emmanuel Durand, Caroline Caramella, Luc Darrasse, and Xavier Maître". We have implemented this trajectory on our scanner and save the readout endpoints as a txt file for reconstruction. 

Code for calculating trajectories for the Phyllotaxis scheme has been provided by Tobias Wood based on [`this paper`](https://pmc.ncbi.nlm.nih.gov/articles/PMC9321117/): "Motion corrected silent ZTE neuroimaging. Ljungberg E, Wood TC, Solana AB, Williams SCR, Barker GJ, Wiesinger F."

## Reproducing paper figures
The folder contains Jupyter notebooks that reproduces figures in the paper. Acquired phantom and in-vivo data will be available soon for download. 

These notebooks can be run locally on Jupyter Notebook or in Google Colab:

- Visualize trajectories in 3D (Figure 1): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Figure1.ipynb)
- Analysis of Arc-ZTE coverage uniformity and incoherence (Figures 3 - 5): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Figures3_4_5.ipynb)
- Reconstructions of phantom acquisitions with Arc-ZTE and comparison radial ZTE (Figure 7): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Figure7.ipynb)

- Visualize comparison radial ZTE trajectories (Supplementary Figure 1): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Supplementary_Figure1.ipynb)

#### Data
Data from the phantom experiments can be downloaded from here: [https://doi.org/10.17605/OSF.IO/C3SP5](https://doi.org/10.17605/OSF.IO/C3SP5). To download the data as a zip file, go to `Files` and click `Download as zip`. Place folder `data_for_arczte_paper` in the same folder as the Jupyter notebooks. Data from the in-vivo patient experiments cannot be shared due to IRB restrictions. 

## Dependencies and packages
The repository uses the BART framework from this fork [here](https://github.com/s-ramachandran/bart), which can be cloned and compiled using `make`. Here, we have implemented polynomial preconditioning to use with `pics` (Paper: [Iyer et al., 2024](https://epubs.siam.org/doi/10.1137/22M1530355); [`Code`](https://github.com/sidward/ppcs)).

A `requirements.txt` file is provided for convenient virtual environment setup. This file will install `sigpy` and other Python packages used by the code. 

Full high-resolution reconstructions were run on an A100 GPU. This hardware is not needed if code is run in "demo mode", which only runs a few iterations and low-resolution volumes to demonstrate that the code can run without failing. 
