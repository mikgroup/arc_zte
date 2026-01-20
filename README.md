**Work in Progress**

# Arc-ZTE
This repository contains code to reproduce figures in paper "Arc-ZTE: Incoherent temporal sampling for flexible, dynamic, quiet Zero-TE MRI using continuously-slewed gradients". 

This repository also contains code to compute Arc-ZTE segment trajectories for any desired arc angle using the optimization scheme to calculate per-TR twist angles without inducing gradient refocusing, as described in the paper. 

## Computing Arc-ZTE trajectories
The optimization to select per-TR twist angles can be run with the provided script using a call like:

`python run_arczte_seg_optim.py --arc_angle 53 --nSpokes_seg 384`

This script will save the rotation matrices each TR in the designed segment to a .txt file and the trajectory coordinates for the segment as a .npy file. Paths can be specified using the arguments `--out_rotmat_txt_path` and `--out_coords_npy_path` respectively. 

The folder `rot_txt_files` contains the rotation matrices .txt files we used for our tests shown in the paper (Figure 3 and 4). These files can be used directly for a scanner implementation to rotate the arc spoke gradients every TR and create gradient waveforms for a continuous segment. 

To create the different segments of the trajectory, we used golden angles to rotate this single segment in 3D; the rotation matrices we used are listed in `rot_txt_files/seg_golden3d_rotMats.txt`.

## Reproducing paper figures
The folder contains Jupyter notebooks that reproduces figures in the paper. Acquired phantom and in-vivo data will be available soon for download. 

Notebooks can be run locally on Jupyter Notebook or in Google Colab:

- Visualize trajectories (Figure 1): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Figure1.ipynb)
- Comparison of optimization scheme vs. naive schemes for per-TR twist angle selection (Figure 3): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Figure3.ipynb)
- Comparison across range of arc angles (Figure 4): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikgroup/arc_zte/blob/main/Figure4.ipynb)


## Packages used
The repo uses [`sigpy`](https://github.com/mikgroup/sigpy) and the python interface of [BART](https://mrirecon.github.io/bart/installation.html). Subspace-constrained reconstructions used code from [`ppcs`](https://github.com/sidward/ppcs) (Paper: [Iyer et al., 2024](https://epubs.siam.org/doi/10.1137/22M1530355)), which requires [`sympy`](https://github.com/sympy/sympy) and [`Chebyshev`](https://github.com/mlazaric/Chebyshev). 

To save gifs from reconstructed numpy arrays, [`array2gif`](https://github.com/tanyaschlusser/array2gif) was used. 

A `requirements.txt` file is also provided for convenient virtual environment setup. 
