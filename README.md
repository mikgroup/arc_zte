# Arc-ZTE
This repository contains code to reproduce figures in paper "Arc-ZTE: Incoherent temporal sampling for flexible, dynamic, quiet Zero-TE MRI using continuously-slewed
gradients" and to compute custom Arc-ZTE trajectories. 

## Computing Arc-ZTE trajectories
The folder `arc_zte_sim` contains python code that computes an Arc-ZTE trajectory for any custom arc angle. The optimization-based computation of the per-TR twist angles described in the paper can be run with the provided script with a call similar to this: 

`python run_arczte_seg_optim.py --arc_angle 53 --nSpokes_seg 384`

## Reproducing paper figures
The folder contains Jupyter notebooks that reproduces figures in the paper. Acquired phantom and in-vivo data will be available soon for download. 

## Packages used
The repo uses [`sigpy`](https://github.com/mikgroup/sigpy) and the python interface of [BART](https://mrirecon.github.io/bart/installation.html). Subspace-constrained reconstructions used code from [`ppcs`](https://github.com/sidward/ppcs) (Paper: [Iyer et al., 2024](https://epubs.siam.org/doi/10.1137/22M1530355)), which requires [`sympy`](https://github.com/sympy/sympy) and [`Chebyshev`](https://github.com/mlazaric/Chebyshev). 

To save gifs from reconstructed numpy arrays, [`array2gif`](https://github.com/tanyaschlusser/array2gif) was used. 

A `requirements.txt` file is also provided with exact versions used in our tests. 
