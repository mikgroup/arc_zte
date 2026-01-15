# Arc-ZTE
This repository contains code to reproduce figures in paper "Arc-ZTE: Incoherent temporal sampling for flexible, dynamic, quiet Zero-TE MRI using continuously-slewed
gradients" and to compute custom Arc-ZTE trajectories. 

## Computing Arc-ZTE trajectories
The folder `arc_zte_sim` contains python code that computes an Arc-ZTE trajectory for any custom arc angle by running the optimization-based computation of the per-TR twist angles described in the paper. 

## Reproducing paper figures
The folder `paper_figures` contains Jupyter notebooks that generates figures in the paper. 
- Phantom data can be downloaded at:
- Acquired patient data for lung free-breathing experiments can be downloaded at:
- Acquired patient data for DCE experiment can be downloaded at: 

## Packages used
The repo uses [`sigpy`](https://github.com/mikgroup/sigpy). Subspace-constrained reconstructions used code from [`ppcs`](https://github.com/sidward/ppcs) ([Iyer et al., 2024](https://epubs.siam.org/doi/10.1137/22M1530355)), which requires [`sympy`](https://github.com/sympy/sympy) and [`Chebyshev`](https://github.com/mlazaric/Chebyshev). To save gifs from numpy arrays, [`array2gif`](https://github.com/tanyaschlusser/array2gif) was used. A `requirements.txt` file is also provided with exact versions we used. 
