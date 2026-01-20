import numpy as np
import cupy as cp
import sigpy as sp
import h5py as h5py
from sigpy import backend, util, interp
from math import ceil
import scipy.ndimage as ndimage

from sigpy import backend, interp, util
from sigpy.fourier import _scale_coord, _get_oversamp_shape, _apodize


def nufft_adjoint_postcompensation(input, coord, oshape=None, oversamp=2, 
                                   width=4, norm="ortho"):
    """Adjoint NUFFT. Modified version of sigpy function to use 
    density compensation with gridded ones, since typical k-space weighting
    by sampling density causes issues with ZTE dead time gap.

    Original sigpy function: 
    https://sigpy.readthedocs.io/en/latest/_modules/sigpy/fourier.html#nufft_adjoint

    Density post-compensation is performed by gridding a vector of ones 
    to create a density map (dc). This map is smoothed with a Gaussian 
    filter ($\sigma=0.65$) before dividing the gridded data. 

    A threshold of $10^{-6}$ is applied to the density map to avoid 
    division by zero/samll values in unsampled regions of k-space. 

    Parameters
    ----------
    input : ndarray, (nSpokes, nReadout)
        Non-Cartesian k-space data. Can be a NumPy or CuPy array.
    coord : ndarray, (nSpokes, nReadout, nDim)
        K-space coordinates. Scaled such that they range from -N/2 to N/2.
    oshape : tuple of int, optional 
        Output image shape e.g. [nX, nY, nZ]
        If None, estimated from `coord`.
    oversamp : float, optional
        Oversampling factor for the internal Cartesian grid, by default 2.
    width : int, optional
        Interpolation kernel width (Kaiser-Bessel), by default 4.
    norm : str, optional
        Normalization for the FFT/IFFT operation, by default "ortho".

    Returns
    -------
    output : ndarray, oshape
        Reconstructed image

    """
    xp = backend.get_array_module(input)
    device = sp.get_device(input)

    # Estimate oshape if not specified
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + sp.estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    output = interp.gridding(input, coord, os_shape,
                                kernel='kaiser_bessel', width=width, param=beta)
    output /= width**ndim
    # Grid ones for density compensation
    dc = interp.gridding(xp.ones(input.shape), coord, os_shape,
                             kernel='kaiser_bessel', width=width, param=beta)
    if xp == cp: 
        dc = dc.get()
    # smooth density compensation with gaussian with size 2*ceil(osamp/2)+1
    dc = sp.to_device(ndimage.gaussian_filter(dc, sigma=0.65), device) # using matlab's default sigma

    # Apply density compensation where > 1e-6. Avoid dividing by small values
    output[dc > 1e-6] /= dc[dc > 1e-6] 

    # IFFT
    output = sp.ifft(output, axes=range(-ndim, 0), norm=norm)

    # Crop
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:])**0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)

    return output