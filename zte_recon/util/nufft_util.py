import numpy as np
import cupy as cp
import sigpy as sp
import h5py as h5py
from sigpy import backend, util, interp
from math import ceil
import scipy.ndimage as ndimage

from sigpy import backend, interp, util
from sigpy.fourier import _scale_coord, _get_oversamp_shape, _apodize


def nufft_adjoint_postcompensation(input, coord, oshape=None, oversamp=2, width=4, norm="ortho"):
    """Adjoint non-uniform Fast Fourier Transform.

    Based on code from sigpy, with small modification for post-compensation

    """
    xp = backend.get_array_module(input)

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
    dc = interp.gridding(xp.ones(input.shape), coord, os_shape,
                             kernel='kaiser_bessel', width=width, param=beta)
    if xp == cp: 
        dc = dc.get()
    # smooth density compensation with gaussian with size 2*ceil(osamp/2)+1
    dc = sp.to_device(ndimage.gaussian_filter(dc, sigma=0.65), 0) # using matlab's default sigma

    # mask where density compensation > 1e-6 instead of  != 0
    output[dc > 1e-6] /= dc[dc > 1e-6] 

    # IFFT
    output = sp.ifft(output, axes=range(-ndim, 0), norm=norm)

    # Crop
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:])**0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)

    return output