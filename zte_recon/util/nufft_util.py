import numpy as np
import cupy as cp
import sigpy as sp
import h5py as h5py
from sigpy import backend, util, interp
from math import ceil
import scipy.ndimage as ndimage

from sigpy import backend, interp, util
from sigpy.linop import Linop, NUFFT

class NUFFTAdjoint_PostComp(Linop):
    """NUFFT adjoint linear operator using cupy with nufft adjoint postcompensation

    Args:
        oshape (tuple of int): Output shape
        coord (array): Coordinates, with values [-ishape / 2, ishape / 2]
        oversamp (float): Oversampling factor.
        width (float): Kernel width.

    """

    def __init__(self, oshape, coord, oversamp=1.25, width=4):
        self.coord = coord
        self.oversamp = oversamp
        self.width = width

        ndim = coord.shape[-1]

        ishape = list(oshape[:-ndim]) + list(coord.shape[:-1])

        super().__init__(oshape, ishape)


    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            coord = backend.to_device(self.coord, device)
            return nufft_adjoint_postcompensation(
                input,
                coord,
                self.oshape,
                oversamp=self.oversamp,
                width=self.width,
            )

    def _adjoint_linop(self):
        return NUFFT(
            self.oshape, self.coord, oversamp=self.oversamp, width=self.width
        )
    


def nufft_adjoint_postcompensation(input, coord, oshape=None, oversamp=2, width=4, norm="ortho"):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:
        input (array): input Fourier domain array of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: signal domain array with shape specified by oshape.

    See Also:
        :func:`sigpy.nufft.nufft`

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
    # smooth density compensation with gaussian with size 2*ceil(osamp/2)+1
    dc = sp.to_device(ndimage.gaussian_filter(dc.get(), sigma=0.65), 0) # using matlab's default sigma

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



def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.copy()
    for i in range(-ndim, 0):
        scale = ceil(oversamp * shape[i]) / shape[i]
        shift = ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [ceil(oversamp * i) for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta):
    xp = backend.get_array_module(input)
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = ceil(oversamp * i)
        idx = xp.arange(i, dtype=output.dtype)

        # Calculate apodization
        apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
        apod /= xp.sinh(apod)
        output *= apod.reshape([i] + [1] * (-a - 1))

    return output


def nufft_adjoint_postcompensation_numpy(input, coord, oshape=None, oversamp=2, width=4, norm="ortho"):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:
        input (array): input Fourier domain array of shape
            (...) + coord.shape[:-1]. That is, the last dimensions
            of input must match the first dimensions of coord.
            The nufft_adjoint is applied on the last coord.ndim - 1 axes,
            and looped over the remaining axes.
        coord (array): Fourier domain coordinate array of shape (..., ndim).
            ndim determines the number of dimension to apply nufft adjoint.
            coord[..., i] should be scaled to have its range between
            -n_i // 2, and n_i // 2.
        oshape (tuple of ints): output shape of the form
            (..., n_{ndim - 1}, ..., n_1, n_0).
        oversamp (float): oversampling factor.
        width (float): interpolation kernel full-width in terms of
            oversampled grid.

    Returns:
        array: signal domain array with shape specified by oshape.

    See Also:
        :func:`sigpy.nufft.nufft`

    """
    xp = np
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
    # smooth density compensation with gaussian with size 2*ceil(osamp/2)+1
    dc = ndimage.gaussian_filter(dc, sigma=0.65) # using matlab's default sigma

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
