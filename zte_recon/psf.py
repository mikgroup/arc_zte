import cupy as cp
import sigpy as sp
import numpy as np

from .recon_gridding import recon_adjoint_postcomp_coilbycoil

def calc_psf_coords(coords_input, img_shape, device_num=0):
    '''
    coords_input dims = [nSPokes, nPts, 3]
    '''
    if device_num >= 0:
        xp = cp
    else:
        xp = np

    point_3d = xp.zeros(img_shape, dtype=xp.complex128)
    point_3d[img_shape[0]//2, 
             img_shape[1]//2, 
             img_shape[2]//2] = 1.0

    # Combine waspi and hires
    ksp_sim = sp.nufft(point_3d, sp.to_device(coords_input, device_num))

    # Adjoint coil-by-coil recon
    im_coils = recon_adjoint_postcomp_coilbycoil(ksp_sim[None], coords_input, img_shape, 
                                                 norm=None, disable_tqdm=True, 
                                                 device=device_num)

    return im_coils[0]


def sidelobe_to_peak_ratio(psf_im):

    psr = np.max(abs(psf_im))/np.partition(abs(psf_im).flatten(), -2)[-2]
    spr = 1/psr
    
    return spr


def sidelobe_to_peak_ratio_from_coords(coords_input, img_shape):

    return sidelobe_to_peak_ratio(calc_psf_coords(coords_input, img_shape))