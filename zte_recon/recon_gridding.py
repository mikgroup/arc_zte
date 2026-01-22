import sigpy as sp
import numpy as np
from .util.nufft_util import nufft_adjoint_postcompensation
from tqdm import tqdm

def recon_adjoint_postcomp_coilbycoil(ksp, coord, img_shape, oversamp=2, 
                                      norm="ortho", device=0):
    """Coil-by coil reconstruction using adjoint NUFFT using
    gridded ones density (post-)compensation

    Parameters
    ----------
    ksp : ndarray, (nCoils, nSpokes, nReadout)
    coord : ndarray (nSpokes, nReadout, 3)
    img_shape : [nX, nY, nZ]
    oversamp : int, optional
        Grid oversampling, by default 2
    norm : str, optional
        FFT mode for sigpy, by default "ortho"
    device_num : int, optional
        sigpy conventions. -1 for CPU, else >=0, by default 0

    Returns
    -------
    ndarray (nCoils, nX, nY, nZ)
        Reconstructed coil images 
    """
    
    nCoils = ksp.shape[0]
    im_coils = np.zeros((nCoils, *img_shape), dtype=np.complex64)

    # Transfer if GPU device
    coord = sp.to_device(coord, device)

    for i in tqdm(range(nCoils), desc ="Coil-by-coil Gridding recon"):

        # Transfer if  GPU device
        ksp_coil = sp.to_device(ksp[i], device)

        im = nufft_adjoint_postcompensation(ksp_coil, 
                                            coord, 
                                            oversamp=oversamp, 
                                            oshape=img_shape, 
                                            norm=norm)
        # Transfer if  GPU device
        im_coils[i] = sp.to_device(im, -1)
    
    print('Coil-by-coil recon finished')

    return im_coils


    

