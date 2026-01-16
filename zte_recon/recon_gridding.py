import sigpy as sp
import sigpy.mri as mr
import numpy as np
from util.nufft_util import nufft_adjoint_postcompensation, nufft_adjoint_postcompensation_numpy
from tqdm import tqdm

def recon_adjoint_postcomp_coilbycoil(ksp, coord, img_shape, oversamp=2, norm="ortho", device_num=0):
    '''
    Gridded ones density compensation NUFFT adjoint
    '''    
    nCoils = ksp.shape[0]
    im_coils = np.zeros((nCoils, *img_shape), dtype=np.complex64)
    coord_gpu = sp.to_device(coord, device_num)

    for i in tqdm(range(nCoils), desc ="Coil-by-coil Gridding recon"):
        ksp_gpu = sp.to_device(ksp[i], device_num)
        im = nufft_adjoint_postcompensation(ksp_gpu, coord_gpu, 
                                            oversamp=oversamp, oshape=img_shape, norm=norm)
        im_coils[i] = im.get()
    
    print('Coil-by-coil recon finished')

    return im_coils


def recon_adjoint_postcomp_coilbycoil_cpu(ksp, coord, img_shape, oversamp=2, norm="ortho"):
    '''
    Gridded ones density compensation NUFFT adjoint on CPU
    '''    
    nCoils = ksp.shape[0]
    im_coils = np.zeros((nCoils, *img_shape), dtype=np.complex64)

    for i in tqdm(range(nCoils), desc ="Coil-by-coil Gridding recon"):
        im = nufft_adjoint_postcompensation_numpy(ksp[i], coord, 
                                            oversamp=oversamp, oshape=img_shape, 
                                            norm=norm)
        im_coils[i] = im
    
    print('Coil-by-coil recon finished')

    return im_coils


    

