import cupy as cp
import numpy as np
import sigpy as sp
from tqdm import tqdm
import bart

from bart_dims import coord_to_bart, ksp_to_bart


def recon_l2_coilbycoil(ksp, coord, maps, img_shape, lamda=0.01, num_iter=30):
    
    coord_gpu = sp.to_device(coord)
    NF = sp.linop.NUFFT(ishape=img_shape, coord=coord_gpu)

    nCoils = maps.shape[0]
    coil_ims = np.zeros((nCoils, *img_shape), dtype=np.complex128)

    for i in tqdm(range(nCoils)):
        maps_gpu = sp.to_device(maps[i], 0)
        ksp_coil = sp.to_device(ksp[i], 0)

        S = sp.linop.Multiply(ishape=img_shape, mult=maps_gpu)
        app = sp.app.LinearLeastSquares(A=NF*S, y=ksp_coil, lamda=lamda, max_iter=num_iter)
        im = app.run()
        coil_ims[i] = im.get()

    im = np.sum(np.conj(maps) * coil_ims, axis=0)

    return im


def recon_l2_sense(ksp_in, coords_in, maps, img_shape, lamda=0.01, num_iter=100):

    recon = bart.bart(1, 'pics -i'+ str(num_iter) + ' -d 3 -l2 -r ' + str(lamda) + ' -t',
                            coord_to_bart(coords_in[None]), 
                            ksp_to_bart(ksp_in[:, None]), 
                            maps.transpose(1,2,3,0))
    

    return recon
        

