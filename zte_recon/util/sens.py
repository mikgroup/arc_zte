import bart
import numpy as np

def est_maps_3d(ksp, coord, img_shape, soft_sens=True, crop_thres=0.001, grid_size=10, ker_size=6):
    '''Calculates ESPIRiT maps from low-resolution non-cartesian k-space and
    corresponding coordinates. Map shape will be img_shape. One map
    per coil and no cropping.
    
    ksp dim: [1, nSpokes, nRO, nCoils]
    coord dim: [3, nSpokes, nRO]
    img_shape dim: [nx, ny, nz]
    '''
    lowres_img = bart.bart(1, 'nufft -i -d'+str(grid_size)+':'+str(grid_size)+':'+str(grid_size), 
                                                                    coord, ksp) #inverse gridding
    lowres_cart_ksp = bart.bart(1, 'fft 7', lowres_img)
    ksp_zeropd = bart.bart(1, 'resize -c 0 '+ str(img_shape[0]) +' 1 ' + str(img_shape[1]) 
                           +' 2 '+ str(img_shape[2]), lowres_cart_ksp)
    if soft_sens:
        maps = bart.bart(1, 'ecalib -m1 -S -k' + str(ker_size) + ' -c'+str(crop_thres)+' -r'+str(grid_size), ksp_zeropd) 
    else:
        maps = bart.bart(1, 'ecalib -m1 -k ' + str(ker_size) + ' -c'+str(crop_thres)+' -r'+str(grid_size), ksp_zeropd) 

    return maps