import numpy as np
import scipy

def apply_cc_ksp(ksp, A_cc):
    ''' 
    Input shape ksp: nCoils, nSpokes, nReaodut
    A_cc as found by function find_A_cc_pca()
    '''

    nCoils = ksp.shape[0]
    ksp_cc = A_cc @ ksp.reshape(nCoils, ksp.shape[1]*ksp.shape[2])

    nVirtualCoils = ksp_cc.shape[0]
    ksp_cc = ksp_cc.reshape(nVirtualCoils, ksp.shape[1], ksp.shape[2])

    return ksp_cc

def apply_cc_ksp_flat(ksp_inner_flat, A_cc):
    '''
    Input shape ksp_inner_flat: 1, nSpokes*nReadout, 1, nCoils
    A_cc as found by function find_A_cc_pca()
    '''

    ksp_waspi_cc = A_cc @ (np.squeeze(ksp_inner_flat).T)
    ksp_waspi_cc = ksp_waspi_cc.T
    ksp_waspi_cc = ksp_waspi_cc[None, :, None, :]

    return ksp_waspi_cc

def find_A_cc_pca(ksp_inner_flat, nVirtualCoils):
    '''
    Input shape ksp_inner_flat: 1, nSpokes*nReadout, 1, nCoils
    '''

    k_lr = np.squeeze(ksp_inner_flat) # compressing based on waspi dataset
    k_lr = k_lr.T
    
    # need to produce vectors with mean 0 
    k_lr = k_lr - np.mean(abs(k_lr), axis=0, keepdims=True)
    
    print('Using SVD to find cc matrix')
    u, s, vh = np.linalg.svd(k_lr, full_matrices=False)

#     C = k_lr @ np.conj(k_lr.T)
#     w, v = np.linalg.eig(C)
#     idx = np.argsort(w) # sort in ascending order
#     w = w[idx]
#     v = v[:,idx]
#     A_cc = v[:, -1*nVirtualCoils:].T

    A_cc = vh[0:nVirtualCoils] # compression matrix is largest eigenvectors of cov matrix

    return A_cc


def find_A_cc_rovir(coil_ims_lowres, mask_sphere, nVirtualCoils):

    nCoils = coil_ims_lowres.shape[0]

    # Step 0 
    # Get coil images from ROI
    coil_ims_roi = coil_ims_lowres*mask_sphere[None]

    # Get coil images from interference region
    mask_intf = abs(mask_sphere - 1)
    coil_ims_intf = coil_ims_lowres*mask_intf[None]

    # Step 1 - Find matrices A and B from Equations 7 and 10 from paper
    A = np.zeros((nCoils, nCoils), dtype=complex)
    B = np.zeros((nCoils, nCoils), dtype=complex)

    for i in range(nCoils):
        for j in range(nCoils):
            
            A[i, j] = np.sum(np.conj(coil_ims_roi[i])*coil_ims_roi[j], axis=(0,1,2))
            B[i, j] = np.sum(np.conj(coil_ims_intf[i])*coil_ims_intf[j], axis=(0,1,2))

    # Step 2 - Solve generalized eigenvalue problem - eqn 13 from paper
    eigvals, eigvecs = scipy.linalg.eigh(a=A, b=B)

    eigvals = eigvals[::-1] # sort in descending order
    eigvecs = eigvecs[:, ::-1] # sort in descending order of eigvals

    # Step 3 - Orthgonalize eigenvectors using QR decomp
    # Orthogonalize using QR decomp
    eigvecs_orth, _ = np.linalg.qr(eigvecs)

    # Step 4 - Choose top k eigenvectors where k = nVirtualCoils
    rovir_weights = eigvecs_orth[:, 0:nVirtualCoils].T

    A_cc = rovir_weights
    return A_cc


def create_spherical_mask(h, w, d, center=None, diam=None):

    if center is None: # if none, use the middle of the image
        center = (int(w/2), int(h/2), int(d/2))
        
    if diam is None: # if none, use the smallest distance between the center and image walls
        rad = min(center[0], center[1], center[2], w-center[0], h-center[1], d-center[2])
        diam = [rad, rad, rad]

    Z, Y, X = np.ogrid[:d, :h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 / (diam[0]/2)**2 
                               + (Y-center[1])**2 / (diam[1]/2)**2 
                               + (Z-center[2])**2/ (diam[2]/2)**2 )

    mask = dist_from_center <= 1

    return mask