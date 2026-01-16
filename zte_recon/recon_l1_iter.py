import sigpy as sp
import numpy as np
import sigpy.mri as mr
from tqdm import tqdm
import cupy as cp

class L1Wave_Recon(sp.app.LinearLeastSquares):
    r"""L1 Wavelet regularized reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 + \lambda \| W x \|_1

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, W is the wavelet operator,
    x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        wave_name (str): wavelet name.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
        Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Lustig, M., Donoho, D., & Pauly, J. M. (2007).
        Sparse MRI: The application of compressed sensing for rapid MR imaging.
        Magnetic Resonance in Medicine, 58(6), 1082-1195.

    """

    def __init__(self, A, y, img_shape, lamda_wave, alpha=None,
                 weights=None, coord=None, sigma=None,
                 wave_name='db4', device=sp.cpu_device,
                 coil_batch_size=None, comm=None, show_pbar=True,
                 transp_nufft=False, **kwargs):

        W = sp.linop.Wavelet(img_shape, wave_name=wave_name)
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, lamda_wave), W)

        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return lamda_wave * xp.sum(xp.abs(W(input))).item()
        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(A, y, alpha=alpha, sigma=sigma, proxg=proxg, g=g, show_pbar=show_pbar, **kwargs)

def recon_l1_wave_precond_coilbycoil(ksp, coord, img_shape, lamda_wave, max_iter=30, maps=None, device_num=0):
    '''
    Input kspace and coord on CPU as numpy arrays
    If maps provided, only useed for coil combination
    '''

    nCoils = ksp.shape[0]
    im_coils = np.zeros((nCoils, *img_shape), dtype=complex)
    
    # Calculate single-channel kspace preconditioner
    ones = np.ones([1, *img_shape], dtype=complex)
    precond_sc = mr.kspace_precond(ones, coord=coord, device=-1)
    precond_sc = sp.to_device(precond_sc, device_num)

    # NUFFT operator (forward model)
    NF = sp.linop.NUFFT(ishape=img_shape, coord=sp.to_device(coord, device_num))

    for i in tqdm(range(nCoils)):
        ksp_coil = sp.to_device(ksp[i], device_num) #load single coil onto GPU
        app = L1Wave_Recon(A=NF, y=ksp_coil, img_shape=img_shape, lamda_wave=lamda_wave, 
                           max_iter=max_iter, solver='PrimalDualHybridGradient', 
                           sigma=precond_sc[0])
        im = app.run()
        im_coils[i] = im.get()

    if maps is None:
        return sp.rss(im_coils, axes=0), im_coils
    
    else:
        cc_im = np.sum(np.conj(maps) * im_coils, axis=0)

        return cc_im, im_coils
    


def sing_vals_soft_thresholding(matrix, threshold):
    """
    Applies soft thresholding to the singular values of a matrix.
    """
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    S_thresholded = np.maximum(S - threshold, 0)
    matrix_low_rank = np.dot(U, np.dot(np.diag(S_thresholded), V))
    return matrix_low_rank



def recon_l1_wave_precond_lowrank_svst(ksp_binned, coord_binned, maps, lamda_wave, lamda_lr, 
                                       img_shape, num_iter, num_LR_steps, nEpochs, scale_factor=1e5):
    '''Run recon with l1 wavelet regularization spatially and singular value soft-thresholding across bins
    for low rank regularization
    '''

    nBins = ksp_binned.shape[0]

    # calculate preconditioner for each frame
    precond_sc_list = []
    for frame_num in range(nBins):
            coord_gpu = sp.to_device(coord_binned[frame_num], 0)

            # Calculate kspace preconditioner - don't need to recompute for same sampling coordinates
            print('Calculating preconditioner')
            ones = cp.ones(maps.shape, dtype=np.complex128)
            ones /= len(maps)**0.5
            precond_sc = mr.kspace_precond(ones, coord=coord_gpu, device=0)
            precond_sc_list.append(precond_sc)


    print('Transferring to GPU')
    ims_frames = np.zeros((nBins, *img_shape), dtype=np.complex128)
    taus = np.zeros((nBins)) # save taus to enforce maxEig is only run on the first epoch

    for epoch in tqdm(np.arange(0, nEpochs)):
        for frame_num in np.random.permutation(range(nBins)):
            coord_gpu = sp.to_device(coord_binned[frame_num], 0)
            ksp_gpu = sp.to_device(ksp_binned[frame_num] / scale_factor, 0)

            S = sp.linop.Multiply(ishape=[*img_shape], mult=maps)
            NF = sp.linop.NUFFT(ishape=S.oshape, coord=coord_gpu)
            
            if epoch == 0:
                app = L1Wave_Recon(A=NF*S, y=ksp_gpu, 
                                img_shape=img_shape, lamda_wave=lamda_wave, max_iter=num_iter, 
                                solver='PrimalDualHybridGradient', sigma=precond_sc_list[frame_num])
                ims_frames[frame_num] = app.run().get()
                taus[frame_num] = app.tau # save tau for future epoch and avoid recalculation
            else:
                # initialize x and tau in epochs > 0
                app = L1Wave_Recon(A=NF*S, y=ksp_gpu, x=sp.to_device(ims_frames[frame_num], 0), tau=taus[frame_num],
                                img_shape=img_shape, lamda_wave=lamda_wave, max_iter=num_iter, 
                                solver='PrimalDualHybridGradient', sigma=precond_sc_list[frame_num])
                ims_frames[frame_num] = app.run().get()
                
        # low rank regularization with singular value soft thresholding
        for i in range(num_LR_steps):
            ims_frames = sing_vals_soft_thresholding(ims_frames.reshape(nBins, -1).T, lamda_lr).T.reshape(nBins, *img_shape)




