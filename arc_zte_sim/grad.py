import numpy as np

def ksp_coords_to_grad(ksp_coords_spoke, dt):
    '''
    Convert from ksp coordinates to gradients, with both in physical units
    ksp in cm-1 with dims[nSpokes, 3, nPts]
    dt in s

    Return grad with [nSpokes, 3, nPts-1] in G/cm
    '''
    gamma_bar = 4.257588 # kHz / G
    grad_spoke = np.diff(ksp_coords_spoke, axis=-1) # cm-1
    grad_spoke = grad_spoke * (1 / gamma_bar) * (1/(dt * 0.001)) # G/cm

    return grad_spoke # G/cm


def grad_to_ksp_coords(grad_spoke, dt):
    '''
    Convert from gradients to ksp coordinates, both in physical units
    grad_coords in G/cm with dims[3, nPts]
    dt in s

    Return ksp with [3, nPts] in cm^-1
    '''
    gamma_bar = 4.258 # kHz / G
    ksp_coords_spoke = np.cumsum(grad_spoke, axis=-1) * gamma_bar * dt * 0.001
    zeros = np.zeros((3,1))

    return np.concatenate((zeros, ksp_coords_spoke[:, :-1]), axis=-1) # cm-1


def grad_to_ksp_coords_norm(grad_spoke, dt):
    '''
    Convert from gradients to ksp coordinates, where ksp normalized to 0.5 
    grad in G/cm with dims[3, nPts]
    dt in s

    Return ksp_coords with [3, nPts] normalized to 0.5
    '''
    gamma_bar = 4.258 # kHz / G
    ksp_coords_spoke = np.cumsum(grad_spoke, axis=-1) * gamma_bar * dt * 0.001
    ksp_coords_spoke /= (2 * np.linalg.norm(ksp_coords_spoke[:, -1])) # normalize to 0.5 max
    zeros = np.zeros((3,1))

    return np.concatenate((zeros, ksp_coords_spoke[:, :-1]), axis=-1) # max norm 0.5

def grad_to_slew(grad_cat, dt):
    '''
    Args:
        grad_cat: dims [3, nPts] in G/cm 
        dt (float) : in  seconds
    Output:
        slew: dims [3, nPts-1] in T/m/s
    '''
    slew = (np.diff(grad_cat, axis=-1)/dt) # G/cm/s = G/(cm*s)
    slew = slew / 10**4 # Gauss to Tesla
    slew = slew / 10**-2 # cm -> m in denominator so T/m/s 

    return slew # T/m/s