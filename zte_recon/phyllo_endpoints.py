import numpy as np 

##### Phyllotaxis trajectory code ####

def phyllo_endpoints_merlin(spokes_per_seg, smoothness, spokes_total, gm):
    '''
    Function that outputs endpoints of phyllotaxis 
    '''
    # gm is True if golden means rotation between segments.
    # smoothness is s in MERLIN paper

    nseg = np.uint32(spokes_total // spokes_per_seg)
    phi_gold = 2.399963229728653
    dphi = phi_gold * _fib(smoothness)

    traj = np.zeros((spokes_total, 3))

    for ii in range(nseg):

        z_ii = (ii * 2) / (spokes_total - 1)
        phi_ii = (ii * phi_gold)
        rot = _gm(gm, ii) # golden meaens rotation per seg if gm True

        for ij in range(spokes_per_seg):

            z = (1 - (2 * nseg * ij) / (spokes_total - 1)) - z_ii
            theta = np.arccos(z)
            phi = (ij * dphi) + phi_ii

            ep = rot @ np.array([np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi),
                                z])
            traj[ij + ii*spokes_per_seg, :] = ep

    return traj/2 # normalize to [-0.5, 0.5]


def _gm(gm, ii):
    # Rotation matrix is golden means rotation if gm is True. Else identity
    # ii is index of segment for which # rotation matrix
    if gm:
        phi_gm1 = 0.465571231876768
        phi_gm2 = 0.682327803828019

        gm_phi = 2 * np.pi * ii * phi_gm2
        gm_theta = np.arccos(np.fmod(ii * phi_gm1, 1) * 2 - 1)

        rot = np.array([[np.cos(gm_theta) * np.cos(gm_phi), -np.sin(gm_phi), np.cos(gm_phi) * np.sin(gm_theta)],
                        [np.cos(gm_theta) * np.sin(gm_phi),  np.cos(gm_phi), np.sin(gm_theta) * np.sin(gm_phi)],
                        [-np.sin(gm_theta), 0, np.cos(gm_theta)]])

    else:
        rot = np.eye(3)

    return rot   
    
def _fib(n):
    # return fibonacci number n

    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return _fib(n-1) + _fib(n-2)
