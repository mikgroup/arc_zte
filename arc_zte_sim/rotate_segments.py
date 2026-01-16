import numpy as np

from .mat import rotation_matrix_from_vectors
from .golden_angles import tinygolden3dCoords


def golden3d(m):
    # original golden means
    phi1 = 0.4656
    phi2 = 0.6823

    z = ((m*phi1) % 2) - 1 # between -1 and 1
    phi = np.arccos(z) # for spherical coordinates, phi is between z axis and vector
    theta = 2*np.pi*((m*phi2) % 1) #xy angle 
    
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)

    return [x, y, z]

def seg_golden_angle_getRotMats(numRots):
    # find rotation from 1st golden angle onto ith angle(+2)
    M_list = []
    for i in range(numRots):
        v1 = golden3d(1)
        v2 = golden3d(i+2)
        M = rotation_matrix_from_vectors(v1, v2) 
        M_list.append(M)
    return M_list


def getRep(period, nSpokes, r):
    assert nSpokes >= period
    v1 = tinygolden3dCoords(2, 1)
    v2 = tinygolden3dCoords(2, 2)
    M = rotation_matrix_from_vectors(v1, v2)
    r.rotate()
    group = r.spoke_arr[0:period]
    total = group[:]
    #rotation matrix between golden angle points 1 and 2, and applying that to an entire group
    R = M
    while len(total) < nSpokes:
        group = R @ group
        total = np.concatenate((total, group))
    return np.array(total[0:nSpokes])