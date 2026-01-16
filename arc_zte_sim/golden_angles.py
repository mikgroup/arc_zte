import numpy as np

def tinygolden3dCoords(N, m, rad=0.5):
    """
    Calculates the tiny 3D golden angle for a certain set (N) and iteration (m) for sphere radius rad
    
    Args:
        N, m: integers
    Returns:
        [x,y,z] : 3d point
    """
    φ1 = 0.465571231876768
    φ2 = 0.682327803828019
    phi2 = 1/(N - 1 + 1/φ2)
    phi1 = phi2/(1 + φ1)
    
    z = 2*((m*phi1) % 1) - 1 # goes from -1 to 1
    azi = 2*np.pi*((m*phi2) % 1) #goes from 0 to 2pi, angle on the xy plane
    phi = np.arccos(z) #angle between the z axis and the point , goes from 0 to pi
    x = np.cos(azi)*np.sin(phi)
    y = np.sin(azi)*np.sin(phi)
    
    return [rad*x,rad*y, rad*z]

def tinygolden3dangles(N, m, rad=0.5):
    """
    Calculates the tiny 3D golden angle for a certain set (N) and iteration (m) for sphere radius rad
    
    Args:
        N, m: integers
    Returns:
        [x,y,z] : 3d point
    """
    φ1 = 0.465571231876768
    φ2 = 0.682327803828019
    phi2 = 1/(N - 1 + 1/φ2)
    phi1 = phi2/(1 + φ1)
    
    z = 2*((m*phi1) % 1) - 1 # goes from -1 to 1
    azi = 2*np.pi*((m*phi2) % 1) #goes from 0 to 2pi, angle on the xy plane
    phi = np.arccos(z) #angle between the z axis and the point , goes from 0 to pi

    
    return azi, phi

def tiny_golden_2d(N):
    tau = (1 + np.sqrt(5))/2
    return np.pi/(tau + N - 1) * 180/np.pi