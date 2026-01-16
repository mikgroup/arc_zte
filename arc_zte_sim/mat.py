import numpy as np

def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that when applied to vec1 aligns it to vec2

    Args:
        vec1 (ndarray): [3,1] "source" vector
        vec2 (ndarray): [3,1] "destination" vector

    Returns:
        np.ndarray: [3,3] rotation matrix which when applied to vec1 aligns it with vec2
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_mat_axis_angle(u, angle):
    """Find rotation matrix that rotates around axis u by angle in radians

    Args:
        u (np.ndarray): [3,1] axis of rotation
        angle (float): angle of rotation in radians

    Returns:
        np.ndarray: [3,3] rotation matrix
    """
    u /= np.linalg.norm(u)
    [ux, uy, uz] = u
    t = angle
    R = np.array([[np.cos(t) + (ux**2)*(1-np.cos(t)), ux*uy*(1-np.cos(t)) - uz*np.sin(t), ux*uz*(1-np.cos(t)) + uy*np.sin(t)], 
                    [uy*ux*(1-np.cos(t))+uz*np.sin(t), np.cos(t) + (uy**2)*(1-np.cos(t)), uy*uz*(1-np.cos(t)) - ux*np.sin(t)],
                    [uz*ux*(1-np.cos(t))-uy*np.sin(t), uz*uy*(1-np.cos(t))+ux*np.sin(t), np.cos(t) + (uz**2)*(1-np.cos(t))]])

    return R