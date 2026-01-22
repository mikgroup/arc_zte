import numpy as np
from scipy.spatial import SphericalVoronoi


def cov_uniformity_metric(endpoints_in, n=10000, print_flag=False, seed=0):
    """
    Calculates the average distance to closest spoke from endpoints from n randomly generated points
    
    Args:
        endpoints: array of 3d points (numpoints, 3)
        n: number of test points
    Returns:
        average: float
    """
    endpoints = endpoints_in / (2*np.linalg.norm(endpoints_in, axis=-1)[:, None])

    # random testpoints on 0.5 sphere
    random_testpoints = randPointsOnSphere(n, r=0.5, seed=seed)

    # random endpoints on 0.5 sphere
    random_points = randPointsOnSphere(len(endpoints), r=0.5, seed=seed+1) # different seed to generate diff endpoints

    # generate n random testpoints and find sum of distances to nearest endpoint
    total  = 0
    for i in range(n):
        testpoint = random_testpoints[i]
        addition = min([distance(testpoint, e) for e in endpoints])
        total += addition
        
    # do same test with random distributed endpoints
    total_rand  = 0
    for i in range(n):
        testpoint = random_testpoints[i]
        addition = min([distance(testpoint, e) for e in random_points])
        total_rand += addition

    # average = total/n
    U = 1 - np.tanh((total/total_rand)- 1) # from AZTEK paper
    if print_flag:
        print('Total: ' + str(total))
        print('Total rand: ' + str(total_rand))

    return U

    
def randPointsOnSphere(num_points, r=0.5, seed=0):
    """
    Generates a random 3d point with radius 0.5 from uniformly random spherical angles
    
    Returns:
        [num_points, 3]
    """
    # azi = 2*np.pi*np.random.uniform(size=num_points)
    # phi = np.pi*np.random.uniform(size=num_points)

    np.random.seed(seed) 
    points = np.random.normal(0, 1, size=(num_points, 3))
    points /=  np.linalg.norm(points, axis=-1, keepdims=True)
    points *= r

    return points



def distance(p1, p2):
    """
    Calculates the distance between 2 points in 3d
    
    Args:
        p1, p2: 3d points
    Returns:
        distance : float
    """
    return np.linalg.norm(np.array(p1)-np.array(p2))


def refocusing_metrics(coords_seg, T2_star=60, TR=2.3, refocus_level=1, num_TR_dephasing=2, print_flag=False):
    '''
    coords dims = [nSpokes, nROpts, 3] - NOTE: this should be one segment, since assumes no gradient rampdown for these coords
    T2_star in ms
    TR in ms
    refocus level in cycles/voxel (kmax is 0.5 cycles/voxel). The threshold below which coherences are considered refocused
    num_TR_dephasing = number of TRs after the DAQ TR when coherences are considered as still dephasing, instead of refocusing

    '''
    [nSpokes, nRO, nDims] = coords_seg.shape

    if print_flag:
        print('Number of spokes in one segment is ' + str(nSpokes))
    coords = coords_seg / (2 * np.max(np.linalg.norm(coords_seg, axis=-1)))

    T2_decay_num_TRs = int(3 * T2_star/ TR) # number of TRs beyond which a coherence is negligible

    kstronauts = np.empty((nSpokes, nRO, 3))
    numRefocus = 0
    refocus_TR_list = []
    refocus_coherence_idx_list = []
    refocus_energy_list = []

    for i in range(nSpokes):
        spoke = coords[i] 
        kstronauts[i] = spoke # new kstronaut

        for j in range(i): # for all previous kstronauts

            # Current location is current spoke displaced by kstronauts endpoints
            kstronauts[j] = spoke + kstronauts[j][-1, :][None]

            # Check whether kstronaut is within refocusing phase level and if kstronaut TR # is within relevant region
            if (any(np.linalg.norm(kstronauts[j], axis=-1) < refocus_level) and 
                    num_TR_dephasing < i-j <= T2_decay_num_TRs):

                numRefocus += 1
                refocus_TR_list.append(i)
                refocus_coherence_idx_list.append(j)
                refocus_energy_list.append(min(np.linalg.norm(kstronauts[j], axis=-1))) 

                if print_flag:
                    print("Coherence " + str(j) + " refocused during TR " + str(i) + " at distance " + str(min(np.linalg.norm(kstronauts[j], axis=-1))))

    
    num_TRs_with_refocusing = len(np.unique(np.array(refocus_TR_list)))
    if len(refocus_energy_list) > 0:
        mean_refocusing_energy = np.mean(np.array(refocus_energy_list))
        worst_refocusing_instance = np.min(np.array(refocus_energy_list))
    else:
        mean_refocusing_energy = None
        worst_refocusing_instance = None

    if print_flag:
        print('Percentage of TRs with refocusing: ' + str(num_TRs_with_refocusing / nSpokes))

    return num_TRs_with_refocusing, mean_refocusing_energy, worst_refocusing_instance


def percentage_TRs_with_refocusing_metric(coords, **kwargs):

    metrics = refocusing_metrics(coords, **kwargs)

    return metrics[0]