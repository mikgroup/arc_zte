import numpy as np
import matplotlib.pyplot as plt

from .mat import rotation_matrix_from_vectors
from .grad import ksp_coords_to_grad, grad_to_ksp_coords, grad_to_ksp_coords_norm, grad_to_slew


def single_spoke_arc(nPts, arc_angle=60, kmax=0.5):
    """Returns coordinates of a single kspace spoke along the arc of a 2d circle

    Args:
        nPts (int): number of points along the spoke
        angle (float/int, optional): angle subtended by the arc in degrees
        kmax: location where endpoint of arc will hit

    Returns:
        np.ndarray: [3,nPts] array with coordinates of spoke
    """
    theta_end = arc_angle*np.pi/180
    theta = np.linspace(0, theta_end, nPts)

    r = kmax * np.sqrt(1/(2 - 2*np.cos(theta_end))) # radius to land on unit sphere

    kz = 0*np.ones((nPts)) # put arc along 2d circle
    kx = r*np.sin(theta)
    ky = r*np.cos(theta) - r
    spoke = np.array([kx, ky, kz])
    
    return spoke

def save_Rs_txt(scheme, txt_file_name):
    R1_list = scheme.R1_list
    R2_list = scheme.R2_list
    R_list = []
    for i in range(len(R1_list)):
        R_list.append(R2_list[i] @ R1_list[i])

    nrots = len(R_list)
    R_list = np.array(R_list)
    np.savetxt(txt_file_name, R_list.reshape(nrots,9))


class Rotate(object): 
    '''Abstract class for different theta_i selection schemes - Do not instantiate. 

    Args:
        arc angle: in degrees
        res: in cm. Prescribed image Resolution
        nSpokes (int)
        grad_dt: in sec. Sampling rate of gradients
        TR: in sec. Duration of each arc

    '''
    
    def __init__(self, arc_angle, res, nSpokes, grad_dt=8e-6, TR=2.3e-3):

        # Save relevant parameters
        self.arc_angle = arc_angle 
        self.arc_angle_rad = arc_angle * np.pi/180
        self.kmax = 1/(2*res) # in inverse cm
        self.nSpokes = nSpokes
        self.grad_dt = grad_dt # sampling rate of gradients
        self.TR = TR # duration of each spoke
        self.nPts = int(self.TR / self.grad_dt)

        # Initialize lists to store results
        self.R2_list = []
        self.R1_list = []

        ## First spoke and corresponding gradients
        self.first_spoke_phys = single_spoke_arc(self.nPts+1, arc_angle=self.arc_angle, 
                                            kmax=self.kmax) # phys in cm^-1 - not used for any calculations
        self.first_grad = ksp_coords_to_grad(self.first_spoke_phys, self.grad_dt) # [3, nPts] in G/cm
        self.first_spoke = grad_to_ksp_coords_norm(self.first_grad, dt=self.grad_dt) # normalized to kmax 0.5

        # intialize arrays to accumulate all spokes and concatenated gradient waveforms
        self.spoke_arr = np.zeros((self.nSpokes, 3, self.nPts))
        self.spoke_arr[0] = self.first_spoke # store first spoke
        self.grad_cat = np.copy(self.first_grad)

        # initialize spoke and grads that will be updated for each TR
        self.spoke = self.first_spoke
        self.grad = self.first_grad

        
    def rotate(self):

        for i in np.arange(1, self.nSpokes): # for i TRs

            #find R1
            self.R1 = self.findR1(self.grad)
            self.R1_list.append(self.R1)

            # apply R1 to gradient, find resulting spoke
            self.grad_R1_rot = self.R1 @ self.grad
            self.spoke_R1_rot = grad_to_ksp_coords(self.grad_R1_rot, self.grad_dt)
            
            #find R2
            self.R2 = self.findR2(self.spoke_R1_rot, i)
            self.R2_list.append(self.R2)
            
            # apply R2 to gradient, find resulting spoke for next loop
            self.grad = self.R2 @ self.grad_R1_rot
            self.spoke = grad_to_ksp_coords(self.grad, self.grad_dt)
            
            # store final result of R2 * R1 rotations
            self.spoke_arr[i] = self.spoke
            self.grad_cat = np.concatenate((self.grad_cat, self.grad), axis=-1)

  
    def getSpokeArr(self):
        return self.spoke_arr
    
    def getGrads(self):
        return self.grad_cat
    
    def getSlew(self):
        return grad_to_slew(self.grad_cat, self.grad_dt)
    
    def get_Rlist(self):

        R1_list = self.R1_list
        R2_list = self.R2_list
        R_list = []
        for i in range(len(R1_list)):
            R_list.append(R2_list[i] @ R1_list[i])

        return R_list
    
        
    def findR1(self, grad):
        """Find R1 rotation matrix to match start dir to end dir of grad
        Can also be used on kspace spokes

        Args:
            grad (np.ndarray): [3, nReadoutPoints]

        Returns:
            np.ndarray: [3,3] rotation matrix
        """
        
        # direction of gradient at end of spoke
        v = grad[:, -1] - grad[:, -2]
        v /= np.linalg.norm(v)

        # direction of gradient at beginning of spoke
        u = grad[:, 1] - grad[:, 0]
        u /= np.linalg.norm(u) 
        
        # rotation matrix to align start direction to end direction
        R1 = rotation_matrix_from_vectors(u, v)

        return R1
