from tqdm import tqdm
import numpy as np

from .mat import rotation_mat_axis_angle, rotation_matrix_from_vectors
from .grad import grad_to_ksp_coords_norm, ksp_coords_to_grad
from .rotate_spokes import Rotate, single_spoke_arc
from .golden_angles import tiny_golden_2d, tinygolden3dCoords

class FurthestDist_CostFunction(Rotate):
    '''Optimization based view ordering that maximizes coverage of spoke endpoints 
    and penalizes history of coherences to prevent refocusing
    '''

    def __init__(self, lamda, arc_angle, res, nSpokes,
                 grad_dt=8e-6, TR=2.3e-3, nTestAngles=360):
        
        # Initialize general Arc-ZTE rotatio class
        super().__init__(arc_angle, res, nSpokes, grad_dt, TR)

        # Save parameters specific to schemes
        self.lamda = lamda #lamda value for cost function. higher for more uniformity but refocusing
        self.test_angles = np.linspace(0, 2*np.pi, nTestAngles)

        self.nTestAngles = len(self.test_angles)

        # Cosine and sine matrix for Rodrigues formula
        cosines = np.cos(self.test_angles)
        self.cosines = cosines
        self.A = np.tile(np.array([cosines]).transpose(), (1,3))
        
        sines = np.sin(self.test_angles)
        self.sines = sines
        self.B = np.tile(np.array([sines]).transpose(), (1,3))
        
        self.C = np.ones(self.A.shape) - self.B

        ## First spoke and corresponding gradients
        self.first_spoke_phys = single_spoke_arc(self.nPts+1, arc_angle=self.arc_angle, 
                                            kmax=self.kmax) # phys in cm^-1 - not used for any calculations
        self.first_grad = ksp_coords_to_grad(self.first_spoke_phys, self.grad_dt) # [3, nPts] in G/cm
        self.first_spoke = grad_to_ksp_coords_norm(self.first_grad, dt=self.grad_dt) # normalized to kmax 0.5

        # intialize arrays to accumulate all spokes and concatenated gradient waveforms
        self.spoke_arr = np.zeros((self.nSpokes, 3, self.nPts))
        self.spoke_arr[0] = self.first_spoke # store first spoke
        self.astronauts = np.zeros((self.nSpokes, 3, self.nPts))
        self.astronauts[0] = self.first_spoke
        self.grad_cat = np.copy(self.first_grad)

        # initialize spoke and grads that will be updated for each TR
        self.spoke = self.first_spoke
        self.grad = self.first_grad

        # Lists to store some metrics later
        self.diff_norms_list = []
        self.refocuses_list = []
        self.angle_best_list = [] # stores all theta_i angles chosen

    def rotate(self):

        for tr_num in tqdm(range(1, self.nSpokes), 'Rotation matrix calculation'):

            #find R1
            self.R1 = self.findR1(self.grad)
            self.R1_list.append(self.R1)

            # apply R1 to gradient, find resulting spoke
            self.grad_R1_rot = self.R1 @ self.grad
            self.spoke_R1_rot = grad_to_ksp_coords_norm(self.grad_R1_rot, self.grad_dt)
            
            #find R2
            self.R2 = self.findR2(self.spoke_R1_rot, tr_num)
            self.R2_list.append(self.R2)
            
            # apply R2 to gradient, find resulting spoke for next loop
            self.grad = self.R2 @ self.grad_R1_rot
            self.spoke = grad_to_ksp_coords_norm(self.grad, self.grad_dt) #[3, nPts]
            
            # store final result of R2 * R1 rotations
            self.spoke_arr[tr_num] = self.spoke
            self.grad_cat = np.concatenate((self.grad_cat, self.grad), axis=-1)

            # Update kstronauts
            self.astronauts[0:tr_num] = self.spoke[None] + self.astronauts[0:tr_num, :, -1, None] # update previous kstronauts
            self.astronauts[tr_num] = self.spoke # Should be redundant because added 0s previously


    def findR2(self, R1_spoke, tr_num):
        # u is axis of rotation = start direction of R1*spoke
        u = R1_spoke[:, 1] - R1_spoke[:, 0] 
        u = u/np.linalg.norm(u)
 
        v = R1_spoke[:, -1]
        #output R_theta_v's and plot each row to see if it looks like the cone

        #R_theta_v =  self.A @ v.T + self.B @ np.cross(u, v).T + self.C @  u.T*np.dot(u, v)
        R_theta_v = np.outer(self.cosines, v) + np.outer(self.sines, np.cross(u, v)) + np.outer((np.ones(self.cosines.shape) - self.cosines), u*np.dot(u, v))
        ####this above one should be the correct formula using outer products and making a 100 x 3
        
        # dims prev spokes = [numPrev, 3]
        prev_endpoints = self.spoke_arr[0:tr_num, :, -1] # dims [nPrev, 3]
        
        diffs = R_theta_v[:, None] - prev_endpoints[None] # dims [nAngles, nPrev, 3]
        diff_norms = (np.linalg.norm(diffs, axis=-1)) # dims [nAngles, nPrev]
        mean_distances = np.mean(diff_norms, axis=-1) # mean over all nPrev

        refocuses = np.zeros((self.nTestAngles))
        for i, testAngle in enumerate(self.test_angles):

            R_test = rotation_mat_axis_angle(u, testAngle)
            rot_spoke_test = (R_test @ R1_spoke) 

            astro = rot_spoke_test[None] + self.astronauts[0:tr_num, :, -1, None] # kstronauts from previous TRs
            norm_astro = (np.linalg.norm(astro, axis=1))

            # Adding norms across time and across all astronauts
            refocuses[i] = np.mean(np.log((norm_astro + np.finfo(np.float128).tiny)), axis=(0,1)) # to penalize refocusing until 1 (0.5=kmax)

        if tr_num < 20:
            # Noisy greedy for first few TRs to avoid local minimum and pure in-plane rotations - v2 txt files
            # min_index = np.argsort(cost)[20]            

            # we want to promote large distances and large norms of refocusing - v3 txt files
            cost = np.add(-1*mean_distances, -1*5*self.lamda*(refocuses)) # heavier penalty on refocusing

        else:
            # we want to promote large distances and large norms of refocusing
            cost = np.add(-1*mean_distances, -1*self.lamda*(refocuses))
        min_index = np.argmin(cost)


        angle_best = self.test_angles[min_index]
        self.refocuses_list.append([refocuses])
        self.diff_norms_list.append([mean_distances])

        # print(diff_norms[min_index])
        # print(refocuses[min_index])

        self.angle_best_list.append(angle_best)
        R_best = rotation_mat_axis_angle(u, angle_best)

        return R_best


         
class FurthestR2(Rotate):
    """_summary_

    Args:
        Rotate (_type_): _description_
    """

    def __init__(self, arc_angle, res, nSpokes, 
                 grad_dt=8e-6, TR=2.3e-3, nTestAngles=360):
        """_summary_

        Args:
            arc_angle (_type_): _description_
            res (_type_): _description_
            nSpokes (_type_): _description_
            grad_dt (_type_, optional): _description_. Defaults to 8e-6.
            TR (_type_, optional): _description_. Defaults to 2.3e-3.
            nTestAngles (int, optional): _description_. Defaults to 360.
        """
        
        # Initialize general Arc-ZTE rotatio class
        super().__init__(arc_angle, res, nSpokes, grad_dt, TR)

        # Specific parameters for scheme
        self.nTestAngles = nTestAngles
        self.test_angles = np.linspace(0, 2*np.pi, nTestAngles) # radians
        
    
    def findR2(self, R1_spoke, i):
        """Find R2 rotation matrix to get the endpoint of rotation spoke furthest
            away from prev endpoints

        Args:
            prev_spokes (np.ndarray): [nSpokes, 3, nReadoutPoints]
            nAngles (int): number of rotation angles to test
            R1_spoke (np.ndarray): [3, nReadoutPoints] spoke coords after rotation by R1

        Returns:
            float: best angle of rotation in radians
        """

        # u is axis of rotation = start direction of R1*spoke
        u = R1_spoke[:, 1] - R1_spoke[:, 0]  

        # initialize 
        dist_best = -1 
        angle_best = 0 # best rotation angle around u

        # loop through test angles and calculate mean distance from all previous spoke endpoints
        for i in range(self.nTestAngles):
            R_test = rotation_mat_axis_angle(u, self.test_angles[i])
            rot_spoke_test = R_test @ R1_spoke

            # norm across x,y,z which is dim 1
            prev_endpoints = self.spoke_arr[:, :, -1] # [nSpokes, 3]
            dist_to_endpoints = np.linalg.norm((rot_spoke_test[None, :, -1] - prev_endpoints), axis=1)
            # mean across all prev spokes
            mean_dist_endpoints = np.mean(dist_to_endpoints, axis=0)

            # store new best if needed
            if mean_dist_endpoints > dist_best:
                angle_best = self.test_angles[i]
                dist_best = mean_dist_endpoints

        R_best = rotation_mat_axis_angle(u, angle_best)
        return angle_best, R_best
    

    
class RandomR2(Rotate):
    """_summary_

    Args:
        Rotate (_type_): _description_
    """

    def __init__(self, arc_angle, res, nSpokes, grad_dt=8e-6, TR=2.3e-3, nTestAngles=360):
        """_summary_

        Args:
            arc_angle (_type_): _description_
            res (_type_): _description_
            nSpokes (_type_): _description_
            grad_dt (_type_, optional): _description_. Defaults to 8e-6.
            TR (_type_, optional): _description_. Defaults to 2.3e-3.
            nTestAngles (int, optional): _description_. Defaults to 360.
        """

        super().__init__(arc_angle, res, nSpokes, grad_dt, TR)

        # Generate nSpokes number of random angles
        self.nTestAngles = nTestAngles
        self.test_angles = np.linspace(0, 2*np.pi, nTestAngles)
        self.R2_angles = np.random.choice(self.test_angles, size=nSpokes)

    def findR2(self, R1_spoke, i):
        """_summary_

        Args:
            R1_spoke (_type_): _description_
            i (_type_): _description_

        Returns:
            _type_: _description_
        """
        # u is axis of rotation = start direction of R1*spoke
        u = R1_spoke[:, 1] - R1_spoke[:, 0] 

        return rotation_mat_axis_angle(u, self.R2_angles[i])

