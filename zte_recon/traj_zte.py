import numpy as np
from phyllo_endpoints import phyllo_endpoints_merlin
from arc_zte_traj import calc_all_curved_grads, get_segment_waveform_from_kacq_file, rotate_integrate_all_segments, read_params_from_kacq
from scipy.io import loadmat

'''
Classes to only set up trajectories without data files
Standard, AZTEK, and Arc-ZTE require external files from acquisition
'''

class Traj_ZTE():
    ''' DO NOT Instantiate - just a parent class '''   

    def __init__(self, opxres, waspi_scale=8, dead_time=0):
        self.opxres = opxres 
        self.waspi_scale = waspi_scale
        self.highMerge = 6
        self.dead_time = dead_time # if 0, this trajectory doesn't have dead time gap

    def setup_waspi_spokes(self):
        '''
        Extract waspi spokes from ksp_full and coord_full
        '''
        self.coord_waspi = self.coord_full[0:self.nInSpokes, 0:(self.highMerge+4)*self.waspi_scale, :]


    def setup_hires_spokes(self):
        '''
        Shared by all radial ZTE classes. 
        Extract sampled (hires/RUFIS) spokes from ksp_full and coord_full
        Rerun this function if nPtsPerSpoke changes
        '''
        coord_sampl_hires = self.coord_full[self.nInSpokes:, 0:self.opxres]

        # Remove 0s from coord_sampl
        self.coord_sampl_hires = coord_sampl_hires[:, self.dead_time:]



class Traj_Product_ZTE(Traj_ZTE):

    def __init__(self, opxres, matlab_save_dir, waspi_scale=8):
         
        super().__init__(opxres, waspi_scale)

        ## Load parameters file
        try:
            param_file = loadmat(matlab_save_dir +'/param.mat')
        except:
            raise FileNotFoundError(f"Param.mat file not found in {matlab_save_dir}")
        self.nInSpokes = param_file['N']['spokeslow'][0][0][0][0] # only WASPI spokes
        self.nSpokes = param_file['N']['spokeshig'][0][0][0][0] # only RUFIS spokes
        

        ## Read coords from matlab directory
        try:
            ktraj_file = loadmat(matlab_save_dir + '/coord.mat')
            self.coord_full = (ktraj_file['K']).transpose(1,0,2)
        except: 
            print("Coord.mat file not found in matlab_save_dir")

        self.setup_hires_spokes()
        self.setup_waspi_spokes()



class Traj_CalcPhyllo_ZTE(Traj_ZTE):

    def __init__(self, opxres, smoothness, gm_flag=True, segsPerInterleaf=1, spokes_per_seg=384, 
                 num_segs=None, num_segs_lowres=None, waspi_scale=8):
        
        super().__init__(opxres, waspi_scale)

        self.smoothness = smoothness
        self.opxres = opxres
        self.gm_flag = gm_flag
        self.segsPerInterleaf = segsPerInterleaf
        self.spokes_per_seg = spokes_per_seg
    
        if num_segs is None or num_segs_lowres is None:
            raise ValueError("Need to provide both num_segs and num_segs_lowres to calculate Phyllo traj")
        else:
            self.num_segs = num_segs
            self.num_segs_lowres = num_segs_lowres

        self.nSpokes = self.num_segs * self.spokes_per_seg
        self.nInSpokes = self.num_segs * self.spokes_per_seg

        self.get_coords()
        self.setup_hires_spokes()
        self.setup_waspi_spokes()


    def get_coords(self):
        # Calculate phyllotaxis coordinates for num_segs + num_segs_lowres
        # endpoints scaled [-0.5, 0.5] and have shape [nSpokes, 3]
        endpoints_hires = phyllo_endpoints_merlin(spokes_per_seg=self.spokes_per_seg * self.segsPerInterleaf, 
                                smoothness=self.smoothness, 
                                spokes_total=self.nSpokes, gm=self.gm_flag)
        endpoints_waspi = phyllo_endpoints_merlin(spokes_per_seg=self.spokes_per_seg * self.segsPerInterleaf, 
                        smoothness=self.smoothness, 
                        spokes_total=self.nInSpokes, gm=self.gm_flag)
        # Concatenate endpoints
        endpoints = np.concatenate((endpoints_waspi, endpoints_hires), axis=0)
        
        # Setup coordinates
        coords = np.linspace(0, 1, self.opxres)[None, :, None] * endpoints[:, None]

        # Normalize from -0.5 to 0.5
        coords /= (2*abs(coords).max())

        # Scale to nReadoutPts/2
        coords *= self.opxres

        # Scale WASPI spokes by waspi factors
        coords[0:self.nInSpokes] *= 1/self.waspi_scale

        # Save as attribute
        self.coord_full = coords
                 

    
class Traj_AZTEK_from_endpoints(Traj_ZTE):
    '''
    AZTEK trajectory using endpoints file saved in EPIC
    '''

    def __init__(self, opxres, endpoints_txt_path, waspi_scale=8):

        super().__init__(opxres, waspi_scale)

        self.endpoints_txt_path = endpoints_txt_path
        self.opxres = opxres

        self.get_coords()
        self.setup_hires_spokes()
        self.setup_waspi_spokes()

    def get_coords(self):
        # Read endpoints file
        if self.endpoints_txt_path is None:
            return ValueError('Must provide endpoints text file')
        
        # The file is read and its data is stored
        data = open(self.endpoints_txt_path, 'r').read()
        data = data.split()

        endpoints = []
        for i in range(len(data)):
            # Only first three numbers per line are stored (ignore slew and rot sign)
            if i >= 5 and (i % 5 == 0 or i % 5 == 1 or i % 5 == 2):
                endpoints.append(int(data[i]))
        
        # convert into np array with shape: [nSpokes, 3]
        endpoints = np.array(endpoints).reshape(-1, 3)

        # Setup coordinates
        coords = np.linspace(0, 1, self.opxres)[None, :, None] * endpoints[:, None]

        # Normalize from -0.5 to 0.5
        coords /= (2*abs(coords).max())

        # Scale to nReadoutPts/2
        coords *= self.opxres

        # Save as attribute
        self.coord_full = coords



class Traj_ArcZTE_from_kacq(Traj_ZTE):
    '''
    Arc ZTE trajectory using saved kacq file and pre-computed rotation matrix files
    grad_dt_sampling and dt_sampling in us

    '''

    def __init__(self, opxres, seg_rot_file, grad_segment_file, grad_dt_sampling=4, 
                 waspi_scale=8):
        
        super().__init__(opxres, waspi_scale)
        
        # Save Arc-ZTE arguments
        self.seg_rot_file = seg_rot_file
        self.grad_dt_sampling = grad_dt_sampling

        # Grad file saved by EPIC - testing for now
        if grad_segment_file is not None: 
            self.grad_segment_file = grad_segment_file
        else:
            raise ValueError("Requires kacq file to be provided as grad_segment_file")

        acq_params = read_params_from_kacq(txt_file=grad_segment_file)
        self.spokes_per_seg = acq_params['spokes_per_seg']
        self.a_grad = acq_params['a_grad']
        self.points_per_spoke = acq_params['points_per_spoke']
        self.points_before_curve = acq_params['points_before_curve']
        self.num_segs = acq_params['num_segs']
        self.num_segs_lowres = acq_params['num_segs_lowres']
        self.dt_sampling = acq_params['dt_sampling']
        self.arc_angle = acq_params['arc_angle']
        self.params = acq_params

        self.get_coords_using_kacq_file()
        self.setup_hires_spokes()
        self.setup_waspi_spokes()


    def get_coords_using_kacq_file(self):
        '''
        Find kspace coordinates using gradient file saved in EPIC code
        '''
        gx, gy, gz = get_segment_waveform_from_kacq_file(self.grad_segment_file)
        self.grad_first_seg = np.stack((gx, gy, gz))


        self.spokes_all = rotate_integrate_all_segments(self.grad_first_seg, self.seg_rot_file, 
                            self.points_per_spoke, self.points_before_curve, 
                            self.dt_sampling, self.num_segs, self.num_segs_lowres, 
                            self.spokes_per_seg, self.waspi_scale, self.opxres, self.grad_dt_sampling)
        
        ## Calculate correct scaling for reconstructing correcting FOV
        self.coord_full = np.copy(self.spokes_all[:, 0:self.opxres]) # only sampling part of gradients

        acquired_res = self.opxres # TODO: check that this is giving the correct resolution
        scale_factor_fov = acquired_res * 1/(2*self.coord_full.max()) # digitize acquired resolution
        self.coord_full *= scale_factor_fov




class Traj_Calc_ArcZTE(Traj_ZTE):

    '''Calculate Arc-ZTE trajectory given parameters
    Does not account for any timing errors in real scanner, so not recommended to use for recon
    '''

    def __init__(self, opxres, arc_angle, spoke_rot_file, seg_rot_file, params, 
                 grad_dt_sampling=4, waspi_scale=8):
        
        super().__init__(opxres, waspi_scale)

        # Save Arc-ZTE arguments
        self.seg_rot_file = seg_rot_file
        self.grad_dt_sampling = grad_dt_sampling
        self.waspi_scale = waspi_scale
        self.arc_angle = arc_angle

        # Save Arc-ZTE arguments
        self.seg_rot_file = seg_rot_file
        self.grad_dt_sampling = grad_dt_sampling
        self.spoke_rot_file = spoke_rot_file

        self.spokes_per_seg = params['spokes_per_seg']
        self.a_grad = params['a_grad']
        self.points_per_spoke = params['points_per_spoke']
        self.points_before_curve = params['points_before_curve']
        self.num_segs = params['num_segs']
        self.num_segs_lowres = params['num_segs_lowres']
        self.dt_sampling = params['dt_sampling']
        self.params = params
        self.nInSpokes = self.num_segs_lowres * self.spokes_per_seg

        self.setup_all_curved_grads()
        self.setup_hires_spokes()
        self.setup_waspi_spokes()


    def setup_all_curved_grads(self):

        gamma_bar = 4.257588
        arc_angle_rad = self.arc_angle * np.pi/180.0
        nSegs_total = self.num_segs + self.num_segs_lowres

        ## Read spoke rotations from file
        R_file = np.zeros((self.spokes_per_seg, 9), dtype=float)
        count = 0

        with open(self.spoke_rot_file, "r") as file1:
            
            for line in file1.readlines():
                line_list = line.split(" ")
                if (count >= self.spokes_per_seg):
                    break
                for i in range(9):
                    R_file[count, i] = float(line_list[i])
                    
                count += 1

        ## Read segment rotations from file
        M_file = np.zeros((nSegs_total, 9), dtype=float)
        count = 0

        with open(self.seg_rot_file, "r") as file1:
            
            for line in file1.readlines():
                line_list = line.split(" ")
                if (count >= nSegs_total):
                    break
                for i in range(9):
                    M_file[count, i] = float(line_list[i])
                    
                count += 1

        ## Using rotation matrices from file, find gradients for each spoke in physical units G/cm
        # then use cumsum to find spoke_arr in physical units of cm^-1
        spoke_arr = np.zeros((self.spokes_per_seg, 3, self.points_per_spoke - self.points_before_curve)) # one segment of spokes
        grad_cat = np.zeros((self.spokes_per_seg, 3, self.points_per_spoke))

        delta_k = gamma_bar * self.a_grad * self.dt_sampling * 0.001
        delta_angle_rad = arc_angle_rad / (self.points_per_spoke)
        r = np.sqrt((delta_k**2)/(2 - 2*np.cos(delta_angle_rad)))

        first_spoke = single_spoke_arc(self.points_per_spoke+1, angle=self.arc_angle, r=r)
        first_grad = np.diff(first_spoke)/(self.dt_sampling*0.001*gamma_bar)

        grad_cat[0] = first_grad
        spoke_arr[0] = np.cumsum(grad_cat[0, :, self.points_before_curve:], axis=-1) * gamma_bar * self.dt_sampling * 0.001

        ## Rotate the gradient by spokeRots to find one segment
        for i in range(self.spokes_per_seg-1):
            # rotate gradients
            grad_cat[i+1] = R_file[i].reshape(3,3) @ first_grad
            first_grad = grad_cat[i+1]
            
            # solve for spoke
            spoke_arr[i+1] = np.cumsum(grad_cat[i+1, :, self.points_before_curve:], axis=-1) * gamma_bar * self.dt_sampling * 0.001
            
        # reformat grad_cat to have continuous waveforms
        grad_cat_flat = grad_cat.transpose(1,0,2)
        grad_cat_flat = grad_cat_flat.reshape(3, self.spokes_per_seg*self.points_per_spoke)

        ## Rotate the gradients by segRots, then find spoke_arr
        spokes_full_arc = np.zeros((nSegs_total*self.spokes_per_seg, 3, self.points_per_spoke - self.points_before_curve))
        grads_full_arc = np.zeros((nSegs_total*self.spokes_per_seg, 3, self.points_per_spoke))

        for i in range(nSegs_total):

            if i < self.num_segs_lowres:
                scale = (1/self.waspi_scale)
            else:
                scale = 1
            
            if i == 0: # don't rotate first segment
                for j in range(self.spokes_per_seg):
                    spokes_full_arc[i*self.spokes_per_seg + j] = scale * spoke_arr[j]
                    grads_full_arc[i*self.spokes_per_seg + j] = scale * grad_cat[j]
            
            else:
                for j in range(self.spokes_per_seg):
                    grads_full_arc[i*self.spokes_per_seg + j] = scale * (M_file[i-1].reshape(3,3) @ grad_cat[j])
                    spokes_full_arc[i*self.spokes_per_seg + j] = np.cumsum(grads_full_arc[i*self.spokes_per_seg + j, :, 
                                                            self.points_before_curve:], axis=-1) * gamma_bar * self.dt_sampling * 0.001
                    

        ## Saved attributes include all spokes/gradient waveforms s(not just sampled parts)
        ## Scaled by waspi factor
        self.spokes_all = spokes_full_arc.transpose(0,2,1) 
        self.grads_all = grads_full_arc.transpose(0,2,1)
        self.grads_flat_one_seg = grad_cat_flat     

        ## Calculate correct scaling for reconstructing correcting FOV
        self.coord_full = np.copy(self.spokes_all[:, 0:self.opxres]) # only sampling part of gradients

        acquired_res = self.opxres # TODO: check that this is giving the correct resolution
        scale_factor_fov = acquired_res * 1/(2*self.coord_full.max()) # digitize acquired resolution
        self.coord_full *= scale_factor_fov 



def single_spoke_arc(nReadoutPts, angle=60, r=None, sign_angle=-1):
    """Returns coordinates of a single kspace spoke along the arc of a 2d circle

    Args:
        nReadoutPts (int): number of points along the spoke
        angle (float/int, optional): angle subtended by the arc in degrees

    Returns:
        np.ndarray: [3,1] array with coordinates of spoke
    """
    theta_end = angle*np.pi/180
    theta = np.linspace(0, theta_end, nReadoutPts)

    # choose radius so end of spoke lands on [-0.5, 0.5] sphere
    if (r == None):
        r = np.sqrt(1/(2-2*np.cos(theta_end))) # radius to land on unit sphere
        r = r / 2.0

    kz = 0*np.ones((nReadoutPts)) # put arc along 2d circle
    kx = r*np.sin(sign_angle*theta)
    ky = -r*np.cos(sign_angle*theta) + r
    spoke = np.array([kx, ky, kz])
    
    return spoke
