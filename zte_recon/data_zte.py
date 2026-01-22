import numpy as np
import h5py
from scipy.io import loadmat
import sigpy as sp

from .arc_zte_traj import get_segment_waveform_from_kacq_file, rotate_integrate_all_segments
from .phyllo_endpoints import phyllo_endpoints_merlin
from .bellows import read_bellows_data, create_bins, resample_bellows_data
from .recon_gridding import recon_adjoint_postcomp_coilbycoil

from .util.nufft_util import nufft_adjoint_postcompensation
from .util.grad_corr import shift_coord
from .util.cc import apply_cc_ksp, find_A_cc_rovir

class Data_ZTE():
    '''
    Do not instantiate. Just template data class for different types of ZTE data
    '''
    def __init__(self, h5_save_dir, nPtsPerSpoke=None, rBW=31.25e3, 
                 FOV_scale=1, FOV_scale_dir=[1,1,1], ndrop_ROpts=1):
        '''
        h5_save_dir should have ksp.h5 and param.mat
        nPtsPerSpoke to choose points < opxres. Defaults to acquired opxres. 
        
        FOV_scale_dir scales differently in different directions
        FOV_scale scales in all directions equally

        coord_full not scaled by FOV scale
        coord_radial_spokes - FOV scaled
        coord_sampl_hires - Same as coord_radial_spokes but removes dead time gap 

        '''

        # Save all passed arguments
        self.h5_save_dir = h5_save_dir
        self.rBW = rBW
        self.FOV_scale = FOV_scale
        self.FOV_scale_dir = FOV_scale_dir
        self.ndrop_ROpts = ndrop_ROpts
        
        # Read parameters file from h5_save_dir
        self.read_params_file()
        self.load_ksp_from_file()

        if nPtsPerSpoke is None:
            nPtsPerSpoke = self.opxres
            print(f"Setting nPtsPerSpoke to opxres {self.opxres}")
        self.nPtsPerSpoke = nPtsPerSpoke


    def read_params_file(self):
        '''Load params.mat and save parameters in class 
        '''
        ## Load parameters file
        try:
            param_file = loadmat(self.h5_save_dir +'/param.mat')
        except:
            raise FileNotFoundError(f"Param.mat file not found in {self.h5_save_dir}")
        
        # Save all relevant parameters
        self.nInSpokes = param_file['N']['spokeslow'][0][0][0][0] # only WASPI spokes
        self.feextra = param_file['N']['feextra'][0][0][0][0] # dead time gap
        self.opxres = param_file['N']['da_xres'][0][0][0][0]
        self.FOV = (0.1) * param_file['N']['fov'][0][0][0][0] # UNITS of self.FOV: in cm
        self.nSpokes = param_file['N']['spokeshig'][0][0][0][0] # only RUFIS spokes
        self.waspi_scale = param_file['N']['WASPI_factor'][0][0][0][0] # default is 8
        self.spokes_per_seg = np.uint32(param_file['N']['spokesPerSeg'][0][0][0][0])
        self.tr = (1e-6) * (param_file['N']['act_tr'][0][0][0][0] / self.spokes_per_seg) # in s
        
        self.num_segs = np.uint32(self.nSpokes / self.spokes_per_seg)
        self.num_segs_lowres = np.uint32(self.nInSpokes / self.spokes_per_seg)

        self.highMerge = 6 # used only if both RUFIS and WASPI spokes in recon

        # Dead time total is acquired dead time + dropped points due to coil ringing
        self.dead_time = -1*self.feextra + self.ndrop_ROpts
        

    def load_ksp_from_file(self):
        '''Load full k-space from h5 file
        '''
        print('Loading ksp from h5 dir: ' + self.h5_save_dir)
        f = h5py.File(self.h5_save_dir + 'ksp.h5', 'r')
        self.ksp_full = (np.array(f['real']) + 1j*np.array(f['imag']))[:, 0] 
        self.nCoils = self.ksp_full.shape[0]


    def get_coords(self):
        '''This function should not be called. Children classes will calculate self.coords_full here
        '''
        raise NotImplementedError("Do not instantiate Data_ZTE directly")
    

    def setup_waspi_spokes(self):
        '''
        Extract waspi spokes from ksp_full and coord_full
        '''
        ksp_lores = self.ksp_full[:, 0:self.nInSpokes, 0:(self.highMerge+4)*self.waspi_scale]

        self.ksp_waspi = ksp_lores
        self.coord_waspi = self.coord_full[0:self.nInSpokes, 0:(self.highMerge+4)*self.waspi_scale, :]

        # Scale coord_waspi
        for i in range(3):
            self.coord_waspi[..., i] *= self.FOV_scale_dir[i]
        self.coord_waspi *= self.FOV_scale


    def setup_hires_spokes(self):
        '''
        Shared by all radial ZTE classes. 
        Extract sampled (hires/RUFIS) spokes from ksp_full and coord_full
        Rerun this function if nPtsPerSpoke changes
        '''
        self.sample_shift = 0 # no grad correction to begin with

        nRO = self.nPtsPerSpoke

        # in case not done, zero out center data of hires spokes until HighMerge
        self.ksp_hires = self.ksp_full[:, self.nInSpokes:, self.dead_time:self.nPtsPerSpoke]

        coord_sampl_hires = self.coord_full[self.nInSpokes:, 0:self.nPtsPerSpoke]

        # Scale coord_sampl_hires
        for i in range(3):
            coord_sampl_hires[..., i] *= self.FOV_scale_dir[i]
        coord_sampl_hires *= self.FOV_scale

        # Setup diameter spokes 
        coord_diameter = np.zeros((coord_sampl_hires.shape[0], 
                                    2*coord_sampl_hires.shape[1]-1, 3))
        coord_diameter[:, :nRO-1] = -1*np.flip(coord_sampl_hires[:, 1:nRO], axis=1)
        coord_diameter[:, nRO-1:] = coord_sampl_hires[:, 0:nRO]

        # Remove 0s from coord_sampl
        self.coord_sampl_hires = coord_sampl_hires[:, self.dead_time:]
        # Keep 0s in coord_radial_spokes
        self.coord_radial_spokes = coord_sampl_hires
        self.coord_diameter = coord_diameter

            
    def gradient_corr(self, sample_shift):
        '''
        Gradient delay correction applied to all saved coordinates except coord_full
        '''
        if self.sample_shift == 0: # only if new sample_shift

            self.sample_shift = sample_shift
            print('Applying gradient delay correction of ' + str(self.sample_shift))

            # applying gradient delay correction on high-res spokes
            self.coord_radial_spokes = shift_coord(sample_shift, self.coord_radial_spokes)
            self.coord_sampl_hires = shift_coord(sample_shift, self.coord_sampl_hires)

            # applying gradient delay correction on waspi spokes
            self.coord_waspi = shift_coord(sample_shift, self.coord_waspi)
        
        else:
            print('Gradient delay correction of ' + str(self.sample_shift) + 
                  ' was already applied. Run setup_hires_spokes() and setup_waspi_spokes() again first.')
               

    def coil_compress(self, nVirtualCoils, cc_mode='pca', mask_sphere=None):
        '''
        Apply coil compression on ksp_hires
        '''
        ## PCA coil compressions
        if cc_mode == 'pca':
            # ksp_inner_flat = self.ksp_waspi.reshape(self.nCoils, -1)
            # print('Solving for compression matrix')
            # A_cc = find_A_cc_pca(ksp_inner_flat, nVirtualCoils)
            # print('Applying compression matrix')
            # self.ksp_cc = apply_cc_ksp(self.ksp_hires, A_cc)
            # self.ksp_waspi_cc = apply_cc_ksp(self.ksp_waspi, A_cc)

            # Use BART for coil compression
            raise NotImplementedError

        ## Rovir coil compression
        elif cc_mode == 'rovir':
            if mask_sphere is None:
                raise ValueError('mask_sphere must be provided for Rovir cc')
            self.mask_sphere = mask_sphere
    
            # Calculate lowres coil images using WASPI data
            coil_ims_lowres = nufft_adjoint_postcompensation(self.ksp_waspi, 
                                                            self.coord_waspi, 
                                                            oshape=[self.nCoils, *mask_sphere.shape], 
                                                            device=-1)
            self.coil_ims_lowres = coil_ims_lowres

            print('Solving for rovir compression matrix')
            A_cc = find_A_cc_rovir(coil_ims_lowres, mask_sphere, nVirtualCoils)
            print('Applying compression matrix')
            self.ksp_cc = apply_cc_ksp(self.ksp_hires, A_cc)
            self.ksp_waspi_cc = apply_cc_ksp(self.ksp_waspi, A_cc)

            # print('Saving visualization of rovir')
            # self.rovir_roi_coils = self.coil_ims_lowres * mask_sphere
            # self.rovir_intf_coils = self.coil_ims_lowres * (1-mask_sphere)

        else:
            raise NotImplementedError
        
        self.A_cc = A_cc
        self.nVirtualCoils = nVirtualCoils
        
        return self.ksp_cc, self.ksp_waspi_cc
    
    def visualize_rovir_coils(self, lowres_nPts = 48):
        if self.mask_sphere is None:
            raise ValueError('mask_sphere not set. Cannot visualize rovir coils')
        
        # Interpolate mask_sphere
        mask_sphere_interp = sp.ifft(sp.resize(sp.fft(self.mask_sphere, axes=[0,1,2]), 
                                       [lowres_nPts, lowres_nPts, lowres_nPts]), axes=[0,1,2])

        # Run recon at slightly higher res for visualization
        coil_ims_lowres = nufft_adjoint_postcompensation(self.ksp_hires[:, :, :lowres_nPts], 
                                                   self.coord_sampl_hires[:, :lowres_nPts], 
                                                    oshape=[self.nCoils, *mask_sphere_interp.shape], 
                                                    device=-1)
        return coil_ims_lowres * mask_sphere_interp[None]
    

    def combine_waspiHires_flat(self, use_cc_ksp=False, spokes_subset=None):
        '''Combine waspi and hires spokes (sampled points) to make 1 flat ksp with coords
        Set use_cc_ksp to True if using coil combined ksp for this
        Input indices into spokes_subset to only choose a subset of the hires spokes
        '''
        # use coil compressed ksp if use_cc_ksp is True
        if use_cc_ksp and hasattr(self, 'ksp_cc'):
            ksp_sampl = self.ksp_cc
            ksp_waspi_sampl = self.ksp_waspi_cc
            num_coils_data = self.nVirtualCoils

        elif use_cc_ksp and hasattr(self, 'ksp_cc') == False:
            raise ValueError('Coil compression has not been run. Use_cc_ksp cannot be True')
        else:
            ksp_sampl = self.ksp_hires
            ksp_waspi_sampl = self.ksp_waspi
            num_coils_data = self.nCoils

        coord_sampl = self.coord_sampl_hires
        coord_waspi_sampl = self.coord_waspi

        # Choose subset of spokes if inputted
        if spokes_subset is not None:
            ksp_sampl = ksp_sampl[:, spokes_subset]
            coord_sampl = coord_sampl[spokes_subset]

        coord_flat = np.concatenate((coord_waspi_sampl.reshape(-1, 3), 
                                                    coord_sampl.reshape(-1, 3)), axis=0)
        ksp_flat = np.concatenate((ksp_waspi_sampl.reshape(num_coils_data, -1), 
                                                    ksp_sampl.reshape(num_coils_data, -1)), axis=-1)

        return ksp_flat, coord_flat
    
    
    def bin_data_bellows(self, bellows_path, nBins, margin_discard=3, use_ksp_cc=False):
        '''
        Bin ksp and coord of only RUFIS spokes into respiratory bins
        nBins : number of respiratory bins
        margin_discard : +/- percentile near peaks and valleys to discard
        '''
        if self.tr is None:
            raise ValueError('self.tr is None, but needs to be set')
        self.nBins = nBins
        waveform = read_bellows_data(bellows_path)
        bellows_interp, cropped_waveform, times_sequence, times_crop_wave = resample_bellows_data(
                                                                            waveform, self.num_segs, 
                                                                            self.num_segs_lowres, self.tr, 
                                                                            self.spokes_per_seg, 
                                                                            bellows_sampl_rate=0.04
                                                                        )
        bins, spokes_idx_per_bin = create_bins(bellows_interp, nBins, nWaspi=self.nInSpokes, 
                                               margin_discard=margin_discard)
        # Store 
        self.bins = bins
        self.bellows_interp = bellows_interp
        self.cropped_waveform = cropped_waveform # before interpolation
        self.times_sequence = times_sequence # times for bellows_interp
        self.times_crop_wave = times_crop_wave # times for cropped_waveform

        nSpokesPerBin = np.min([len(bin_idx) for bin_idx in spokes_idx_per_bin])
        self.nSpokesPerBin = nSpokesPerBin

        # Bin ksp and coords into bins and make sure same number of spokes in each bin
        nRO = self.ksp_hires.shape[-1]
        if use_ksp_cc == False:
            # Use ksp_hires
            self.ksp_binned = np.zeros((nBins, self.nCoils, nSpokesPerBin, nRO), dtype=np.complex128)
            self.coord_binned = np.zeros((nBins, nSpokesPerBin, nRO, 3))
            for i in range(nBins):
                spoke_idx = spokes_idx_per_bin[i][0:nSpokesPerBin]
                self.ksp_binned[i] = self.ksp_hires[:, spoke_idx]
                self.coord_binned[i] = self.coord_sampl_hires[spoke_idx]

        elif use_ksp_cc == True:
            # Use ksp_cc and virtualCoils
            self.ksp_binned = np.zeros((nBins, self.nVirtualCoils, nSpokesPerBin, nRO), dtype=np.complex128)
            self.coord_binned = np.zeros((nBins, nSpokesPerBin, nRO, 3))
            for i in range(nBins):
                spoke_idx = spokes_idx_per_bin[i][0:nSpokesPerBin]
                self.ksp_binned[i] = self.ksp_cc[:, spoke_idx]
                self.coord_binned[i] = self.coord_sampl_hires[spoke_idx]


    def bin_data_IR_recovery_time(self, nBins, use_ksp_cc=False, ndropSegs=0):
        '''drop segs in case it takes time for T1 prep curves to reach steady state
        '''

        self.nBins = nBins

        # setup indices for each bin
        nSpokesPerBinSingleSeg = np.uint32(self.spokes_per_seg // nBins)
        nSpokesPerBin = nSpokesPerBinSingleSeg * self.num_segs

        # Bin ksp and coords into bins and make sure same number of spokes in each bin
        nRO = self.ksp_hires.shape[-1]
        if use_ksp_cc == False:
            # Use ksp_hires
            self.ksp_binned = np.zeros((nBins, self.nCoils, nSpokesPerBin, nRO), dtype=np.complex128)
            self.coord_binned = np.zeros((nBins, nSpokesPerBin, nRO, 3))
            for j in range(self.num_segs - ndropSegs):
                for i in range(nBins):
                    start = (j+ndropSegs)*self.spokes_per_seg + i*nSpokesPerBinSingleSeg
                    end = (j+ndropSegs)*self.spokes_per_seg + (i+1)*nSpokesPerBinSingleSeg
                    self.ksp_binned[i, :, j*nSpokesPerBinSingleSeg:(j+1)*nSpokesPerBinSingleSeg] = self.ksp_hires[:,  start:end]
                    self.coord_binned[i, j*nSpokesPerBinSingleSeg:(j+1)*nSpokesPerBinSingleSeg] = self.coord_sampl_hires[start:end]

        elif use_ksp_cc == True:
            # Use ksp_cc and virtualCoils
            self.ksp_binned = np.zeros((nBins, self.nVirtualCoils, nSpokesPerBin, nRO), dtype=np.complex128)
            self.coord_binned = np.zeros((nBins, nSpokesPerBin, nRO, 3))

            for j in range(self.num_segs - ndropSegs):
                for i in range(nBins):
                    start = (j+ndropSegs)*self.spokes_per_seg + i*nSpokesPerBinSingleSeg
                    end = (j+ndropSegs)*self.spokes_per_seg + (i+1)*nSpokesPerBinSingleSeg
                    self.ksp_binned[i, :, j*nSpokesPerBinSingleSeg:(j+1)*nSpokesPerBinSingleSeg] = self.ksp_cc[:,  start:end]
                    self.coord_binned[i, j*nSpokesPerBinSingleSeg:(j+1)*nSpokesPerBinSingleSeg] = self.coord_sampl_hires[start:end]


    def reshape_data_IR_recovery_time(self, use_ksp_cc=False, ndropSegs=0, undersampl_frac=None):
        '''
        reshape spokes to have IR recovery time dimension (spokes_per_seg dimension)
        drop segs in case it takes time for T1 prep curves to reach steady state

        undersample_frac = determines what fraction of data to drop
        '''
        nRO = self.ksp_hires.shape[-1]
        if use_ksp_cc == False:
            ksp_to_use = self.ksp_hires
            numCoils = self.nCoils
        else:
            ksp_to_use = self.ksp_cc
            numCoils = self.nVirtualCoils

        # Initialize
        coord_ir = np.zeros((self.spokes_per_seg, self.num_segs-ndropSegs, nRO, 3))
        ksp_ir = np.zeros((self.spokes_per_seg, numCoils, self.num_segs-ndropSegs, nRO), dtype=np.complex128)

        # Sort
        for spk in range(self.spokes_per_seg):
            start = ndropSegs*self.spokes_per_seg + spk
            coord_ir[spk] = self.coord_sampl_hires[start::self.spokes_per_seg]
            ksp_ir[spk] = ksp_to_use[:, start::self.spokes_per_seg]

        # drop segments based on undersampl_frac
        if undersampl_frac is not None:
            segs_to_keep = np.uint32((1-undersampl_frac) * (self.num_segs-ndropSegs))
            coord_ir = coord_ir[:, 0:segs_to_keep]
            ksp_ir = ksp_ir[:, :, 0:segs_to_keep]

        self.coord_ir = coord_ir
        self.ksp_ir = ksp_ir

        return ksp_ir, coord_ir
    

    def bin_data_frames_consecutive(self, spokesPerFrame, use_ksp_cc=False):

        if use_ksp_cc:
            ksp = self.ksp_cc
            coils = self.nVirtualCoils
        else:
            ksp = self.ksp_hires
            coils = self.nCoils

        nRO = ksp.shape[-1]
        nFrames = np.uint32(np.floor(self.nSpokes / spokesPerFrame))

        ksp_frames = np.zeros((nFrames, coils, spokesPerFrame, nRO), dtype=np.complex128)
        coord_frames = np.zeros((nFrames, spokesPerFrame, nRO, 3))

        for i in range(nFrames):
            start = i*spokesPerFrame
            ksp_frames[i] = ksp[:, start:start+spokesPerFrame]
            coord_frames[i] = self.coord_sampl_hires[start:start+spokesPerFrame]

        self.coord_frames = coord_frames
        self.ksp_frames = ksp_frames

        return ksp_frames, coord_frames


    def visualize_cc_coils(self, nPtsLowres=64):
        '''
        20 coils with 64 pts on cpu takes 7min
        '''
    
        if hasattr(self, 'ksp_cc') == False:
            raise ValueError('Coil compression has not been run yet. Run coil_compress() first')
        
        im_size = np.uint32(self.coord_sampl_hires[:, 0:nPtsLowres, :].max())
        im_coils = recon_adjoint_postcomp_coilbycoil(ksp=self.ksp_cc[:, :, 0:nPtsLowres], 
                                                          coord=self.coord_sampl_hires[:, 0:nPtsLowres, :], 
                                                          img_shape=[im_size, im_size, im_size], device=-1)
        
        
        return im_coils


class Data_EndpointsFile_ZTE(Data_ZTE):
    '''Implementation diffschemes_radial3d on DV26
    '''

    def __init__(self, endpoints_txt_path, 
                 h5_save_dir, nPtsPerSpoke=None, rBW=31.25e3, 
                 FOV_scale=1, FOV_scale_dir=[1,1,1], ndrop_ROpts=1):

        super().__init__(h5_save_dir, nPtsPerSpoke, rBW, FOV_scale, FOV_scale_dir, ndrop_ROpts)

        self.endpoints_txt_path = endpoints_txt_path

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



class Data_CalcPhyllo_ZTE(Data_ZTE):
    '''For Stanford implementation of diffschemes, calc phyllo because not saving endpoints
    '''

    def __init__(self, smoothness, gm_flag, segsPerInterleaf, 
                 h5_save_dir, nPtsPerSpoke=None, rBW=31.25e3, 
                 FOV_scale=1, FOV_scale_dir=[1,1,1], ndrop_ROpts=1):
        
        super().__init__(h5_save_dir, nPtsPerSpoke, rBW, FOV_scale, FOV_scale_dir, ndrop_ROpts)

        self.smoothness = smoothness

        if gm_flag is None:
            gm_flag = True
        self.gm_flag = gm_flag

        if segsPerInterleaf is None:
            segsPerInterleaf = 1
        self.segsPerInterleaf = segsPerInterleaf

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



class Data_Product_ZTE(Data_ZTE):
    '''For product sequence 3dradial, MATLAB code calculates trajectory 
    '''
    def __init__(self, 
                 h5_save_dir, nPtsPerSpoke=None, rBW=31.25e3, 
                 FOV_scale=1, FOV_scale_dir=[1,1,1], ndrop_ROpts=1):
        
        super().__init__(h5_save_dir, nPtsPerSpoke, rBW, FOV_scale, FOV_scale_dir, ndrop_ROpts)

        self.get_coords()
        self.setup_hires_spokes()
        self.setup_waspi_spokes()

    def get_coords(self):
        # Read coords from matlab directory
        try:
            ktraj_file = loadmat(self.h5_save_dir + '/coord.mat')
            self.coord_full = (ktraj_file['K']).transpose(1,0,2)  # is NOT FOV_scaled
        except: 
            print("Coord.mat file not found in h5_save_dir")



class Data_Arc_ZTE(Data_ZTE):
    '''Implementation cont_slew_zte

    spokes_all - calculated from gradients using rotation matrices (WASPI scale included). Includes non-sampled points!
    coord_full - not FOV scaled. Removes non-sampled points at end of TR
    
    '''
    def __init__(self, kacq_file, seg_rot_file, 
                 acq_params, h5_save_dir, nPtsPerSpoke=None, 
                 rBW=31.25e3, FOV_scale=1, FOV_scale_dir=[1,1,1], 
                 ndrop_ROpts=1, points_grad_k0_delay=0, 
                 dt_sampling=8, grad_dt_sampling=4):

        
        super().__init__(h5_save_dir, nPtsPerSpoke, rBW, FOV_scale, FOV_scale_dir, ndrop_ROpts)

        # Save Arc-ZTE arguments
        self.seg_rot_file = seg_rot_file
        self.arc_angle = acq_params['arc_angle']
        self.points_per_spoke = np.uint32(acq_params['points_per_spoke'] ) 
        self.points_before_curve = np.uint32(acq_params['points_before_curve'])
        self.a_grad = acq_params['a_grad'] 
        
        self.dt_sampling = dt_sampling
        self.grad_dt_sampling = grad_dt_sampling
        # Delay between RF & grad (only needed for ArcZTE) in units of grad_dt_sampling
        # Can do positive or negative or float up to multiples of 0.25. Negative means grad is delayed
        self.points_grad_k0_delay = points_grad_k0_delay 

        # Get coordinates using kacq file
        self.grad_segment_file = kacq_file
        self.get_coords_using_kacq_file()
        # Set up trajectories - RUFIS and WASPI
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
                            self.spokes_per_seg, self.waspi_scale, self.opxres, self.grad_dt_sampling, 
                            self.points_grad_k0_delay)
        
        ## Calculate correct scaling for reconstructing correcting FOV
        self.coord_full = np.copy(self.spokes_all[:, 0:self.opxres]) # only sampling part of gradients

        acquired_res = self.opxres # TODO: check that this is giving the correct resolution
        scale_factor_fov = acquired_res * 1/(2*self.coord_full.max()) # digitize acquired resolution
        self.coord_full *= scale_factor_fov


    def setup_hires_spokes(self):
        '''
        Same as parent class except without setting up diameter spokes
        '''

        # Initialize sample shift
        self.sample_shift = 0 # no grad correction to begin with

        # Save only hires spokes
        self.ksp_hires = self.ksp_full[:, self.nInSpokes:, self.dead_time:self.nPtsPerSpoke]
        coord_sampl_hires = self.coord_full[self.nInSpokes:, 0:self.nPtsPerSpoke]

        # Scale coord_sampl_hires by FOV scale
        for i in range(3):
            coord_sampl_hires[..., i] *= self.FOV_scale_dir[i]
        coord_sampl_hires *= self.FOV_scale

        # Remove 0s from coord_sampl_hires
        self.coord_sampl_hires = coord_sampl_hires[:, self.dead_time:]
        # Keep 0s in coord_radial_spokes
        self.coord_radial_spokes = coord_sampl_hires


    def gradient_corr(self, sample_shift):

        '''
        Gradient delay correction applied to all saved coordinates except coord_full
        '''

        if self.sample_shift != 0: # only if new sample_shift
            print('Gradient delay correction of ' + str(self.sample_shift) + 
                  ' was already applied. Run setup_hires_spokes() and setup_waspi_spokes() again first.')
            return
            
        # Calculate whole and fractional shifts
        frac_shift = np.sign(sample_shift) * (abs(sample_shift) - np.floor(abs(sample_shift)))
        whole_shift = int(np.sign(sample_shift) * np.floor(abs(sample_shift)))

        for i in range(abs(whole_shift)):
            unit_shift = np.sign(sample_shift) * 1
            # applying gradient delay correction on high-res spokes
            self.coord_radial_spokes = shift_coord(unit_shift, self.coord_radial_spokes)
            self.coord_sampl_hires = shift_coord(unit_shift, self.coord_sampl_hires)

            # applying gradient delay correction on waspi spokes
            self.coord_waspi = shift_coord(unit_shift, self.coord_waspi)

        if abs(frac_shift) > 0:
            # applying gradient delay correction on high-res spokes
            self.coord_radial_spokes = shift_coord(frac_shift, self.coord_radial_spokes)
            self.coord_sampl_hires = shift_coord(frac_shift, self.coord_sampl_hires)

            # applying gradient delay correction on waspi spokes
            self.coord_waspi = shift_coord(frac_shift, self.coord_waspi)

        # Save that shift was done
        self.sample_shift = sample_shift
        print('Applied gradient delay correction of ' + str(self.sample_shift))           
    


   