import numpy as np
from tqdm import tqdm
import bart
from scipy.optimize import curve_fit

def Mz_IR_readout(times, flip_angle=2, TR=2.5, T1=900, tprep=450, Mz_start=1):
    '''All times in ms
    Mz_start is value at inversion before tprep happens (not at beginning of readout)
    '''
    seg_time = times - tprep
    
    flip_angle_rad = flip_angle * np.pi / 180
    beta = np.exp(-1*seg_time/T1) * np.cos(flip_angle_rad)
    
    # Effect of Mz_start having T1 recovery for tprep amount of time
    M_prep = Mz_start*(np.exp(-1*tprep/T1)) + 1*(1-np.exp(-1*tprep/T1)) # recovery at end of prep time
    
    E1 = np.exp(-1*TR/T1)
    M_spgr = (1 - E1) / (1 - E1*np.cos(flip_angle_rad))
    
    return M_prep * (beta) + M_spgr * (1-beta) # prep is decaying, steady state is recovering



def Mz_IR_readout_noprep(times, M_prep, flip_angle=2, TR=2.5, T1=900):
    '''All times in ms
    Mz_prep is value at beginning of readout
    '''
    seg_time = times
    
    flip_angle_rad = flip_angle * np.pi / 180
    beta = np.exp(-1*seg_time/T1) * np.cos(flip_angle_rad)
    
    E1 = np.exp(-1*TR/T1)
    M_spgr = (1 - E1) / (1 - E1*np.cos(flip_angle_rad))

    return M_prep * (beta) + M_spgr * (1-beta) # prep is decaying, steady state is recovering



def Mz_IR_prep(ir_times, T1=900, tprep=None, Mz_start=1):
    '''All times in ms
    '''    
    E1 = np.exp(-1*ir_times/T1)
    M_prep = Mz_start*(E1) + 1*(1-E1) # assumes complete recovery between segments

    
    return M_prep

def Mz_Sat_readout(times, M_prep, flip_angle=2, TR=2.5, T1=900):
    '''All times in ms
    Mprep is value at beginning of readout
    '''
    seg_time = times
    
    flip_angle_rad = flip_angle * np.pi / 180
    beta = np.exp(-1*seg_time/T1) * np.cos(flip_angle_rad)
       
    E1 = np.exp(-1*TR/T1)
    M_spgr = (1 - E1) / (1 - E1*np.cos(flip_angle_rad))
    
    return M_prep * (beta) + M_spgr * (1-beta) # prep is decaying, steady state is recovering


def Mz_Sat_prep(times, T1=900):
    '''All times in ms
    '''   
    E1 = np.exp(-1*times/T1)
    M_prep = (1 - E1)

    return M_prep


def sim_ss_recovery_irzte(T1, tprep, trecovery, flip_angle, tr, spokes_per_seg=512, num_segs=20):
    
    prep_times = np.arange(0, tprep, tr) # before readout
    readout_times = tprep + (np.arange(0, spokes_per_seg, 1)*tr)
    recovery_times = np.arange(0, trecovery, tr)

    Mz_start = -1
    
    Mz_cum = np.array([])
    for i in range(num_segs):
        Mz_prep = Mz_IR_prep(prep_times, T1, tprep=tprep, Mz_start=Mz_start)
        Mz_cum = np.concatenate((Mz_cum, Mz_prep))
        Mz_acq = Mz_IR_readout(readout_times, flip_angle, tr, T1, tprep=tprep, Mz_start=Mz_start)
        Mz_cum = np.concatenate((Mz_cum, Mz_acq))
        
        Mz_start_recovery = Mz_acq[-1]
        Mz_recovery = Mz_IR_prep(recovery_times, T1, tprep=tprep, Mz_start=Mz_start_recovery)
        Mz_cum = np.concatenate((Mz_cum, Mz_recovery))
        
        Mz_start = -1*Mz_recovery[-1]
        
    return Mz_cum, Mz_acq


def sim_ss_recovery_fatsatprep_zte(isFat, T1, tprep, trecovery, flip_angle, tr, 
                                   spokes_per_seg=512, num_segs=20, fatsat_flip=180):
    
    prep_times = np.arange(0, tprep, tr) # before readout
    readout_times = np.arange(0, spokes_per_seg, 1)*tr
    recovery_times = np.arange(0, trecovery, tr)

    
    Mz_cum = np.array([])

    if isFat and fatsat_flip==90:
        for i in range(num_segs):
            Mz_prep = Mz_Sat_prep(prep_times, T1) # saturation starts it from Mz=0
            Mz_cum = np.concatenate((Mz_cum, Mz_prep))

            Mz_prep = 0 # Fat gets saturated again
            Mz_acq = Mz_Sat_readout(readout_times, Mz_prep, flip_angle, tr, T1)
            Mz_cum = np.concatenate((Mz_cum, Mz_acq))
            
            Mz_start_recovery = Mz_acq[-1]
            Mz_recovery = Mz_IR_prep(recovery_times, T1, trecovery, Mz_start=Mz_start_recovery)
            Mz_cum = np.concatenate((Mz_cum, Mz_recovery))
    
    elif isFat and fatsat_flip > 90:

        for i in range(num_segs):
            Mz_prep = Mz_Sat_prep(prep_times, T1) # saturation starts it from Mz=0
            Mz_cum = np.concatenate((Mz_cum, Mz_prep))

            Mz_prep = Mz_prep[-1] * -1 * (np.cos((180 - fatsat_flip) * (np.pi/180))) # Mz new = Mz_old * cos(180-flip) * -1
            Mz_acq = Mz_IR_readout_noprep(readout_times, Mz_prep, flip_angle, tr, T1)
            Mz_cum = np.concatenate((Mz_cum, Mz_acq))
            
            Mz_start_recovery = Mz_acq[-1]
            Mz_recovery = Mz_IR_prep(recovery_times, T1, trecovery, Mz_start=Mz_start_recovery)
            Mz_cum = np.concatenate((Mz_cum, Mz_recovery))


    elif not isFat:
        for i in range(num_segs):
            Mz_prep = Mz_Sat_prep(prep_times, T1) # saturation starts it from Mz=0
            Mz_cum = np.concatenate((Mz_cum, Mz_prep))

            Mz_prep = Mz_prep[-1] # Other tissue continues recovering
            Mz_acq = Mz_Sat_readout(readout_times, Mz_prep, flip_angle, tr, T1)
            Mz_cum = np.concatenate((Mz_cum, Mz_acq))
            
            Mz_start_recovery = Mz_acq[-1]
            Mz_recovery = Mz_IR_prep(recovery_times, T1, trecovery, Mz_start=Mz_start_recovery)
            Mz_cum = np.concatenate((Mz_cum, Mz_recovery))
            
        
    return Mz_cum, Mz_acq


def sim_ss_recovery_double_sat_zte(T1, tprep1, tprep2, trecovery, flip_angle, tr, 
                                   spokes_per_seg=256, num_segs=20):
    

    prep_times1 = np.arange(0, tprep1-tprep2, tr) # before readout
    prep_times2 = np.arange(0, tprep2, tr) # before readout
    readout_times = (np.arange(0, spokes_per_seg, 1)*tr)
    recovery_times = np.arange(0, trecovery, tr)

    Mz_start = -1
    
    Mz_cum = np.array([])
    for i in range(num_segs):
        Mz_prep1 = Mz_Sat_prep(prep_times1, T1)
        Mz_cum = np.concatenate((Mz_cum, Mz_prep1))

        # Second inversion
        Mz_prep2 = Mz_Sat_prep(prep_times2, T1)
        Mz_cum = np.concatenate((Mz_cum, Mz_prep2))

        Mz_acq = Mz_IR_readout_noprep(readout_times, Mz_prep2[-1], flip_angle, tr, T1)
        Mz_cum = np.concatenate((Mz_cum, Mz_acq))
        
        Mz_start_recovery = Mz_acq[-1]
        Mz_recovery = Mz_IR_prep(recovery_times, T1, Mz_start=Mz_start_recovery)
        Mz_cum = np.concatenate((Mz_cum, Mz_recovery))
        
        Mz_start = -1*Mz_recovery[-1] # for next segment
        
    return Mz_cum, Mz_acq


def sim_ss_recovery_double_ir_zte(T1, tprep1, tprep2, trecovery, flip_angle, tr, 
                                   spokes_per_seg=256, num_segs=20):
    

    prep_times1 = np.arange(0, tprep1-tprep2, tr) # before readout
    prep_times2 = np.arange(0, tprep2, tr) # before readout
    readout_times = (np.arange(0, spokes_per_seg, 1)*tr)
    recovery_times = np.arange(0, trecovery, tr)

    Mz_start = -1
    
    Mz_cum = np.array([])
    for i in range(num_segs):
        Mz_prep1 = Mz_IR_prep(prep_times1, T1, Mz_start=Mz_start)
        Mz_cum = np.concatenate((Mz_cum, Mz_prep1))

        # Second inversion
        Mz_prep2 = Mz_IR_prep(prep_times2, T1, Mz_start=-1*Mz_prep1[-1])
        Mz_cum = np.concatenate((Mz_cum, Mz_prep2))

        Mz_acq = Mz_IR_readout_noprep(readout_times, Mz_prep2[-1], flip_angle, tr, T1)
        Mz_cum = np.concatenate((Mz_cum, Mz_acq))
        
        Mz_start_recovery = Mz_acq[-1]
        Mz_recovery = Mz_IR_prep(recovery_times, T1, Mz_start=Mz_start_recovery)
        Mz_cum = np.concatenate((Mz_cum, Mz_recovery))
        
        Mz_start = -1*Mz_recovery[-1] # for next segment
        
    return Mz_cum, Mz_acq


T1_LIST_SIM = np.linspace(10, 8000, 800)
T1_LIST_FAT = np.linspace(200, 450, 100)

def compute_temporal_basis_phi(type, spokes_per_seg, tr, ti, trecovery, numComps=3, flip_angle=3, t1_list=T1_LIST_SIM, t1_list_fat=T1_LIST_FAT):
    '''
    Compute for types 'sat_prep' or 'ir_prep' or 'sat_prep_with_fat'
    Returns phi [spokes_per_seg, numComps]

    '''
    tprep = ti
    trecovery = trecovery
    flip_angle = 3
    num_segs_sim = 30 # to ensure steady state is reached

    num_t1s = len(t1_list)
    Mz_over_t1s = np.zeros((num_t1s, spokes_per_seg))
    Mz_cum_over_t1s = []

    # Simulate Mz curves for each T1
    print('Simulating Mz curves for each T1...')
    if type == 'ir_prep':
        for i, t1 in tqdm(enumerate(t1_list), total=num_t1s):
            Mz_cum_t1, Mz_over_t1s[i] = sim_ss_recovery_irzte(t1, tprep, trecovery, flip_angle, tr, 
                                                              spokes_per_seg, num_segs=num_segs_sim)
            Mz_cum_over_t1s.append(Mz_cum_t1)

    elif type == 'sat_prep' or type == 'sat_prep_with_fat':
        for i, t1 in tqdm(enumerate(t1_list), total=num_t1s):
            isFat = False
            Mz_cum_t1, Mz_over_t1s[i] = sim_ss_recovery_fatsatprep_zte(isFat, t1, tprep, trecovery, flip_angle, tr, 
                                                                       spokes_per_seg, num_segs=num_segs_sim, fatsat_flip=180)
            Mz_cum_over_t1s.append(Mz_cum_t1)

        # Additionally simulate fat T1s if specified
        if type == 'sat_prep_with_fat':

            print('Simulating Mz curves for range of fat T1s...')
            num_t1s_fat = len(t1_list_fat)
            Mz_over_t1s_fat = np.zeros((num_t1s_fat, spokes_per_seg))

            for i, t1 in tqdm(enumerate(t1_list_fat), total=num_t1s_fat):
                isFat = True
                Mz_cum_t1, Mz_over_t1s_fat[i] = sim_ss_recovery_fatsatprep_zte(isFat, t1, tprep, trecovery, flip_angle, tr, 
                                                                               spokes_per_seg, num_segs=num_segs_sim, fatsat_flip=180)
                Mz_cum_over_t1s.append(Mz_cum_t1)
            Mz_over_t1s = np.vstack((Mz_over_t1s, Mz_over_t1s_fat))
    else:
        raise ValueError("Type must be 'ir_prep' or 'sat_prep'")
    
    # Run SVD using BART to get temporal basis
    print('Runing SVD to get temporal basis...')
    u, s, vh = bart.bart(3, 'svd -e', Mz_over_t1s.T)
    # Extract desired number of orthonormal columns from U
    extract_cmd = 'extract 1 0 ' + str(numComps)
    basis = bart.bart(1, extract_cmd, u)

    # transpose the basis to have time in the 5th dimension 
    # and coefficients in the 6th dimension
    basis = bart.bart(1, 'transpose 1 6', basis)
    phi = np.real(bart.bart(1, 'transpose 0 5', basis))

    return np.squeeze(phi), s, Mz_over_t1s


### Fit T1* from reconstructed 
def exp_recovery_T1star(t, A, B, T1star):
    """Apparent recovery model with T1*."""
    return A - B * np.exp(-t / T1star) # three parameter model


def fit_T1star_map(t, S_im):
    """
    Fit apparent T1* exponential recovery.
    S_im shape: (num_times, x,y,z)
    
    Returns:
        popt: fitted [A, B, T1*]
        perr: 1-sigma uncertainties
    """
    t = np.asarray(t, dtype=float)

    T1_star_map = np.zeros(S_im.shape[1:])  # x,y,z
    T1_map = np.zeros(S_im.shape[1:])  # x,y,z

    for x in range(S_im.shape[1]):
        for y in range(S_im.shape[2]):
            for z in range(S_im.shape[3]):

                S = S_im[:, x, y, z]
                if S.max() > 0:

                    p0 = None
                    if p0 is None:
                        A0 = S.max()
                        B0 = A0 - S.min()
                        T10 = (t.max() - t.min())/2
                        p0 = (A0, B0, T10)

                    popt, pcov = curve_fit(exp_recovery_T1star, t, S, p0=p0, maxfev=10000)

                    T1_star_map[x, y, z] = popt[2]
                    T1_map[x, y, z] = popt[2] * ((popt[1] / popt[0]) - 1) # correct to T1

    return T1_star_map, T1_map


def fit_T1star(t, S):
    """
    Fit apparent T1* exponential recovery.
    S_im shape: (num_times, x,y,z)
    
    Returns:
        popt: fitted [A, B, T1*]
        perr: 1-sigma uncertainties
    """
    t = np.asarray(t, dtype=float)

    p0 = None
    if p0 is None:
        A0 = S.max()
        B0 = A0 - S.min()
        T10 = (t.max() - t.min())/2
        p0 = (A0, B0, T10)

    popt, pcov = curve_fit(exp_recovery_T1star, t, S, p0=p0, maxfev=10000)

    return popt