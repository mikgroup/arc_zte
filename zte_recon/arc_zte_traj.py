import numpy as np
from scipy.integrate import cumulative_trapezoid

gamma_bar = 4.257588 # KHz/ G

def corr_arc_coords(coords_in):
    '''
    Transpose x and y coordinates
    Dims [nSpokes, nRO, 3]
    '''
    coords_out = np.copy(coords_in)
    coords_out[..., 0] = coords_in[..., 1]
    coords_out[..., 1] = coords_in[..., 0]
    
    return coords_out


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



def calc_all_curved_grads(spoke_rot_file, seg_rot_file, arc_angle, 
                          points_per_spoke, points_before_curve, 
                          dt_sampling, a_grad, num_segs, num_segs_lowres, 
                          spokes_per_seg, waspi_scale):
    '''
    a_grad in G/cm
    dt_sampling in us

    '''

    arc_angle_rad = arc_angle * np.pi/180.0
    nSegs_total = num_segs + num_segs_lowres

    ## Read spoke rotations from file
    R_file = np.zeros((spokes_per_seg, 9), dtype=float)
    count = 0

    with open(spoke_rot_file, "r") as file1:
        
        for line in file1.readlines():
            line_list = line.split(" ")
            if (count >= spokes_per_seg):
                break
            for i in range(9):
                R_file[count, i] = float(line_list[i])
                
            count += 1

    ## Read segment rotations from file
    M_file = np.zeros((nSegs_total, 9), dtype=float)
    count = 0

    with open(seg_rot_file, "r") as file1:
        
        for line in file1.readlines():
            line_list = line.split(" ")
            if (count >= nSegs_total):
                break
            for i in range(9):
                M_file[count, i] = float(line_list[i])
                
            count += 1

    ## Using rotation matrices from file, find gradients for each spoke in physical units G/cm
    # then use cumsum to find spoke_arr in physical units of cm^-1
    spoke_arr = np.zeros((spokes_per_seg, 3, points_per_spoke - points_before_curve)) # one segment of spokes
    grad_cat = np.zeros((spokes_per_seg, 3, points_per_spoke))

    delta_k = gamma_bar * a_grad * dt_sampling * 0.001 # convert us to ms
    delta_angle_rad = arc_angle_rad / (points_per_spoke)
    r = np.sqrt((delta_k**2)/(2 - 2*np.cos(delta_angle_rad)))

    first_spoke = single_spoke_arc(points_per_spoke+1, angle=arc_angle, r=r)
    first_grad = np.diff(first_spoke)/(dt_sampling*0.001*gamma_bar)

    grad_cat[0] = first_grad
    spoke_arr[0] = np.cumsum(grad_cat[0, :, points_before_curve:], axis=-1) * gamma_bar * dt_sampling * 0.001

    ## Rotate the gradient by spokeRots to find one segment
    for i in range(spokes_per_seg-1):
        # rotate gradients
        grad_cat[i+1] = R_file[i].reshape(3,3) @ first_grad
        first_grad = grad_cat[i+1]
        
        # solve for spoke
        spoke_arr[i+1] = np.cumsum(grad_cat[i+1, :, points_before_curve:], axis=-1) * gamma_bar * dt_sampling * 0.001
        
    # reformat grad_cat to have continuous waveforms
    grad_cat_flat = grad_cat.transpose(1,0,2)
    grad_cat_flat = grad_cat_flat.reshape(3, spokes_per_seg*points_per_spoke)

    ## Rotate the gradients by segRots, then find spoke_arr
    spokes_full_arc = np.zeros((nSegs_total*spokes_per_seg, 3, points_per_spoke - points_before_curve))
    grads_full_arc = np.zeros((nSegs_total*spokes_per_seg, 3, points_per_spoke))

    for i in range(nSegs_total):

        if i < num_segs_lowres:
            scale = (1/waspi_scale)
        else:
            scale = 1
        
        if i == 0: # don't rotate first segment
            for j in range(spokes_per_seg):
                spokes_full_arc[i*spokes_per_seg + j] = scale * spoke_arr[j]
                grads_full_arc[i*spokes_per_seg + j] = scale * grad_cat[j]
        
        else:
            for j in range(spokes_per_seg):
                grads_full_arc[i*spokes_per_seg + j] = scale * (M_file[i-1].reshape(3,3) @ grad_cat[j])
                spokes_full_arc[i*spokes_per_seg + j] = np.cumsum(grads_full_arc[i*spokes_per_seg + j, :, 
                                                        points_before_curve:], axis=-1) * gamma_bar * dt_sampling * 0.001
                

    ## Saved attributes include all spokes/gradient waveforms (not just sampled parts)
    ## Scaled by waspi factor
    spokes_all = spokes_full_arc.transpose(0,2,1) # nSpokesTotal, nPtsPerSpoke, 3
    grads_all = grads_full_arc.transpose(0,2,1) # nSpokesTotal, nPtsPerSpoke, 3
    grads_flat_one_seg = grad_cat_flat  # 3, nPts_per_segment

    return spokes_all, grads_all, grads_flat_one_seg


def get_segment_waveform_from_kacq_file(txt_file):
    '''Read waveforms from debug txt file where grads start with line '---' and string header row like "Gx Gy Gz Gnrom"
    '''
    f = open(txt_file, "r")
    data = f.read()

    # Split data into rows and then into columns
    rows = data.split('\n')
    cols = [row.split('\t') for row in rows]
    
    # find rows with ---
    bounds = []
    for i, row in enumerate(rows):
        if row == '---':
            bounds.append(i)

    assert len(bounds) == 1, "Incorrect number of txt file boundaries. Found " + str(len(bounds) + " lines that are '---' ")

    # Get gradients
    gx = [float(data_pt[0]) for data_pt in cols[bounds[0]+2:-2]]
    gx = np.array(gx)
    
    # Get gradients
    gy = [float(data_pt[1]) for data_pt in cols[bounds[0]+2:-2]]
    gy = np.array(gy)
    
    # Get gradients
    gz = [float(data_pt[2]) for data_pt in cols[bounds[0]+2:-2]]
    gz = np.array(gz)

    return gx, gy, gz


def rotate_integrate_all_segments(grad_first_seg, seg_rot_file, 
                            points_per_spoke, points_before_curve, 
                            dt_sampling, num_segs, num_segs_lowres, 
                            spokes_per_seg, waspi_scale, opxres, grad_dt_sampling=4, 
                            points_grad_k0_delay=0):
    '''
    grad_one_seg: [3, points_per_spoke * spokes_per_segment]
    dt_sampling in us

    Handles all segments (WASPI + RUFIS). Scales WASPI by 1/waspi_scale
    '''
    num_segs_total = num_segs + num_segs_lowres

    ## Read segment rotations from file
    M_file = np.zeros((num_segs_total, 9), dtype=float)
    count = 0

    with open(seg_rot_file, "r") as file1:
        
        for line in file1.readlines():
            line_list = line.split(" ")
            if (count >= num_segs_total):
                break
            for i in range(9):
                M_file[count, i] = float(line_list[i])
            count += 1

    ## Rotate the gradients by segRots, then find spoke_arr
    spokes_full_arc = np.zeros((num_segs_total*spokes_per_seg, 3, opxres))

    for i in range(num_segs_total):
        # scale factor for WASPI scale for first num_segs_lowres segments
        if i < num_segs_lowres:
            scale = (1/waspi_scale)
        else:
            scale = 1
        
        if i == 0:
            # don't rotate for first segment
            grad_ith_seg = scale * grad_first_seg
        else:
            # rotate all other segments. File does not have identity as first line. 
            # In EPIC, first line of M_file is read in segment after identity applied
            grad_ith_seg = scale * M_file[i-1].reshape(3,3) @ grad_first_seg

        # save gradients 
        spoke_arr = integrate_grads_one_segment(grad_ith_seg, spokes_per_seg, opxres, 
                                                points_per_spoke, points_before_curve, dt_sampling, 
                                                grad_dt_sampling, points_grad_k0_delay)
        spokes_full_arc[spokes_per_seg*i : spokes_per_seg*(i+1)] = spoke_arr

    return spokes_full_arc.transpose(0,2,1) # nSpokesTotal, nPtsPerSpoke, 3



def integrate_grads_one_segment(grad_one_seg, spokes_per_seg, opxres, points_per_spoke, 
                                points_before_curve, dt_sampling, grad_dt_sampling, points_grad_k0_delay=0):
    '''
    Integrate grads for a single segment to get kspace coords.  

    dt_sampling in us. Must be multiple of grad_dt_sampling (4us). DAQ sampling period based on dwell time
    grad_dt_sampling in us. Gradient sampling period minimum is 4us. 
    spoke_arr: [3,]

    grad_k0_delay: is delay in ***grad_dt_sampling*** units to start integrating. e.g. -1 will be a delay of 4us
    Can be positive or negative integer or float multiple of 0.25. 
    If positive, RF is late. If negative, gradients are late (start integrating earlier than expect). 

    Both points_per_spoke and points_before_curve do not include last point of interval
    '''

    ## Find kspace spokes for 1 segment
    spoke_arr = np.zeros((spokes_per_seg, 3, opxres)) # one segment of spokes

    # Check that integration won't go beyond bounds of gradient waveform for one segment
    # i.e. shift if more negative than points_before_curve or more positive than points after DAQ
    if points_grad_k0_delay < (-1*points_before_curve) or points_grad_k0_delay > (points_per_spoke - points_before_curve - opxres):
        raise NotImplementedError('Shift is too large for current implementation of grad delay compensation')
    
    # check dt_sampling is multiple of grad_dt_sampling
    assert dt_sampling % grad_dt_sampling == 0, "dt_sampling needs to be a multiple of gradient sampling time"
    m = dt_sampling // grad_dt_sampling 

    ## Old code
        # if m == 1:

        #     for k in range(spokes_per_seg):

        #         # Integrate spoke starting at points_before_curve. Add 0 to be first element for k=0
        #         intg_start = (k*points_per_spoke)+points_before_curve + points_grad_k0_delay
        #         intg_end = intg_start + opxres
        #         spoke_arr[k] = cumulative_trapezoid(grad_one_seg[:, intg_start:intg_end], 
        #                                                 dx=gamma_bar * dt_sampling * 0.001, axis=-1, 
        #                                                 initial=0) * gamma_bar * dt_sampling * 0.001
        #         # spoke_arr[k] = np.cumsum(grad_one_seg[:, intg_start:intg_end], axis=-1) * gamma_bar * dt_sampling * 0.001
        
        #######################
        ## Integrate at m times finer, then choose every mth point to get kspace coords. Reduces to above code if m == 1

        # ## Find kspace spokes for 1 segment sampled at grad_dt_sampling
        # spoke_arr_for_grad_sampling = np.zeros((spokes_per_seg, 3, (opxres-1)*m + 1)) # one segment of spokes

        # for k in range(spokes_per_seg):

        #     # Integrate spoke starting at points_before_curve. Add 0 to be first element for k=0
        #     intg_start = (k*points_per_spoke) + points_before_curve + points_grad_k0_delay
        #     intg_end = intg_start + (opxres-1)*m + 1
        #     spoke_arr_for_grad_sampling[k] = cumulative_trapezoid(grad_one_seg[:, intg_start:intg_end], 
        #                                             dx=gamma_bar * grad_dt_sampling * 0.001, axis=-1, 
        #                                             initial=0) * gamma_bar * grad_dt_sampling * 0.001

        # # Pick out integral values every mth value to get final spoke arr
        # if m > 1:
        #     spoke_arr = spoke_arr_for_grad_sampling[:, :, 0::m]
        # else:
        #     spoke_arr = spoke_arr_for_grad_sampling


    ### If points_grad_k0_delay is integer, then can use directly (no changes to calculation)
    if points_grad_k0_delay % 1 == 0:
        interp_factor = 1
        grads_for_integ = grad_one_seg # no need to interpolate gradients

    ### Interpolate to 4x of current gradient sampling. Then integrate with offset in units
    elif points_grad_k0_delay % 0.25 == 0:
        interp_factor = 4

        # New positions are 4x finer
        t_idx = np.arange(0, len(grad_one_seg[0]), 1)
        t_idx_interp = np.linspace(0, len(grad_one_seg[0])-1, interp_factor*(len(grad_one_seg[0])-1) + 1)

        assert (points_grad_k0_delay*interp_factor) % 1 == 0

        # Linearly interpolate gradients to 4x finer
        grads_interp = np.zeros((3, len(t_idx_interp)))
        for axis in range(3):
            grads_interp[axis] = np.interp(t_idx_interp, t_idx, grad_one_seg[axis])

        grads_for_integ = grads_interp
    
    else:
        raise NotImplementedError('Shift is too fine for current implementation. Need multiples of 1/4 of gradient sampling')
    
    ### Do integration
    ## Find kspace spokes for 1 segment sampled at grad_dt_sampling
    spoke_arr_for_grad_sampling = np.zeros((spokes_per_seg, 3, (opxres-1)*m*interp_factor + 1)) # one segment of spokes

    for k in range(spokes_per_seg):

        # Integrate spoke starting at points_before_curve. Add 0 to be first element for k=0
        intg_start = np.uint32( ((k*points_per_spoke)+points_before_curve + points_grad_k0_delay) * interp_factor)
        intg_end = intg_start + (opxres-1)*m*interp_factor + 1 
        try: 
            # Initial integration value is 0. Output has same shape as grad_one_seg[:, intg_start:intg_end],
            spoke_arr_for_grad_sampling[k] = cumulative_trapezoid(grads_for_integ[:, intg_start:intg_end], 
                                                    dx=gamma_bar * grad_dt_sampling * 0.001, axis=-1, 
                                                    initial=0)
            # spoke_arr_for_grad_sampling[k, :, 1:] = np.cumsum(grads_for_integ[:, intg_start:intg_end-1], axis=-1) * gamma_bar * grad_dt_sampling * 0.001
        except:
            import pdb; pdb.set_trace()

    # Pick out integral values every mth value to get final spoke arr
    if m > 1:
        spoke_arr = spoke_arr_for_grad_sampling[:, :, 0::(m*interp_factor)]
    else:
        spoke_arr = spoke_arr_for_grad_sampling

    return spoke_arr


def read_params_from_kacq(txt_file, kacq_version=None):
    '''Function to read parameters from kacq file
    '''
    if txt_file == "dummy_kacq":
        raise ValueError('Cannot read Arc-ZTE params from dummy_kacq')

    f = open(txt_file, "r")
    data = f.read()

    param_end_idx = 0
    # Split data into rows and then into columns
    rows = data.split('\n')
    for i, row in enumerate(rows):
        if row == '---':
            param_end_idx = i

    # if kacq version not set, check the first word of the first row
    # version 1 has "CV <name> <val>" and version 2 has "<name>\t<val>"
    if kacq_version is None:
        col_0_row_0 = rows[0][0:2]
        if col_0_row_0 == 'CV':
            kacq_version = 1
        else:
            kacq_version = 2

    # create dictionary 
    params = {}
    float_params = ['a_grad', 'arc_angle', 'opslthick', 'opflip'] # values are floats, rest are ints

    if kacq_version == 2:
        # Store values in dict
        for row in rows[0:param_end_idx]:
            # split row
            cols = row.split('\t')

            param_name = cols[0]
            param_val = cols[1]

            if param_name in float_params:
                params[param_name] = float(param_val)
            else:
                params[param_name] = int(param_val)

    elif kacq_version == 1:
        # Store values in dict
        for row in rows[0:param_end_idx]:
            # split row
            cols = row.split(' ')

            param_name = cols[1]
            param_val = cols[2]

            if param_name in float_params:
                params[param_name] = float(param_val)
            else:
                params[param_name] = int(param_val)


    return params

        