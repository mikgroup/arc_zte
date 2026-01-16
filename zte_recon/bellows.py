from scipy import interpolate
import numpy as np

def read_bellows_data(bellows_path):
    '''
    Read bellows text file into numpy array given path to RESPData file
    '''
    
    bellows_file = open(bellows_path, "r")
    content = bellows_file.readline()
    waveform = []

    while True:
        content = bellows_file.readline()
        if not content:
            break
        waveform.append(int(content))

    bellows_file.close()
    waveform = np.array(waveform)
    
    return waveform

def resample_bellows_data(waveform, num_segs, num_segs_lowres, tr,
                      spokes_per_seg=384, bellows_sampl_rate=0.04, debug=True):
    '''
    Resample bellows data onto each TR of sequence to bin each TR into bin

    tr in seconds
    bellows_sampl_rate is sampling period in s. Typically 40ms for GE

    Returns:
        bellows_interp : bellows interpolated to each TR of entire sequence (includes WASPI)
        cropped_waveform : bellows waveform cropped to length of entire sequence (includes WASPI)
        times_sequence : times in s for bellows_interp 
        times_bellows : times in s for cropped_waveform

    '''
    
    # Set up sampling times for one segment
    # additional sample for beginning of rampup and beginning of rampdown
    segment_sampl_times = np.zeros((spokes_per_seg + 2)) 
    segment_sampl_times[0] = 0 # beginning of segment at t=0
    segment_sampl_times[1] = 0.01 # rampup is fixed at 10ms

    for i in np.arange(2, spokes_per_seg+1): # goes until spokes_per_seg-th TR
        segment_sampl_times[i] = segment_sampl_times[i-1] + tr
    segment_sampl_times[-1] = segment_sampl_times[-2] + tr # last sampl time is before rampdown

    # Figure out how much of bellows waveform to chop off
    segment_duration = segment_sampl_times[-1] + 0.01 # adding 10ms rampdown to get full segment duration
    sequence_duration = segment_duration * (num_segs + num_segs_lowres) # in sec
    sequence_num_samples = int(sequence_duration // bellows_sampl_rate) + 1
    cropped_waveform = waveform[-1*sequence_num_samples:]
    
    # Generate sequence timing array
    times_sequence = np.copy(segment_sampl_times[1:-1])  # first segment
    last_time = segment_duration # end time of the segment
    
    for i in range(num_segs+num_segs_lowres-1):
        times_sequence = np.concatenate((times_sequence, segment_sampl_times[1:-1]+last_time))
        last_time += segment_duration
    
    # Generate cropped waveform timing array
    times_bellows = np.linspace(0, (len(cropped_waveform)-1)*bellows_sampl_rate, len(cropped_waveform))
    
    # Interpolate cropped waveform onto sequence timing array
    # Shift times bellows to match end time of sequence. Should not affect binning if shift is small
    cropped_waveform_new = np.copy(cropped_waveform)
    times_bellows_new = np.copy(times_bellows)
    while times_sequence[-1] > times_bellows_new[-1]:
        cropped_waveform_new = np.concatenate((cropped_waveform_new, np.array([cropped_waveform[-1]])))
        times_bellows_new = np.concatenate((times_bellows_new, np.array([times_bellows[-1] + bellows_sampl_rate])))

    f = interpolate.interp1d(times_bellows_new, cropped_waveform_new)
    bellows_interp = f(times_sequence)

    if debug:
        return bellows_interp, cropped_waveform, times_sequence, times_bellows
    else:
        return bellows_interp


def create_bins(bellows_interp, nBins, nWaspi, margin_discard=5):
    '''
    Create bins from interpolated bellows, only RUFIS spokes - waspi not considered for binning
    May be +/- 1 spoke in each bin (not exactly equal, but off by 1 spoke max)
    Returns:
        bins : boundaries of each bin, created based on evenly spaced percentiles of the data
        spoke_idx_per_bin : spoke indices within each bin, where spoke #0 = RUFIS spoke 0

    '''
    # Remove waspi spokes from resp_wave
    resp_wave_rufis = bellows_interp[nWaspi:]
    
    bins = np.percentile(resp_wave_rufis, np.linspace(0 + margin_discard, 100 - margin_discard, nBins + 1))

    spokes_per_bin_tmp = []
    spoke_idx_per_bin_tmp = []
    # Bin non-waspi spokes.
    for b in range(nBins):
        # Resp wave doesnt have waspi spokes
        idx = (resp_wave_rufis >= bins[b]) & (resp_wave_rufis < bins[b + 1])
        spoke_idx = np.where(idx==True)[0]
        spoke_idx_per_bin_tmp.append(spoke_idx)
        spokes_per_bin_tmp.append(len(spoke_idx))
        
    return bins, spoke_idx_per_bin_tmp


def soft_gating_weight(resp, percentile=30, alpha=1, flip=False):
    '''
    From Frank extreme MRI code. If percentile increases, more samples are accepted at wt 1
    '''
    sigma = 1.4628 * np.median(np.abs(resp - np.median(resp)))
    resp = (resp - np.median(resp)) / sigma
    thresh = np.percentile(resp, percentile)
    if flip:
        resp *= -1
    return np.exp(-alpha * np.maximum((resp - thresh), 0))