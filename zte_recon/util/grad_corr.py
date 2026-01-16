import numpy as np

def shift_coord(sample_shift_frac, coord):
    '''
    To account for delay between grad and DAQ
    
    Only use for sample_shift < 1
    coord dim = [spokes, readout, 3]
    '''

    # positive shift to increase recon coords (acquisition was late)
    if sample_shift_frac > 0:
        coord_shift = np.copy(coord)
        for axes in range(3):
            diff_corr = np.zeros(coord[:, :, 0].shape)
            diff = np.diff(coord[:, :, axes], axis=-1) # take diff between each sample along readout dim
            diff_corr[:, :-1] = diff
            diff_corr[:, -1] = diff[:, -1] # shift last point by same amount as 2nd last point

            coord_shift[:, :, axes] = coord_shift[:, :, axes] + sample_shift_frac*diff_corr # add some fraction of the diff


    # negativve shift to decrease recon coords (gradients were late)
    else:
        coord_shift = np.copy(coord)
        for axes in range(3):
            diff_corr = np.zeros(coord[:, :, 0].shape) # [spokes, readout]
            diff = np.diff(coord[:, :, axes], axis=-1) # take diff between each sample along readout dim
            diff_corr[:, 1:] = diff
            diff_corr[:, 0] = diff[:, 0] # shift first point by same amount as 2nd point (assume gradients havent changed much)

            coord_shift[:, :, axes] = coord_shift[:, :, axes] + sample_shift_frac*diff_corr # add some fraction of the diff

    return coord_shift
