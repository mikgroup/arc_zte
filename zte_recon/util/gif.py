import numpy as np
import multiprocessing as mp
import os
from PIL import Image, ImageEnhance
from array2gif import write_gif

def gif_gray(arr, fname, fps=8):
    '''
    Write time+2D array into gif. Time should be along first axis
    arr [time, x, y]

    '''
    assert arr.ndim == 3, "Array must be 3D"

    arr_rgb = np.array(abs(arr), copy=True, dtype="float32")
    arr_rgb /= np.percentile(arr_rgb, 95) # normalize for mri windowing

    arr_rgb -= arr_rgb.min()
    arr_rgb /= np.ptp(arr_rgb)
    arr_rgb *= 255

    arr_rgb = np.repeat(arr_rgb[:, None], 3, 1)

    write_gif(arr_rgb.astype(np.uint8), fname, fps)


def save_triplane_gifs_timeplot_for_path(im_to_plot, 
                                         im_path, 
                                         ax_slice=None, 
                                         cor_slice=None, 
                                         sag_slice=None,
                                         gif_basepath='./gifs_from_recon', 
                                         fps=8):
    
    '''Save 3-plane gifs from volume im_to_plot. Save gifs at
    gif_basepath using name from im_path

    Dims of im_to_plot: [sag_slice, cor_slice, ax_slice, nTimeFrames]

    '''

    parts = im_path.rstrip("/").split("/")[-2:]  # get last two
    dir_name, file_name = parts  # unpack into two variables

    gif_path = f"{gif_basepath}/{dir_name}_{file_name}"
    os.makedirs(gif_basepath, exist_ok=True)
    
    if cor_slice is not None:
        gif_gray(im_to_plot[:, cor_slice].transpose(2,0,1), 
                            gif_path + f'_cor_sl{cor_slice}.gif', fps)
    if sag_slice is not None:
        gif_gray(im_to_plot[sag_slice].transpose(2,0,1), 
                            gif_path + f'_sag_sl{sag_slice}.gif', fps)
    if ax_slice is not None:
        gif_gray(im_to_plot[:, :, ax_slice].transpose(2,0,1), 
                            gif_path + f'_ax_sl{ax_slice}.gif', fps)
    
    print(f"Saved gifs to {gif_path}")
