from array2gif import write_gif
import numpy as np
import multiprocessing as mp
import os
import time
from tqdm import tqdm
import subprocess
from PIL import Image, ImageEnhance, ImageSequence

def gif_gray(arr, fname, fps=10, brightness=0, contrast=1.0):
    '''
    Write time+2D array into gif. Time should be along first axis
    arr [time, x, y]

     brightness : float
        Additive brightness offset. For uint8 use 0-255 scale. For float [0,1], use 0-1.
    contrast : float
        Multiplicative contrast factor. 1.0 = no change.

    '''
    assert arr.ndim == 3, "Array must be 3D"

    arr_rgb = np.array(abs(arr), copy=True, dtype="float32")
    arr_rgb /= np.percentile(arr_rgb, 95) # normalize for mri windowing

    arr_rgb = arr_rgb * contrast + brightness

    arr_rgb -= arr_rgb.min()
    arr_rgb /= np.ptp(arr_rgb)
    arr_rgb *= 255

    arr_rgb = np.repeat(arr_rgb[:, None], 3, 1)

    write_gif(arr_rgb.astype(np.uint8), fname, fps)
