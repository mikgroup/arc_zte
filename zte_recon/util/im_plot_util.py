import numpy as np
import cfl
from .gif import array_to_gif_pillow


def normalize_im_stack(stack):
    
    stack_new = []
    for im in stack:
        
        stack_new.append(im / np.percentile(abs(im), 95))
        
    return np.array(stack_new)


def flip_roll(im, axis):
    '''Need to flip and circshift to maintain [-128,..., 0, ...127]
    '''

    return np.roll(np.flip(im, axis), shift=1, axis=axis)


def save_triplane_gifs_timeplot_for_path(path, ax_slice, cor_slice, sag_slice,
                                         gif_basepath='./gifs_self_nav_patient' ):
    
    im_view = np.squeeze(cfl.readcfl(path))
    
    parts = path.rstrip("/").split("/")[-2:]  # get last two
    dir_name, file_name = parts  # unpack into two variables

    gif_path = f"{gif_basepath}/{dir_name}_{file_name}"
    
    array_to_gif_pillow(im_view[:, cor_slice].transpose(2,0,1), gif_path + '.gif', fps=10)
    array_to_gif_pillow(im_view[sag_slice].transpose(2,0,1), gif_path + '_sag.gif', fps=10)
    array_to_gif_pillow(im_view[:, :, ax_slice].transpose(2,0,1), gif_path + '_ax.gif', fps=10)

    print(f"Saved gifs to {gif_path}")