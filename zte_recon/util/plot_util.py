import numpy as np

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def get_viridis_color_grad(nPoints):
    color5 = "#fde725"
    color4 = "#5ec962"
    color3 = "#21918c"
    color2 = "#3b528b"
    color1 = "#440154"

    num_points = int(np.ceil(nPoints/4))
    color_grad = [*get_color_gradient(color1, color2, num_points), 
                *get_color_gradient(color2, color3, num_points), 
                *get_color_gradient(color3, color4, num_points), 
                *get_color_gradient(color4, color5, num_points)]
    
    return color_grad