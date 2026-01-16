import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util.plot_util import get_viridis_color_grad

def plot_3d_axes():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    return ax

def plot_spoke_endpoints_3d(ax, spoke_arr, color='tab:blue'):
    
    # plot k-space spokes - note the plot's interactive 
    plot_sphere(ax, rad=0.5)

    spoke_arr_plot = spoke_arr / 2*spoke_arr.max()
    
    for i in range(len(spoke_arr)):
        ax.plot(spoke_arr_plot[i, -1, 0], spoke_arr_plot[i, -1, 1], spoke_arr_plot[i, -1, 2], color)

    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')   


def plot_sphere(ax, rad=0.5):
    # plot transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = rad * np.outer(np.cos(u), np.sin(v))
    y = rad * np.outer(np.sin(u), np.sin(v))
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='y', linewidth=0, alpha=0.1)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')


def plot_sphere_opaque(ax, rad=0.5):
    # plot transparent sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = rad * np.outer(np.cos(u), np.sin(v))
    y = rad * np.outer(np.sin(u), np.sin(v))
    z = rad * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='y', linewidth=0)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')


def plot_ksp_spoke(ax, spoke, color, normalize=False):
    # plot spoke with dimensions [3, nPoints]
    norm_end = 2*np.linalg.norm(spoke[:, -1])

    if normalize == True:
        ax.plot(spoke[0]/norm_end, spoke[1]/norm_end, spoke[2]/norm_end, color)
    else: 
        ax.plot(spoke[0], spoke[1], spoke[2], color)


def plot_waveforms(grad_cat, dt):
    # grad dims [3, nReadoutPoints*nSpokes] in mT/m - can also be used for slew
    # dt = time b/w samples in seconds 

    time = np.linspace(0, grad_cat.shape[1]*dt, grad_cat.shape[1])
    time_ms = time*(10**3)

    plt.figure()
    plt.plot(time_ms, np.linalg.norm(grad_cat, axis=0), label='|G|')
    plt.plot(time_ms, grad_cat[0], label='Gx')
    plt.plot(time_ms, grad_cat[1], label='Gy')
    plt.plot(time_ms, grad_cat[2], label='Gz')
    plt.legend()
    plt.ylabel('Gradient amplitude [mT/m]')
    plt.xlabel('Time [ms]')

    
def plot_slew_waveforms(grad_cat, dt):
    # grad dims [3, nReadoutPoints*nSpokes] in mT/m - can also be used for slew
    # dt = time b/w samples in seconds 

    time = np.linspace(0, grad_cat.shape[1]*dt, grad_cat.shape[1])
    time_ms = time*(10**3)

    plt.figure()
    plt.plot(time_ms, np.linalg.norm(grad_cat, axis=0), label='|S|')
    plt.plot(time_ms, grad_cat[0], label='Sx')
    plt.plot(time_ms, grad_cat[1], label='Sy')
    plt.plot(time_ms, grad_cat[2], label='Sz')
    plt.legend()
    plt.ylabel('Slew amplitude [T/m/s]')
    plt.xlabel('Time [ms]')


def plot_format_ismrm_abstract(ax):
    # Format figure for Arc ZTE ISMRM abstract
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_zticks([-0.5, 0, 0.5])
    # ax.set_xticklabels([-1, 0, 1])
    # ax.set_yticklabels([-1, 0, 1])
    # ax.set_zticklabels([-1, 0, 1])
    ax.set_xlabel(r'$\mathrm{k_x}$', fontsize=15)
    ax.set_ylabel(r'$\mathrm{k_y}$', fontsize=15)
    ax.set_zlabel(r'$\mathrm{k_z}$', fontsize=15)


def plot_coherence_pathways(DataObj, nSpokes_plot=20, spoke_start_idx=0, ylim=2):
    '''
    Coords dim [nSpokes, nPts, 3]
    '''
    # Initialize array to store kstronauts to plot
    kstronauts = np.empty((nSpokes_plot, 3, DataObj.nPtsPerSpoke))
    nPoints =  DataObj.nPtsPerSpoke
    
     # Normalize coords from DataObj
    coords_normalized = DataObj.coord_radial_spokes / (2*DataObj.coord_radial_spokes.max())

    for i in range(nSpokes_plot):
        
        # Get ith spoke
        spoke = coords_normalized[i+spoke_start_idx].transpose(1,0)
        kstronauts[i] = spoke
        
        # Plot all kstronaut from ith excitation
        if (i == 0):
            c = 'b'
            label='From TR #' + str(spoke_start_idx+1)
        elif (i == 1):
            c = 'c'
            label='From TR #' + str(spoke_start_idx+2)
        else:
            c = 'y'
            label=None
        plt.plot(np.arange(i*nPoints, (i+1)*nPoints, 1), np.linalg.norm(spoke, axis=0), c, label=label)

        # Plot all previous coherences evolving during same TR
        # Plot only 60 previous coherences, assuming complex T2* dephasing
        for j in np.arange(0, i):
            kstronauts[j] = spoke + kstronauts[j][:, -1][:, None] # update previous kstronauts
            label=None

            if (j == 0):
                c = 'b'

            elif (j == 1):
                c = 'c'

            else:
                c = 'y'
            plt.plot(np.arange(i*nPoints, (i+1)*nPoints, 1), np.linalg.norm(kstronauts[j], axis=0), c)


    plt.ylabel('Cycles/voxel')
    plt.xlabel('time')
    plt.ylim([0, ylim])
    plt.yticks([0, 0.5, 1, ylim])
    ax = plt.gca()
    
    ax.set_yticklabels(['0', '0.5', '1', str(ylim)])

    plt.tick_params(left = True, right = False , labelleft = True , 
                    labelbottom = False, bottom = False) 
    plt.legend(loc='upper right')
    plt.axhline(y=0.5, linestyle='--', c='crimson', alpha=0.75)
    plt.axhline(y=1, linestyle='--', c='green', alpha=0.75)




def plot_coherence_pathways_from_coords(coords, nSpokes_plot=20, spoke_start_idx=0):
    '''
    Coords dim [nSpokes, nPts, 3]
    '''
    [nSpokes, nPoints, nDims] = coords.shape
    
    # Initialize array to store kstronauts to plot
    kstronauts = np.empty((nSpokes_plot, 3, nPoints))
    
    # Normalize coords from DataObj
    coords_normalized = coords / (2*coords.max())

    for i in range(nSpokes_plot):
        
        # Get ith spoke
        spoke = coords_normalized[i+spoke_start_idx].transpose(1,0)
        kstronauts[i] = spoke
        
        # Plot all kstronaut from ith excitation
        if (i == 0):
            c = 'b'
            label='From TR #'+str(spoke_start_idx+1)
        elif (i == 1):
            c = 'c'
            label='From TR #'+str(spoke_start_idx+2)
        else:
            c = 'y'
            label=None
        plt.plot(np.arange(i*nPoints, (i+1)*nPoints, 1), np.linalg.norm(spoke, axis=0), c, label=label)

        # Plot all previous coherences evolving during same TR
        for j in range(i):
            kstronauts[j] = spoke + kstronauts[j][:, -1][:, None] # update previous kstronauts
            label=None

            if (j == 0):
                c = 'b'

            elif (j == 1):
                c = 'c'

            else:
                c = 'y'
            plt.plot(np.arange(i*nPoints, (i+1)*nPoints, 1), np.linalg.norm(kstronauts[j], axis=0), c)


    plt.ylabel('Cycles/voxel')
    plt.xlabel('time')
    plt.ylim([0, 2])
    plt.yticks([0, 0.5, 1, 2])
    ax = plt.gca()
    ax.set_yticklabels(['0', '0.5', '1', '2'])

    plt.tick_params(left = True, right = False , labelleft = True , 
                    labelbottom = False, bottom = False) 
    plt.legend(loc='upper right')

    plt.axhline(y=0.5, linestyle='--', c='r')
    plt.axhline(y=1, linestyle='--', c='m')


def plot_spokes_temporal_color(ax, coords_in, nSpokes_plot=384, elev=-174, azim=-53):
    '''Coords dim[nSpokes, 3, nPts]
    '''
    color_grad = get_viridis_color_grad(nSpokes_plot)
    
    # normalize spokes
    coords_normalized = coords_in / (2*coords_in.max())

    for i in range(nSpokes_plot):
        plot_ksp_spoke(ax, coords_normalized[i].transpose(1,0), color_grad[i])
        
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Ticks only at [-0.5, 0, 0.5]
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_zticks([-0.5, 0, 0.5])

    ax.view_init(elev=elev, azim=azim)



def plot_endpoints_temporal_color(ax, endpoints, style='.', nSpokes_plot=384, elev=-174, azim=-53):
    '''Endpoints dim[nSpokes, 3]
    '''
    
    if len(endpoints) < nSpokes_plot:
        nSpokes_plot = len(endpoints)

    color_grad = get_viridis_color_grad(nSpokes_plot)

    # normalize endpoints
    endpoints_normalized = endpoints / (2*endpoints.max())
    for i in range(nSpokes_plot):
        spoke = endpoints_normalized[i]
        ax.plot(spoke[0], spoke[1], spoke[2], c=color_grad[i], marker=style)


def plot_endpoints_single_color(ax, endpoints, style='.', nSpokes_plot=384, color='tab:blue', elev=-174, azim=-53, **kwargs):
    '''Endpoints dim[nSpokes, 3]
    '''
    
    if len(endpoints) < nSpokes_plot:
        nSpokes_plot = len(endpoints)

    color_grad = get_viridis_color_grad(nSpokes_plot)

    # normalize endpoints
    endpoints_normalized = endpoints / (2*endpoints.max())
    
    for i in range(nSpokes_plot):
        spoke = endpoints_normalized[i]
        ax.plot(spoke[0], spoke[1], spoke[2], c=color, marker=style, **kwargs)
        
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Ticks only at [-0.5, 0, 0.5]
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_zticks([-0.5, 0, 0.5])

    ax.view_init(elev=elev, azim=azim)