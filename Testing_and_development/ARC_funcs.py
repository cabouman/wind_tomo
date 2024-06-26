import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
from scipy.interpolate import PchipInterpolator, CubicSpline


def gen_wind_tunnel3(num_slices, num_rows, num_cols, left_freq=14, right_freq=16, center_rod=False,make_square=False):
    """
    Generate a phantom cheaply mimicking a wind tunnel

    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.

    Return:
        out_image: 2D array, (slices,num_rows, num_cols)
    """
    radius = 0.07
    axis_x = np.linspace(-1, 1, num_cols)
    axis_y = np.linspace(1, -1, num_rows)
    axis_z = np.linspace(radius+0.01, -radius-0.01, num_slices)

    y_grid,z_grid, x_grid = np.meshgrid(axis_y,axis_z, axis_x)
    image = x_grid * 0.0

    # image += _gen_roundrect(x_grid=x_grid, y_grid=y_grid, x0=0, y0=0, a=0.3, b=1.4, c=0.3, gray_level=.8)
    left_mid=-0.8
    right_mid=0.8
    left_thick=0.015
    right_thick=0.015
    right_amp=0.1
    left_amp=0.05
    #add left squiggle
    image += ( (
            x_grid <= right_amp * np.sin(y_grid * 2 * np.pi * right_freq / 4) + right_mid+right_thick) & (
                      x_grid >= 0.1 * np.sin(y_grid * 2 * np.pi * right_freq / 4) + right_mid-right_thick)) * 5
    #add right squiggle
    image += ( (
            x_grid >= left_amp * np.sin(y_grid * 2 * np.pi * left_freq / 4) + left_mid - left_thick) & (
                      x_grid <= 0.05 * np.sin(y_grid * 2 * np.pi * left_freq / 4) + left_mid + left_thick)) * 5
    #add center rod
    if center_rod:
        image += (((x_grid<=0.01) & (x_grid>=-0.01)) & ((y_grid<=0.01) & (y_grid>=-0.01)))*3

    xy_centers=[(-0.45,0.11),(-0.45,-0.1),(-0.25,0.12),(0.45,-0.05),(0.1,-0.15),(0.15,0.06),(0.35,0.11),(-0.15,-0.06)]
    xy_distort=[(1,2),(2,1),(2,1),(3,1),(1,3),(4,1),(1,3),(1,4)]
    value=[2,3,1,1,3,2,4,2]

    for i in range(len(xy_centers)):
        circle=xy_distort[i][0]*(x_grid - xy_centers[i][0])**2 + xy_distort[i][1]*(y_grid - xy_centers[i][1])**2 + (z_grid)**2 - radius**2
        image += np.where(circle<=0,value[i],0)

    if make_square==False:
        phantom=image
    else:
        #not that for an odd number (mdim - num_cols) or (mdim - num_cols) odd this changes center of rotation
        mdim=max(num_rows,num_cols)
        phantom=np.zeros((num_slices,mdim,mdim))
        row_range=int(floor((mdim-num_rows)/2))
        col_range = int(floor((mdim - num_cols)/2))
        phantom[:,row_range:row_range+num_rows,col_range:col_range+num_cols]=image
    return phantom


def windtunnel_block(sinogram,angles,num_rows,num_cols):
    """
    Modify a sinogram to simulate CT through window

    Args:
        sinogram(ndarray): 2D sinogram
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        num_rows(int): number of rows in original image
        num_cols(int): number of cols in original image
    Return:
        newsinogram(ndarray): 2D sinogram with regions set to zero were window edges would block

    """
    #make copy
    newsino=sinogram.copy()
    num_channel=newsino.shape[2]
    #find center index
    center=num_channel/2 # This will need to be changed eventually to accommodate a different center of rotation.
    for i,theta in enumerate(angles):
        #find FOV length
        d=num_rows*np.cos(theta) - num_cols*np.sin(abs(theta))
        #determine middle section
        lowInd= int(round(center - d/2))
        highInd= int(round(center + d/2))+1
        # set lower and upper sections to zero
        newsino[i,:,:lowInd]=0
        newsino[i,:,highInd:]=0

    return newsino


def circ_block(view,diameter,center_offset=(0,0)):
    center=(view.shape[0]//2+center_offset[0],view.shape[1]//2+center_offset[1])
    H, W =view.shape
    x, y = np.mgrid[:H, :W]
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    circ = (r<=diameter/2)*1
    return circ*view

def multi_circ_block(view,diameter,num_stack=1,stack_offset=0, left_right_offset=0):
    circ = 0
    H, W = view.shape
    x, y = np.mgrid[:H, :W]
    stack_positions = [j * stack_offset - (num_stack - 1) * stack_offset / 2 for j in range(num_stack)]
    for y_pos in stack_positions:
        center=(view.shape[0]//2+y_pos,view.shape[1]//2+left_right_offset)
        r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
        circ += (r<=diameter/2)*1
    return view*(circ>0)

# def sino_window_and_circ_block(sinogram,angles,diameters,window_dim):
#     """
#     Modify a sinogram to simulate CT through window
#
#     Args:
#         sinogram(ndarray): 3D sinogram
#         angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
#         num_rows(int): number of rows in original image
#         num_cols(int): number of cols in original image
#     Return:
#         newsinogram(ndarray): 2D sinogram with regions set to zero were window edges would block
#
#     """
#     num_rows, num_cols = window_dim #(# of slices, # of channels)
#     #make copy
#     newsino=sinogram.copy()
#     num_channel=newsino.shape[2]
#     #find center index
#     center=num_channel/2 # This will need to be changed eventually to accommodate a different center of rotation.
#     for i,theta in enumerate(angles):
#         #find FOV length
#         d=num_rows*np.cos(theta) - num_cols*np.sin(abs(theta))
#         #determine middle section
#         lowInd= int(round(center - d/2))
#         highInd= int(round(center + d/2))+1
#         # set lower and upper sections to zero
#         newsino[i,:,:lowInd]=0
#         newsino[i,:,highInd:]=0
#
#         newsino[i,:,:]=circ_block(newsino[i,:,:],diameters[i])
#
#     return newsino

def sino_window_and_circ_block(sinogram,angles,slice_dim,diameter,num_stack=1,stack_offset=0,center_offset=0):
    """
    Modify a sinogram to simulate CT through window

    Args:
        sinogram(ndarray): 3D sinogram
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        num_rows(int): number of rows in original image
        num_cols(int): number of cols in original image
    Return:
        newsinogram(ndarray): 2D sinogram with regions set to zero were window edges would block

    """
    num_rows, num_cols = slice_dim #(# of slices, # of channels)
    #make copy
    newsino=sinogram.copy()
    num_channel=newsino.shape[2]
    #find center index
    center=num_channel/2 # This will need to be changed eventually to accommodate a different center of rotation.
    for i,theta in enumerate(angles):
        #find FOV length
        d=num_rows*np.cos(theta) - num_cols*np.sin(abs(theta))
        #determine middle section
        lowInd= int(round(center - d/2))
        highInd= int(round(center + d/2))+1
        # set lower and upper sections to zero
        newsino[i,:,:lowInd]=0
        newsino[i,:,highInd:]=0
        oldsino=newsino.copy()

        # do beam circle thing
        newsino[i,:,:]=multi_circ_block(oldsino[i,:,:],diameter,num_stack,stack_offset,center_offset*np.sin(-theta))

    return newsino

def JAX_sino_window_and_circ_block(sinogram,angles,diameters,window_dim):
    """
    Modify a sinogram to simulate CT through window

    Args:
        sinogram(ndarray): 2D sinogram
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        num_rows(int): number of rows in original image
        num_cols(int): number of cols in original image
    Return:
        newsinogram(ndarray): 2D sinogram with regions set to zero were window edges would block

    """
    num_rows, num_cols = window_dim #(# of slices, # of channels)
    #make copy
    newsino=sinogram.copy()
    num_channel=newsino.shape[2]
    #find center index
    center=num_channel/2 # This will need to be changed eventually to accommodate a different center of rotation.
    for i,theta in enumerate(angles):
        #find FOV length
        d=num_rows*np.cos(theta) - num_cols*np.sin(abs(theta))
        #determine middle section
        lowInd= int(round(center - d/2))
        highInd= int(round(center + d/2))+1
        # set lower and upper sections to zero
        newsino=newsino.at[i,:,:lowInd].set(0)
        newsino=newsino.at[i,:,highInd:].set(0)

        newsino=newsino.at[i,:,:].set(circ_block(newsino[i,:,:],diameters[i]))

    return newsino

def grid_rescale_and_pad(arr,rescale_factor,upsample_factor=1,center_offset=0,monotonic=False):
    col_scale=rescale_factor
    num_rows,num_cols = arr.shape
    col_grid = np.linspace(0,num_cols-1,num_cols)
    if monotonic:
        interp=PchipInterpolator(col_grid,arr,axis=1,extrapolate=False)
    else:
        interp=CubicSpline(col_grid,arr,axis=1,extrapolate=False)

    new_length=int(round(num_cols*upsample_factor))
    col_interpts=np.zeros(new_length)
    col_interpts[0]= ((num_cols-1)/2 + center_offset)*(1- col_scale)
    for i in range(new_length):
        col_interpts[i]=col_interpts[0] + (rescale_factor/upsample_factor)*i

    scaled_arr = np.nan_to_num(interp(col_interpts))
    return scaled_arr

# def ARC_sino_transform_old(sino,angles,gamma,mu=1,center_offset=0,monotonic=False):
#     num_channels=sino.shape[2] #
#     num_slices = sino.shape[1] #1
#     new_channels=int(round(num_channels*mu))
#     sino_tilde= np.zeros((sino.shape[0],num_slices,new_channels))
#     new_angles=np.zeros(angles.shape)
#     for thetaidx, theta in enumerate(angles):
#         if theta==np.pi/2:
#             theta_tilde= np.pi/2
#         else:
#             theta_tilde= np.arctan2(gamma*np.sin(theta),np.cos(theta))
#
#         beta= np.sqrt((gamma**2)*(np.sin(theta)**2)+np.cos(theta)**2)/gamma
#         if theta_tilde==0:
#             alpha=gamma
#         else:
#             alpha= gamma*np.sin(theta)/np.sin(theta_tilde)
#
#         sino_tilde[thetaidx,:,:]=beta*grid_rescale_and_pad(sino[thetaidx,:,:],alpha,mu,center_offset=center_offset,monotonic=monotonic)
#
#         new_angles[thetaidx]=theta_tilde
#     return sino_tilde, new_angles

def ARC_sino_transform(sino,angles,gamma,mu=1,center_offset=0):
    num_channels=sino.shape[2] #
    num_slices = sino.shape[1] #1
    new_channels=int(round(num_channels*mu))
    sino_tilde= np.zeros((sino.shape[0],num_slices,new_channels))
    new_angles=np.zeros(angles.shape)
    for thetaidx, theta in enumerate(angles):
        theta_tilde= np.arctan2(gamma*np.sin(theta),np.cos(theta))
        alpha= np.sqrt((gamma**2)*(np.sin(theta)**2)+np.cos(theta)**2)
        sino_tilde[thetaidx,:,:]=(alpha/gamma)*grid_rescale_and_pad(sino[thetaidx,:,:],alpha,mu,center_offset=center_offset)
        new_angles[thetaidx]=theta_tilde
    return sino_tilde, new_angles

def find_diameters(angles,dist,thick):
    return [(dist*np.sin(abs(theta)) + thick*np.cos(abs(theta))) for theta in angles]


def find_min_diam_from_max(max_diam, max_angle,dist):
    if max_diam<=dist*np.sin(abs(max_angle)):
        print(f"distance must be less than {max_diam/np.sin(abs(max_angle))}. Returning nan...")
        min_val=np.nan
    else:
        min_val= (max_diam-dist*np.sin(abs(max_angle)))/np.cos(abs(max_angle))
    return min_val