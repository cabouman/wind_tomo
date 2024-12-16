import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import jax
import mbirjax
import jax.numpy as jnp
import scipy.signal as signal
from numpy.linalg import lstsq
import numpy.ma as ma
from scipy.special import factorial
from scipy.ndimage import gaussian_filter


def remove_tip_tilt(arr, axis=None):
    """
    Remove tip-tilt from a 2D or 3D numpy array and ensure zero mean.

    Parameters:
    arr (numpy.ndarray): Input 2D or 3D array.
    axis (int, optional): Axis along which to remove the trend in case of 3D array. Default is None.

    Returns:
    numpy.ndarray: Array with linear trends removed and zero mean.
    """

    def fit_plane_2d(data, mask):
        """Fit and subtract a plane from 2D data."""
        m, n = data.shape
        X, Y = np.meshgrid(np.arange(n), np.arange(m))
        A = np.c_[X[mask].ravel(), Y[mask].ravel(), np.ones(mask.sum())]
        C, _, _, _ = lstsq(A, data[mask].ravel(), rcond=None)
        plane = (C[0] * X + C[1] * Y + C[2]).reshape(m, n)
        return data - plane

    def fit_plane_3d(data, mask, axis):
        """Fit and subtract planes from 3D data along the specified axis."""
        if axis == 0:
            for i in range(data.shape[0]):
                data[i] = fit_plane_2d(data[i], mask[i])
        elif axis == 1:
            for i in range(data.shape[1]):
                data[:, i] = fit_plane_2d(data[:, i], mask[:, i])
        elif axis == 2:
            for i in range(data.shape[2]):
                data[:, :, i] = fit_plane_2d(data[:, :, i], mask[:, :, i])
        return data

    was_masked = np.ma.is_masked(arr)

    if was_masked:
        mask = ~arr.mask
        arr = arr.data.copy()  # Work with a writable copy of the data
    else:
        mask = np.ones_like(arr, dtype=bool)

    if arr.ndim == 2:
        result = fit_plane_2d(arr, mask)
    elif arr.ndim == 3:
        if axis is None:
            raise ValueError("Axis must be specified for 3D arrays.")
        result = fit_plane_3d(arr, mask, axis)
    else:
        raise ValueError("Input array must be 2D or 3D.")

    if was_masked:
        result = np.ma.array(result, mask=~mask)

    return result


def zernike_radial(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial (m, n) on the unit disk.
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    radial = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        radial += rho ** (n - 2 * k) * (-1) ** k * factorial(n - k) / (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
    return radial


def zernike_polynomial(m, n, rho, theta):
    """
    Calculate the normalized Zernike polynomial (m, n) on the unit disk.
    """
    radial = zernike_radial(abs(m), n, rho)
    if m == 0:
        Zmn = radial
    elif m > 0:
        Zmn = radial * np.cos(m * theta)
    else:
        Zmn = radial * np.sin(abs(m) * theta)

    if n != 0:
        Zmn = Zmn - np.average(Zmn[rho <= 1])

    return Zmn


def fit_zernike(image, max_radial_degree, pixel_diameter=None):
    """
    Fit Zernike polynomials to a 2D image over a disk.

    Parameters:
        image (2D array): The input image.
        max_radial_degree (int): The maximum radial degree of Zernike polynomials.
        pixel_diameter (int, optional): The pixel diameter of the disk. Defaults to the minimum side length of the image.

    Returns:
        list of tuples: Zernike coefficients organized by radial degree.
    """
    # Get the dimensions of the image
    height, width = image.shape

    # Set pixel_diameter to the minimum side length if not provided
    if pixel_diameter is None:
        pixel_diameter = min(height, width)

    # Determine the center and radius of the disk
    center_y, center_x = height // 2, width // 2
    radius = pixel_diameter // 2

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate normalized coordinates within the disk
    rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / radius
    theta = np.arctan2(y - center_y, x - center_x)

    # Create a mask for the disk
    mask = rho <= 1

    # Initialize list to store Zernike coefficients
    zernike_coeffs = []

    # Iterate over each radial degree
    for n in range(max_radial_degree + 1):
        coeffs_n = []
        for m in range(-n, n + 1, 2):
            # Calculate normalized Zernike polynomial (m, n) on the masked region
            Zmn = zernike_polynomial(m, n, rho[mask], theta[mask])

            # Flatten the masked image and Zernike polynomial
            image_flat = image[mask].flatten()
            Zmn_flat = Zmn.flatten()

            # Calculate the coefficient using inner product
            coeff, _, _, _ = lstsq(Zmn_flat[:, np.newaxis], image_flat, rcond=None)

            coeffs_n.append(coeff)

        zernike_coeffs.append(tuple(coeffs_n))

    return zernike_coeffs


def reconstruct_image(zernike_coeffs, pixel_diameter):
    """
    Reconstruct a 2D image from Zernike coefficients.

    Parameters:
        zernike_coeffs (list of tuples): Zernike coefficients organized by radial degree.
        pixel_diameter (int): The pixel diameter of the disk.

    Returns:
        np.ma.MaskedArray: Reconstructed 2D image with the region outside the inner disc masked.
    """
    # Determine the radius of the disk
    radius = pixel_diameter // 2

    # Create coordinate grids
    y, x = np.ogrid[:pixel_diameter, :pixel_diameter]

    # Calculate normalized coordinates within the disk
    center_y, center_x = pixel_diameter // 2, pixel_diameter // 2
    rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / radius
    theta = np.arctan2(y - center_y, x - center_x)

    # Create a mask for the disk
    mask = rho <= 1

    # Initialize the reconstructed image
    reconstructed_image = np.zeros((pixel_diameter, pixel_diameter), dtype=np.float64)

    # Iterate over each tuple in the list of coefficients
    for coeffs in zernike_coeffs:
        n = len(coeffs) - 1
        for m_index, coeff in enumerate(coeffs):
            m = 2 * m_index - n
            if coeff != 0:
                Zmn = zernike_polynomial(m, n, rho[mask], theta[mask])
                reconstructed_image[mask] += coeff * Zmn

    # Create a masked array with the region outside the inner disc masked
    reconstructed_image = np.ma.masked_array(reconstructed_image, mask=~mask)

    return reconstructed_image


def gaussian_kernel2d(nsig, truncation=4):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    kernlen = 1 + 2 * np.ceil(truncate * nsig).astype(int)
    inp = np.zeros((kernlen, kernlen))
    inp[kernlen // 2, kernlen // 2] = 1
    return gaussian_filter(inp, nsig)


def gaussian_kernel3d(nsig=np.array([3, 3, 3]), truncate=4):
    """Returns a 3D Gaussian kernel array."""
    kernlen = 1 + 2 * np.ceil(truncate * nsig).astype(int)
    # create nxn zeros
    inp = np.zeros(kernlen)
    # set element at the middle to one, a dirac delta
    inp[kernlen[0] // 2, kernlen[1] // 2, kernlen[2] // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    filt = gaussian_filter(inp, nsig)

    return filt


def Gaussian_plugandplay(ct_model, sinogram, weights, sigma, truncate, num_iterations, init_recon=0, convg=0.01,
                         show_iter=25):
    # make kernel
    kernel = jnp.array(gaussian_kernel3d(sigma, truncate))

    if isinstance(init_recon, int):
        x = jnp.zeros(ct_model.get_params('recon_shape'))
        v = jnp.zeros(ct_model.get_params('recon_shape'))
    else:
        x = init_recon
        v = jax.scipy.signal.correlate(x, kernel, 'same')

    # initialize arrays
    u = jnp.zeros(ct_model.get_params('recon_shape'))
    error = 1
    iter = 0
    while error > convg and iter < 3000:
        x, _ = ct_model.prox_map(v - u, sinogram, weights, num_iterations=num_iterations, init_recon=x)
        v = jax.scipy.signal.correlate(x + u, kernel, 'same')
        u = u + (x - v)
        error = np.linalg.norm(x - v)

        iter += 1
        if iter % show_iter == 0:
            print(f"iteration {iter}: change {error}")

    return x


def convolve2D(recon, kern):
    return jax.scipy.signal.convolve2d(recon, kern, 'same')


v_2Dconvolve = jax.jit(jax.vmap(convolve2D, in_axes=[0, None], out_axes=0))


def Gaussian_filter_and_zero_out_btw_screens(recon,kernel,screen_ind):
    '''
    Sets xz-planes (of recon) to zero at all index locations along
    the y-axis NOT included in the list screen_ind. Also performs 2D
    convolutional filtering with kernel of the xz-planes (of recon) at
    all index locations along the y-axis that ARE included in the list
    screen_ind.
    Args:
        recon (ndarray): jax numpy 3D Phantom (y,x,z) with 2D phase screens along first axis y
        kernel (ndarray): 2D convolution kernel
        screen_ind (list, int): list of int index location for the placement of the phase screens

    Returns:
        ndarray (float): Modified jax numpy 3D Phantom (y,x,z)
    '''

    # zero out
    reconnew = jnp.zeros(recon.shape)
    # filter
    recon = reconnew.at[screen_ind, :, :].set(v_2Dconvolve(recon[screen_ind, :, :], kernel))
    return recon


def plugandplay_for_screens(ct_model, sinogram, weights, screen_ind, num_iterations=10):
    '''
        Performs Plug-and-Play tomography reconstruction with a prior model that zeroes out all planes along the y-axis that are not given by screen_ind.
        Args:
            ct_model (mbirjax.TomographyModel): Instance of TomographyModel class from mbirjax
            sinogram (ndarray): jax numpy sinogram array
            weights (ndarray): jax numpy sinogram weights array.
            screen_ind (list, int): list of int index location for the placement of the phase screens
            num_iterations (int, optional): number of iterations to perform for ct_model.prox_map()

        Returns:
            ndarray (float): jax numpy 3D reconstruction
            err_vec (list): list of iteration step sizes.
        '''

    x = jnp.zeros(ct_model.get_params('recon_shape'))

    ind_to_zero = list(set([i for i in range(x.shape[0])]).difference(screen_ind))

    # initialize arrays
    u = jnp.zeros(ct_model.get_params('recon_shape'))
    v = jnp.zeros(ct_model.get_params('recon_shape'))
    err_vec = []
    error = 1000
    iter = 0
    while error > 0.25 and iter < 700:
        x, _ = ct_model.prox_map(v - u, sinogram, weights, num_iterations=num_iterations, init_recon=x)
        v = v.at[:, :, :].set(0)
        v = v.at[screen_ind, :, :].set((x + u)[screen_ind, :, :])
        u = u + (x - v)
        error = np.linalg.norm(x - v)
        err_vec.append(error)
        iter += 1
        if iter % 10 == 0:
            print(f"iteration {iter}: change {error}")
    return x, err_vec

def Gaussian_plugandplay_for_screens(ct_model, sinogram, weights, sigma, truncate, screen_ind, num_iterations=10):
    '''
    Performs Plug-and-Play tomography reconstruction with a prior model that performs Gaussian filtering for the xy-planes along the y-axis at index locations given by screen_ind and zeroes out all other planes along the y-axis.
    Args:
        ct_model (mbirjax.TomographyModel): Instance of TomographyModel class from mbirjax
        sinogram (ndarray): jax numpy sinogram array
        weights (ndarray): jax numpy sinogram weights array.
        sigma (float): parameter for the amount of gaussian blurring. stand dev of gaussian kernel in pixels
        screen_ind (list, int): list of int index location for the placement of the phase screens
        truncate (int, optional): number of sigma values at which to truncate gaussian kernel
        num_iterations (int, optional): number of iterations to perform for ct_model.prox_map()

    Returns:
        ndarray (float): jax numpy 3D reconstruction
    '''

    # make kernel
    kernel = jnp.array(gaussian_kernel2d(sigma, truncate))

    # initialize arrays
    x = jnp.zeros(ct_model.get_params('recon_shape'))
    u = jnp.zeros(ct_model.get_params('recon_shape'))
    v = jnp.zeros(ct_model.get_params('recon_shape'))
    err_vec = []
    error = 1000
    iter = 0
    while error > 0.25 and iter < 700:
        x, _ = ct_model.prox_map(v - u, sinogram, weights, num_iterations=num_iterations, init_recon=v)
        v = Gaussian_filter_and_zero_out_btw_screens(x + u, kernel, screen_ind)
        u = u + (x - v)
        error = np.linalg.norm(x - v)
        iter += 1
        if iter % 10 == 0:
            print(f"iteration {iter}: change {error}")
        err_vec.append(error)

    return x, err_vec


def Gaussian_plugandplay(ct_model, sinogram, weights, sigma, truncate, iterations, init_recon=0, convg=0.01,
                         show_iter=25):
    # make kernel
    kernel = jnp.array(gaussian_kernel3d(sigma, truncate))

    if isinstance(init_recon, int):
        x = jnp.zeros(ct_model.get_params('recon_shape'))
        v = jnp.zeros(ct_model.get_params('recon_shape'))
    else:
        x = init_recon
        v = jax.scipy.signal.correlate(x, kernel, 'same')

    # initialize arrays
    u = jnp.zeros(ct_model.get_params('recon_shape'))
    error = 1
    iter = 0
    while error > convg and iter < 3000:
        x, _ = ct_model.prox_map(v - u, sinogram, weights, num_iterations=num_iterations, init_recon=x)
        v = jax.scipy.signal.correlate(x + u, kernel, 'same')
        u = u + (x - v)
        error = np.linalg.norm(x - v)

        iter += 1
        if iter % show_iter == 0:
            print(f"iteration {iter}: change {error}")

    return x


def ift3(X,scale=1):
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(X))) * scale


def ft_phase_volume(r0, N, delta, L0=np.inf, l0=0.0):
    """
    Generate phase volume consistent with random draw of atmospheric turbulence.

    Args:
        r0 (float): Fried's coherence length [m]
        N (int): Volume size (NxNxN)
        delta (float): grid sampling interval in [m]
        L0 (float, optional): [inf] von Karman PSD, one over outer scale frequency [m]
        l0 (float, optional): [0] von Karman PSD, one over inner scale frequency [m]

    Returns:
        ndarray (float): NxNxN phase Volume
    """

    # setup the PSD
    del_f = 1/(N*delta)     # frequency grid spacing [1/m]
    fx = np.arange(-N/2,N/2)*del_f
    fx,fy,fz = np.meshgrid(fx,fx,fx)
    f = np.sqrt(fx**2 + fy**2 + fz**2)

    # modified von Karman atmospheric phase PSD
    #fm = 5.92/l0/(2*np.pi)             # inner scale frequency [1/m]
    oneover_fm = l0*(2.0*np.pi)/5.92   # one over inner scale frequency [1/m]^-1 (avoid div by zero)
    f0 = 1/L0                           # outer scale frequency [1/m]

    #PSD_phi = 0.023*r0**(-5/3) * np.exp(-(f*oneover_fm)**2) / (f**2 + f0**2)**(11/6)
    with np.errstate(divide='ignore'):
        PSD_phi = 0.023*r0**(-5/3) * np.divide(np.exp(-(f*oneover_fm)**2), (f**2 + f0**2)**(11/6))
    PSD_phi[N//2,N//2,N//2] = 0

    # random draws of Fourier coefficients
    cn = (np.random.randn(N,N,N) + 1j*np.random.randn(N,N,N)) * np.sqrt(PSD_phi)*del_f
    phz = np.real(ift3(cn,N**3))

    return phz


def gen_wind_tunnel3(num_slices, num_rows, num_cols, left_freq=14, right_freq=16, boundaries=True, center_rod=False):
    """
    Generate a phantom cheaply mimicking a wind tunnel

    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        left_freq: int, frequency of left squiggle.
        right_freq: int, frequency of right squiggle.
        center_rod: bool, true places rod of magnitude 3 in center of wind tunnel

    Return:
        out_image: 2D array, (slices,num_rows, num_cols)
    """
    radius = 0.07
    axis_x = np.linspace(-1, 1, num_cols)
    axis_y = np.linspace(1, -1, num_rows)
    axis_z = np.linspace(1, -1, num_slices)

    y_grid,z_grid, x_grid = np.meshgrid(axis_y,axis_z, axis_x)
    image = x_grid * 0.0

    # image += _gen_roundrect(x_grid=x_grid, y_grid=y_grid, x0=0, y0=0, a=0.3, b=1.4, c=0.3, gray_level=.8)
    left_mid=-0.8
    right_mid=0.8
    left_thick=0.015
    right_thick=0.015
    right_amp=0.1
    left_amp=0.05
    if boundaries:
        #add left squiggle
        image += ( (
                x_grid <= right_amp * np.sin(y_grid * 2 * np.pi * right_freq / 4) + right_mid+right_thick) & (
                          x_grid >= right_amp * np.sin(y_grid * 2 * np.pi * right_freq / 4) + right_mid-right_thick)
                   #& (y_grid<=0.32)& (y_grid>=-0.3)
                   )  * 3
        #add right squiggle
        image += ( (
                x_grid >= left_amp * np.sin(y_grid * 2 * np.pi * left_freq / 4) + left_mid - left_thick) & (
                          x_grid <= left_amp * np.sin(y_grid * 2 * np.pi * left_freq / 4) + left_mid + left_thick)
                   #& (y_grid<=0.32)& (y_grid>=-0.3)
                    ) * 3
    #add center rod
    if center_rod:
        image += (((x_grid<=0.01) & (x_grid>=-0.01)) & ((y_grid<=0.01) & (y_grid>=-0.01)))*3

    xy_centers=[(-0.55,0.11,0),(-0.45,-0.1,-.1),(-0.25,0.12,.1),(0.6,-0.05,.1),(0.1,-0.15,0),(0.15,0.06,0),(0.45,0.11,.12),(-0.15,-0.06,-.1)]
    xy_distort=[(1,2,0.1),(2,1,0.7),(2,1,0.1),(3,1,0.1),(1,3,0.8),(4,1,0.1),(1,3,0.2),(1,4,0.1)]
    value=[2,3,1,1,3,2,4,2]

    for i in range(len(xy_centers)):
        circle=xy_distort[i][0]*(x_grid - xy_centers[i][0])**2 + xy_distort[i][1]*(y_grid - xy_centers[i][1])**2 + xy_distort[i][2]*(z_grid- xy_centers[i][2])**2 - radius**2
        image += np.where(circle<=0,value[i],0)

    return image


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
    """
    Set everything outside a disk equal to zero

    Args:
        view(ndarray): 2D array to be modified
        diameter(int): Diameter of disk in pixels
        center_offset(tuple): pixel center of disk relative center of array. (0,0) corresponds to center of array. (1,1) corresponds to 4th quadrant
    Return:
        modified_view(ndarray): 2D array

    """
    center=(view.shape[0]//2+center_offset[0],view.shape[1]//2+center_offset[1])
    H, W =view.shape
    x, y = np.mgrid[:H, :W]
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    circ = (r<=diameter/2)*1
    return circ*view

def multi_circ_block(view,diameter,num_stack=1,stack_offset=0, left_right_offset=0,alpha=1,for_weights=False):
    """
    Set everything outside a stack of disks to zero

    Args:
        view(ndarray): 2D array to be modified
        diameter(int): Diameter of disk in pixels
        num_stack (int): number of vertical stacks of disks
        stack_offset (int): pixel distance between the centers of the disks in the stack
        left_right_offset(int): pixel center of disk along the column axis relative to the center of array. 1 corresponds to right side.
    Return:
        modified_view(ndarray): 2D array

    """
    circ = 0
    H, W = view.shape
    x, y = np.mgrid[:H, :W]
    stack_positions = [j * stack_offset - (num_stack - 1) * stack_offset / 2 for j in range(num_stack)]
    for y_pos in stack_positions:
        center=(view.shape[0]//2+y_pos,view.shape[1]//2+left_right_offset)
        r = np.sqrt((x-center[0])**2 + ((y-center[1])*alpha)**2)
        circ += (r<=diameter/2)*1
    view=view * (circ > 0)
    if not for_weights:
        for i in range(H):
            if np.sum(circ[i, :]) > 0:
                left_ind = np.where(circ[i, :] > 0)[0][0]
                right_ind = np.where(circ[i, :] > 0)[0][-1]
                view[i, 0:left_ind] = view[i, left_ind]
                view[i, right_ind + 1:] = view[i, right_ind]
    return view


def sino_window_and_circ_block(sinogram,angles,recon_slice_dim,diameter,num_stack=1,stack_offset=0,center_offset=0,alpha_vals=1,for_weights=False):
    """
    Modify a sinogram to simulate CT with a stack of beams of a volume between two windows

    Args:
        sinogram(ndarray): 3D sinogram of shape (views, slices, channels)
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        diameter(ndarray or int): vector of integer pixel diameters for beams or one integer for all beam diameters
        num_stack(int): number of vertical stacks of disks
        stack_offset (int): pixel distance between the centers of the disks in the stack
        center_offset (int): pixel center of rotation along column axis. Allows user to place center closer to left or right side of recosntruction volume space.
    Return:
        newsinogram(ndarray): 3D sinogram where each view has been modified to account for the windows, and the placement of the beams
    """
    num_rows, num_cols = recon_slice_dim #(# of slices, # of channels)
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
        if np.size(alpha_vals)==1:
            newsino[i,:,:]=multi_circ_block(oldsino[i,:,:],diameter,num_stack,stack_offset,center_offset*np.sin(-theta),for_weights=for_weights)
        elif np.size(alpha_vals) == np.size(angles):
            newsino[i,:,:]=multi_circ_block(oldsino[i,:,:],diameter,num_stack,stack_offset,center_offset*np.sin(-theta),alpha=alpha_vals[i],for_weights=for_weights)
        else:
            raise Exception("Input for alpha_vals must either be an integer or a vector with the same length as angles")
    return newsino

def weights_window_and_circ_block(sinogram_shape,angles,diameter,num_stack=1,stack_offset=0,center_offset=0):
    """
    Make a weight sinogram to simulate CT with a stack of beams of a volume between two windows

    Args:
        sinogram_shape(tuple): tuple describing shape of sinogram (views, slices, channels)
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        diameter(int): pixel diameter of the beams
        num_stack(int): number of vertical stacks of disks
        stack_offset (int): pixel distance between the centers of the disks in the stack
        center_offset (int): pixel center of rotation along column axis. Allows user to place center closer to left or right side of recosntruction volume space.
    Return:
        newsinogram(ndarray): 3D weight array of shape (views, slices, channels) where each view has been modified to account for the windows, and the placement of the beams

    """
    num_rows, num_cols = sinogram_shape[1:3] #(# of slices, # of channels)
    #make copy
    newsino=np.ones(sinogram_shape)
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
        newsino[i,:,:]=multi_circ_block(oldsino[i,:,:],diameter,num_stack,stack_offset,-center_offset*np.sin(theta))

    return newsino

def JAX_sino_window_and_circ_block(sinogram,angles,diameters,window_dim):
    """
    Modify a sinogram to simulate CT through window

    Args:
        sinogram(ndarray): 2D sinogram
        angles(ndarray): 1D array of angles corresponding to the sinogram views
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

def grid_rescale_and_pad(view, compress_factor, upsample_factor=1, center_offset=0,for_weights=False):
    """
    Resample a sinogram view along the channel axes. The resampling period is downsample_factor/upsample_factor. The
    final result is then zero padded to be upsample_factor times it's original length along the channel axis. Function
    centers resampling grid at the center of the original channel axis + center_offset

    Args:
        view(ndarray): 2D sinogram view of shape (slices, channels)
        compress_factor(float): factor by which to compress along the channel axis. When upsample_factor=1
        upsample_factor(float): factor by which to upsample in order to compensate downsampling due to compression
        center_offset(float): distance in pixel increments to offset center of resampling grid from original center of
                              channel axis
    Return:
        compressed_view(ndarray): 2D sinogram view of shape (slices, channels*upsample_factor)
    """
    col_scale=compress_factor
    num_rows,num_cols = view.shape
    col_grid = np.linspace(0,num_cols-1,num_cols)
    if for_weights:
        interp=interp1d(col_grid, view,kind="nearest-up", axis=1, bounds_error=False, fill_value=0)
    else:
        interp=CubicSpline(col_grid, view, axis=1, extrapolate=False)
    new_length=int(round(num_cols*upsample_factor))
    col_interpts=np.zeros(new_length)
    col_interpts[0]= ((num_cols-1)/2 + center_offset)*(1- col_scale)
    for i in range(new_length):
        col_interpts[i]= col_interpts[0] + (compress_factor / upsample_factor) * i

    scaled_view = np.nan_to_num(interp(col_interpts))
    return scaled_view


def ARC_sino_transform(sino,angles,gamma,a=1,center_offset=0,for_weights=False):
    """
    Implements ARC sinogram transformation. Allows for the user to offset the resampling center of each sinogram view

    Args:
        sino(ndarray): 3D sinogram numpy array of shape (views, slices, channels)
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        gamma(float): factor by which to compress the reconstruction along the axis aligned with the zero angle view
        a(float): factor by which to upsample the sinogram along the channel axis
        center_offset(float): distance in pixel increments to offset center of resampling grid from original center of
                              channel axis
    Return:
        sino_tilde(ndarray): 3D sinogram numpy array of shape (views, slices, channels*upsample_factor)
        new_angles(ndarray): 1D numpy array of transformed angles corresponding to the sinogram views
    """
    num_views, num_slices,num_channels=sino.shape
    new_channels=int(round(num_channels*a))
    sino_tilde= np.zeros((num_views,num_slices,new_channels))
    new_angles=np.zeros(angles.shape)
    for thetaidx, theta in enumerate(angles):
        theta_tilde= np.arctan2(gamma*np.sin(theta),np.cos(theta))
        alpha= np.sqrt((gamma**2)*(np.sin(theta)**2)+np.cos(theta)**2)
        if for_weights:
            sino_tilde[thetaidx,:,:]=grid_rescale_and_pad(sino[thetaidx,:,:],alpha,a,
                                                          center_offset=center_offset,
                                                          for_weights=for_weights)
        else:
            sino_tilde[thetaidx, :, :] = (alpha / gamma) * grid_rescale_and_pad(sino[thetaidx, :, :], alpha, a,
                                                                                center_offset=center_offset)
        new_angles[thetaidx]=theta_tilde
    return sino_tilde, new_angles

def ARC_alpha_values(angles,gamma):
    """
    Implements ARC sinogram transformation. Allows for the user to offset the resampling center of each sinogram view

    Args:
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        gamma(float): factor by which to compress the reconstruction along the axis aligned with the zero angle view
    Return:
        alpha_vals(ndarray): 1D numpy array of individual view channel compressions
    """
    alpha_vals=np.zeros(angles.shape)
    for thetaidx, theta in enumerate(angles):
        alpha= np.sqrt((gamma**2)*(np.sin(theta)**2)+np.cos(theta)**2)
        alpha_vals[thetaidx]=alpha
    return alpha_vals

def ARC_angle_transform(angles,gamma):
    """
    Implements ARC sinogram transformation for the angles only

    Args:
        angles(ndarray): 1D numpy array of angles corresponding to the sinogram views
        gamma(float): factor by which to compress the reconstruction along the axis aligned with the zero angle view
    Return:
        new_angles(ndarray): 1D numpy array of transformed angles corresponding to the sinogram views
    """
    new_angles=np.zeros(angles.shape)
    for thetaidx, theta in enumerate(angles):
        theta_tilde= np.arctan2(gamma*np.sin(theta),np.cos(theta))
        new_angles[thetaidx]=theta_tilde
    return new_angles

def find_diameters(angles,dist,thick):
    return [(dist*np.sin(abs(theta)) + thick*np.cos(abs(theta))) for theta in angles]


def find_min_diam_from_max(max_diam, max_angle,dist):
    if max_diam<=dist*np.sin(abs(max_angle)):
        print(f"distance must be less than {max_diam/np.sin(abs(max_angle))}. Returning nan...")
        min_val=np.nan
    else:
        min_val= (max_diam-dist*np.sin(abs(max_angle)))/np.cos(abs(max_angle))
    return min_val