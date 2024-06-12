import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from dh import ft2, ift2

#copied from SVMBIR
def plot_image(img, title=None, filename=None, vmin=None, vmax=None):
    """
    Function to display and save a 2D array as an image.

    Args:
        img: 2D numpy array to display
        title: Title of plot image
        filename: A path to save plot image
        vmin: Value mapped to black
        vmax: Value mapped to white
    """

    plt.ion()
    fig = plt.figure()
    imgplot = plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.title(label=title)
    imgplot.set_cmap('gray')
    plt.colorbar()
    if filename != None:
        try:
            plt.savefig(filename)
        except:
            print("plot_image() Warning: Can't write to file {}".format(filename))


def imshow_hi_res(array, title='', vmin=None, vmax=None, cmap='viridis', show=False):
    """ Display an array using a figure size large enough to map one array element to one pixel
    The scaling isn't exactly correct, so some pixels are slightly larger than others, partially depending on the
    labels on the axes and on the colorbar.  It's difficult to get this exactly correct.
    Args:
        array: The array to display
        title: The title for the plot
        vmin: Minimum of the intensity window - same as vmin in imshow
        vmax: Maximum of the intensity window - same as vmax in imshow
        cmap: The color map as in imshow - same as cmap in imshow
        show: If true, then plt.show() is called to display immediately, otherwise call
              fig.show() on the object returned from this function to show the plot.

    Returns:
        The pyplot figure object
    """
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    # Scale for one pixel per element and account for the border
    fig_height = 3.7 * array.shape[0] * px
    fig_width = 4.6 * array.shape[1] * px
    fontsize = max([fig_height, 6.0])
    plt.rcParams.update({'font.size': fontsize})

    fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
    plt.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()
    return fig


def have_same_shape(array1, array2):
    """
    Check if two arrays have the same shape - return True if so, false if not.
    """
    return np.allclose(np.array(array1.shape), np.array(array2.shape))


def make_flat_image(shape, fname):
    arr = np.zeros(shape) + 50
    im = Image.fromarray(arr)
    im = im.convert("L")
    im.save(fname)

def get_center(img, diam):
    l = np.shape(img)[0]
    overlap = l - diam
    start = int(overlap // 2)
    end = int(overlap // 2 + diam)
    return img[start:end, start:end]

def new_ang_spec_multi_prop(Uin, wvl, delta1, deltan, z, exp_jt, startpoint=False, endpoint=False):
    """
    Propagate field through a series of phase screens
    Args:
        Uin (ndarray): Signal to propagate through space
        wvl (float): Wavelength of the light propagating through space [m]
        delta1 (float): Object plane grid spacing [m]
        deltan (float): Pupil plane grid spacing [m]
        z (ndarray): 1-D array containing the locations of each propagation plane [m]
        exp_jt (ndarray): (nplanes,Nprop,Nprop) Phase distortion screens in complex form, e.g. exp(1j*t)
        startpoint: True if phase screen is at first propagation plane location (z[0])
        endpoint: True if phase screen is at last propagation plane location z[-1]
    Returns:
        Uout (ndarray): Output signal from propagation
        xn (ndarray): x-coordinate grid [samples]
        yn (ndarray): y-coordinate grid [samples]
    """
    # add dimension if there's only one phase screen
    scrn_size = exp_jt.shape
    if exp_jt.ndim == 2:
        exp_jt = exp_jt[np.newaxis, :, :]


    if not startpoint:
        exp_jt = np.concatenate((np.ones((1,scrn_size[-2],scrn_size[-1])), exp_jt), axis=0)
    if not endpoint:
        exp_jt = np.concatenate((exp_jt, np.ones((1,scrn_size[-2],scrn_size[-1]))), axis=0)

    N = Uin.shape[0]  # number of grid points
    x = np.arange(-N / 2, N / 2)
    nx, ny = np.meshgrid(x, x)
    k = 2 * np.pi / wvl  # optical wave vector
    # super-Gaussian absorbing boundary
    # nsq = nx**2 + ny**2
    # w = .49*N
    # sg = np.exp(np.power(-nsq,8)/w**16)
    sg = np.ones((N, N))

    n = len(z)
    Delta_z = z[1:n] - z[0:n - 1]  # propagation distances
    alpha = z / z[-1]
    delta = (1 - alpha) * delta1 + alpha * deltan
    m = delta[1:n] / delta[0:n - 1]
    x1 = nx * delta[0]
    y1 = ny * delta[0]
    r1sq = x1 ** 2 + y1 ** 2
    Q1 = np.exp(1j * k / 2 * (1 - m[0]) / Delta_z[0] * r1sq)


    Uin = Uin * Q1 * exp_jt[0]

    for idx in range(n - 1):
        # spatial frequencies (of i^th plane)
        deltaf = 1 / (N * delta[idx])
        fX = nx * deltaf
        fY = ny * deltaf
        fsq = fX ** 2 + fY ** 2
        Z = Delta_z[idx]  # propagation distance
        Q2 = np.exp(-1j * np.pi ** 2 * 2 * Z / m[idx] / k * fsq)  # quadratic phase factor

        # compute the propagated field
        Uin = sg * exp_jt[idx + 1] * ift2(Q2 * ft2(Uin / m[idx], delta[idx] ** 2), (N * deltaf) ** 2)

    # observation-plane coordinates
    xn = nx * delta[-1]
    yn = ny * delta[-1]
    rnsq = xn ** 2 + yn ** 2
    Q3 = np.exp(1j * k / 2 * (m[-1] - 1) / (m[-1] * Z) * rnsq)
    Uout = Q3 * Uin

    return Uout, xn, yn
