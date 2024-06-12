import matplotlib.pyplot as plt
import numpy as np
from dh.dhsim import ft_phase_screen
from dh.dhdefs import ift2, circ, cmagsq, ft2
from skimage.restoration import unwrap_phase

from Testing_and_development.demo_utils import new_ang_spec_multi_prop

# This demo simulates a beam propagation through a phase screen using angular spectrum propagation

# determine where phase screen is applied
screen_loc = 0  # key 0=start, 1=middle, 2=end

# determine if you want to include results with no phase screen
Test_no_phase_screen = False

key = ['start', 'middle', 'end']

convert = 1e-6  # multiply by to convert microns to meters
# pixel Grid diameter image. Affects resolution
diam = 500

# circle of 100s
xx, yy = np.mgrid[:diam, :diam]
Img = np.array(circ(xx - diam / 2, yy - diam / 2, diam)) * 10

# AFRL sample image
# Img=load_img(None, diam)+1


# Arrays of 100s
# Img=np.ones((diam,diam))*100

# Amount of grid zero padding (not necessary w/o DH)
padval = 0

# make mask for unwrapper
image_mask = (np.pad(Img, padval) == 0)

# real image
Img = np.pad(Img, padval)

# determining total image size
N, _ = np.shape(Img)
print(N)

# Field of view of our image in microns
L = N * 20000 / diam  # Ensuring that centered image is 20 mm in diameter to match the paper

# 20000 microns= 20 mm = 2 centimeters


cmap = 'jet'

plt.figure()
plt.imshow(abs(Img), cmap='gray', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.colorbar()
plt.title("cmag before propagation (cm)")
#

# get axis
dx = L / N  # microns per grid point
x = np.linspace(-L / 2, L / 2, num=N, endpoint=False)  # np.arange(-L / 2, L / 2, dx)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), num=N, endpoint=False)  # np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / L)

[FX, FY] = np.meshgrid(fx, fx)

# wavelength (microns)
lam = 0.78

# distance of propagation in microns (26 cm to match article roughly)
z = 260000 * 1

# wavenumber
k = 2 * np.pi / lam
kz = np.sqrt(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2)

if Test_no_phase_screen == True:
    # -- angular spectrum propogation with no phase screen -- ##
    H = np.exp(1j * z * kz)
    Img_prop = ift2(ft2(Img) * H)

    # Plotting results
    wrapped_phz = np.angle(Img_prop)

    plt.figure()
    plt.imshow(unwrap_phase(wrapped_phz), cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
    plt.title("unwrapped phase of propagated beam no screen (cm)")
    plt.colorbar()

    plt.figure()
    plt.imshow(abs((Img_prop)), cmap='gray', extent=[0, L * 1e-4, 0, L * 1e-4])
    plt.title("cmag of propagated beam no screen (cm)")
    plt.colorbar()

    # -- DH angular spectrum propogation with no phase screen -- ##
    # inputting 3 at end of function here means it will never apply a phase screen
    DHimg_prop, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                               np.array([0, z * convert]), np.ones((N,N)))

    # Plotting results
    wrapped_phz = np.angle(DHimg_prop)

    plt.figure()
    plt.imshow(unwrap_phase(wrapped_phz) - (unwrap_phase(wrapped_phz)), cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
    plt.title("DH:unwrapped phase of propagated beam no screen (cm)")
    plt.colorbar()

    plt.figure()
    plt.imshow(abs((DHimg_prop)), cmap='gray', extent=[0, L * 1e-4, 0, L * 1e-4])
    plt.title("DH: cmag of propagated beam no screen (cm)")
    plt.colorbar()

# Now repeat with phase Screens
# Note, I believe my method is ignoring "Quadratic phase factor" among a few other parameters
np.random.seed(2)

# -- Angular spectrum propagation with one phase screen -- #
# Make Phase screen (ro is 7000 microns (7 mm) to match article)
phz_screen = ft_phase_screen(r0=7000 * convert, N=N, delta=dx * convert)

if screen_loc == 0:
    # --Apply phase screen at start --#
    H = np.exp(1j * (z) * kz)
    Img_prop2 = ift2(ft2(np.exp(1j * phz_screen) * Img) * H)

if screen_loc == 1:
    # --Apply Phase screen at middle --#
    # perform angular spectrum propagation to half total distance and apply phase screen.
    H = np.exp(1j * (z / 2) * kz)
    Img_prop2 = np.exp(1j * phz_screen) * ift2(ft2(Img) * H)

    # perform angular spectrum propagation to remaining distance
    Img_prop2 = ift2(ft2(Img_prop2) * H)

if screen_loc == 2:
    # --Apply phase screen at end --#
    H = np.exp(1j * (z) * kz)
    Img_prop2 = np.exp(1j * phz_screen) * ift2(ft2(Img) * H)

# Plotting results
wrapped_phz = np.angle(Img_prop2)
unwrapped_phz=unwrap_phase(np.ma.array(wrapped_phz, mask=image_mask))
unwrapped_phz-=np.mean(unwrapped_phz)

plt.figure()
plt.imshow(cmagsq(Img_prop2), cmap='gray', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("cmag of propagated beam w/ screen @ " + key[screen_loc] + " (cm)")
plt.colorbar()

plt.figure()
plt.imshow(wrapped_phz, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("wrapped phase of propagated beam w/ screen @ " + key[screen_loc] + " (cm)")
plt.colorbar()

plt.figure()
plt.imshow(unwrapped_phz, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("unwrapped phase of propagated beam w/ screen @ " + key[screen_loc] + " (cm)")
plt.colorbar()

thing1 = lam * z / L

# uses meters for convention
if screen_loc == 0:
    DHimg, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                          np.array([0, z * convert]), np.exp(1j * phz_screen), startpoint=True)
if screen_loc == 1:
    DHimg, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                          np.array([0, z * convert / 2, z * convert]), np.exp(1j * phz_screen))
if screen_loc == 2:
    DHimg, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                          np.array([0, z * convert]), np.exp(1j * phz_screen), endpoint=True)

# Plotting results
DHwrapped_phz = np.angle(DHimg)
DHunwrapped_phz=unwrap_phase(np.ma.array(wrapped_phz, mask=image_mask))
DHunwrapped_phz-=np.mean(DHunwrapped_phz)

plt.figure()
plt.imshow(cmagsq(DHimg), cmap='gray', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("DH:cmag of propagated beam w/ screen @ " + key[screen_loc] + " (cm)")
plt.colorbar()

plt.figure()
plt.imshow(DHwrapped_phz, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("DH:wrapped phase of propagated beam w/ screen @ " + key[screen_loc] + " (cm)")
plt.colorbar()

plt.figure()
plt.imshow(DHunwrapped_phz, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("DH:unwrapped phase of propagated beam w/ screen @ " + key[screen_loc] + " (cm)")
plt.colorbar()

plt.figure()
plt.imshow((DHunwrapped_phz - unwrapped_phz), cmap='gray', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("DH code phase - my code phase")
plt.colorbar()

plt.figure()
plt.imshow(np.real(phz_screen), cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("phase screen (cm)")
plt.colorbar()



plt.show()
