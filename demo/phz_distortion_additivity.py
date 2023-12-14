import dh
import matplotlib.pyplot as plt
import numpy as np
from dh.dhsim import load_img, ft_phase_screen, ang_spec_multi_prop
from dh.dhdefs import ift2, circ, cmagsq, ft2, cnormdb, forward, normdb
from skimage.restoration import unwrap_phase

from demo.demo_utils import new_ang_spec_multi_prop

# pixel diameter of the object. Determines resolution
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

convert = 1e-6  # multiply by to convert microns to meters

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

# -- Angular spectrum propagation with num_phz_screen phase screen -- #
np.random.seed(2)
# Make Phase screens (ro is 7000 microns (7 mm) to match article)
A = ft_phase_screen(r0=7000 * convert, N=N, delta=dx * convert)

B = ft_phase_screen(r0=7000 * convert, N=N, delta=dx * convert)

C = ft_phase_screen(r0=7000 * convert, N=N, delta=dx * convert)

# all phase screens together
A_B_C = np.concatenate((A[np.newaxis,:,:], B[np.newaxis, :,:], C[np.newaxis, :,:]),axis=0)

# propagate with each phase separately
imgA, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                     np.array([0, z * convert / 4, z * convert]), np.exp(1j * A))

imgB, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                     np.array([0, z * convert / 2, z * convert]), np.exp(1j * B))

imgC, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                     np.array([0, z * convert * 3 / 4, z * convert]), np.exp(1j * C))

# propagated with all plates present
imgABC, _, _ = new_ang_spec_multi_prop(Img, lam * convert, dx * convert, dx * convert,
                                       np.array(
                                           [0, z * convert / 4, z * convert / 2, z * convert * 3 / 4, z * convert]),
                                       np.exp(1j * A_B_C))

# collect phase from all images
wrapped_phzA = np.angle(imgA)
unwrapped_phzA = unwrap_phase(np.ma.array(wrapped_phzA, mask=image_mask))
unwrapped_phzA -= np.mean(unwrapped_phzA)

wrapped_phzB = np.angle(imgB)
unwrapped_phzB = unwrap_phase(np.ma.array(wrapped_phzB, mask=image_mask))
unwrapped_phzB -= np.mean(unwrapped_phzB)

wrapped_phzC = np.angle(imgC)
unwrapped_phzC = unwrap_phase(np.ma.array(wrapped_phzC, mask=image_mask))
unwrapped_phzC -= np.mean(unwrapped_phzC)

wrapped_phzABC = np.angle(imgABC)
unwrapped_phzABC = unwrap_phase(np.ma.array(wrapped_phzABC, mask=image_mask))
unwrapped_phzABC -= np.mean(unwrapped_phzABC)



cmap = 'jet'

plt.figure()
plt.imshow(unwrapped_phzA, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Unwrapped phase w/ screen A @ 1/4 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(A, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Screen A prescription (cm)")
plt.colorbar()

plt.figure()
plt.imshow(unwrapped_phzB, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Unwrapped phase w/ screen B @ 2/4 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(B, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Screen B prescription (cm)")
plt.colorbar()


plt.figure()
plt.imshow(unwrapped_phzC, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Unwrapped phase w/ screen C @ 3/4 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(C, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Screen C prescription (cm)")
plt.colorbar()

plt.figure()
plt.imshow(unwrapped_phzA + unwrapped_phzB + unwrapped_phzC, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Sum of unwrapped phases (cm)")
plt.colorbar()

plt.figure()
plt.imshow(unwrapped_phzABC, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Unwrapped phase w/ all screens present (cm)")
plt.colorbar()

plt.figure()
plt.imshow(unwrapped_phzABC-(unwrapped_phzA + unwrapped_phzB + unwrapped_phzC), cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("all screens - sum of individual (cm)")
plt.colorbar()

plt.figure()
plt.imshow(A+B+C, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Sum of all screen prescriptions (cm)")
plt.colorbar()

plt.show()


