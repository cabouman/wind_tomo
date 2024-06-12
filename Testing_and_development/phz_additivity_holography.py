import matplotlib.pyplot as plt
import numpy as np
from dh.dhsim import ft_phase_screen, create_reference, detect
from dh.dhdefs import ift2, circ, cmagsq, ft2
from skimage.restoration import unwrap_phase

from Testing_and_development.demo_utils import new_ang_spec_multi_prop, get_center


# pixel diameter of the object. Determines resolution


resolution = 500  # Image resolution pixel/cms (should be no more than 1000)

# Some resolution math: 1/resolution (cm/pixel) * 1000 (microns/cms) = 10000/resolution (microns/pixel) A
# reasonable CCD camera pixel size is 10 microns across. Thus, so long as resolution < 1000 we can easily assume that
# our camera resolution will be better than the simulation grid resolution.


diam = 2 * resolution  # number of pixels in diameter of beam (forced to be 2 cms)

# circle of 10s
xx, yy = np.mgrid[:diam, :diam]
Img = np.array(circ(xx - diam / 2, yy - diam / 2, diam)) * 10

# AFRL sample image
# Img=load_img(None, diam)+1


# Arrays of 100s
# Img=np.ones((diam,diam))*100

# Amount of grid zero padding (not necessary w/o DH)
padval = diam  # should be at least size of diam

# k = 2*np.pi / wvl
# Dap = Ln        # size of receiving aperture

# real image
Img = np.pad(Img, padval)

# make mask for unwrapper
image_mask = (Img == 0)

# determining total image size
N, _ = np.shape(Img)

print(N)

# Field of view of our image in microns
L = N * 20000 / diam  # Ensuring that centered image is 20 mm in diameter to match the paper
# L = .02 # 0meters
# 20000 microns= 20 mm = 2 centimeters = .02 meters

convert = 1e-6  # multiply by to convert microns to meters

dx = L / N  # meters per grid point
x = np.linspace(-L / 2, L / 2, num=N, endpoint=False)  # np.arange(-L / 2, L / 2, dx)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), num=N, endpoint=False)  # np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / L)

[FX, FY] = np.meshgrid(fx, fx)

# wavelength (microns)
lam = 0.78

# distance of propagation in microns (26 cm to match article roughly)
z = 260000 * 2

### observation plane Specifications ###

x0 = np.linspace(-diam / 2, diam / 2, num=diam, endpoint=False)
[x0, y0] = np.meshgrid(x0, x0)

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
A_B_C = np.concatenate((
    A[np.newaxis, :, :],
    B[np.newaxis, :, :],
    C[np.newaxis, :, :]), axis=0)

print("before propagation " + str(len(Img)))
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
                                           [0,
                                            z * convert / 4,
                                            z * convert / 2,
                                            z * convert * 3 / 4,
                                            z * convert]),
                                       np.exp(1j * A_B_C))

# -- Detect Phase with Holography --##
# sensor specification
welldepth = 5.0e4  # well depth of the detector
bits = 12  # no. of bits for the A/D
sigma_rn = 40.0  # standard dev of read noise

# !!!Might have to change this one. May require a lot of zero padding
Ndetect = N  # detector Grid size.

# !!!Likely problems with how diam is set here
HoloMask = circ(x0, y0, diam)  # aperture (circlular mask)

# !!!Might have to change this one. Not sure what an appropriate shift should be.
Rshift = Ndetect / 2 - diam / 2  # number of samples to shift object spectrum in the frequency domain

# !!!Not sure how units will be relevant here
RefBeam = create_reference(Ndetect, welldepth, Rshift)  # generates a reference field with a phase tilt
scale_fac = (2 ** -bits) * welldepth / np.mean(np.sqrt(cmagsq(RefBeam)))  # compute scale factor for data

## !! probably going to have to consider these things
# Ep_col = np.pad(ap*lens_phz(Ep,wvl,Dz,dn),padsize_det)  # collimate field and pad array to size of the FPA
# Ef = (1/Ndet) * ift2(Ep_col)    # this and above act to apply a focusing lens to produce the image on the FPA

# !!!probably going to be issues with units here. Don't know if detect function assumes certain units

# detect A
A_raw_holo, noise_pwr = detect(cmagsq(imgA + RefBeam), welldepth, bits, sigma_rn)  # detect hologram
A_raw_holo = A_raw_holo * scale_fac  # scale digitized hologram data to represent photoelectrons
A_fft_holo = (1 / Ndetect) * ft2(A_raw_holo)  # convert to freq domain
A_fft_holo_proc = HoloMask * A_fft_holo[-diam:, -diam:]  # demodulate, low-pass filter, and decimate
# H_proc2 = HoloMask * lens_phz(H_proc,wvl,-Dz,dn)  # apply lense curvature to return to the entrance pupil plane
A_holo = (1 / Ndetect) * ift2(A_fft_holo_proc)  # convert back to image domain


# detect B
B_raw_holo, noise_pwr = detect(cmagsq(imgB + RefBeam), welldepth, bits, sigma_rn)  # detect hologram
B_raw_holo = B_raw_holo * scale_fac  # scale digitized hologram data to represent photoelectrons
B_fft_holo = (1 / Ndetect) * ft2(B_raw_holo)  # convert to freq domain
B_fft_holo_proc = HoloMask * B_fft_holo[-diam:, -diam:]  # demodulate, low-pass filter, and decimate
# H_proc2 = HoloMask * lens_phz(H_proc,wvl,-Dz,dn)  # apply lense curvature to return to the entrance pupil plane
B_holo = (1 / Ndetect) * ift2(B_fft_holo_proc)  # convert back to image domain

# detect C
C_raw_holo, noise_pwr = detect(cmagsq(imgC + RefBeam), welldepth, bits, sigma_rn)  # detect hologram
C_raw_holo = C_raw_holo * scale_fac  # scale digitized hologram data to represent photoelectrons
C_fft_holo = (1 / Ndetect) * ft2(C_raw_holo)  # convert to freq domain
C_fft_holo_proc = HoloMask * C_fft_holo[-diam:, -diam:]  # demodulate, low-pass filter, and decimate
# H_proc2 = HoloMask * lens_phz(H_proc,wvl,-Dz,dn)  # apply lense curvature to return to the entrance pupil plane
C_holo = (1 / Ndetect) * ift2(C_fft_holo_proc)  # convert back to image domain

# detect ABC
ABC_raw_holo, noise_pwr = detect(cmagsq(imgABC + RefBeam), welldepth, bits, sigma_rn)  # detect hologram
ABC_raw_holo = ABC_raw_holo * scale_fac  # scale digitized hologram data to represent photoelectrons
ABC_fft_holo = (1 / Ndetect) * ft2(ABC_raw_holo)  # convert to freq domain
ABC_fft_holo_proc = HoloMask * ABC_fft_holo[-diam:, -diam:]  # demodulate, low-pass filter, and decimate
# H_proc2 = HoloMask * lens_phz(H_proc,wvl,-Dz,dn)  # apply lense curvature to return to the entrance pupil plane
ABC_holo = (1 / Ndetect) * ift2(ABC_fft_holo_proc)  # convert back to image domain

# make mask for unwrapping hologram phase
holo_diam = len(A_fft_holo_proc) * diam / N
holo_mask = 1 - np.array(circ(xx - len(A_fft_holo_proc) / 2, yy - len(A_fft_holo_proc) / 2, holo_diam))

# collect phase from all images

# A hologram
wrapped_phzA = np.angle(A_holo)
unwrapped_phzA = unwrap_phase(np.ma.array(wrapped_phzA, mask=holo_mask))
unwrapped_phzA -= np.mean(unwrapped_phzA)

# B hologram
wrapped_phzB = np.angle(B_holo)
unwrapped_phzB = unwrap_phase(np.ma.array(wrapped_phzB, mask=holo_mask))
unwrapped_phzB -= np.mean(unwrapped_phzB)

# C hologram
wrapped_phzC = np.angle(C_holo)
unwrapped_phzC = unwrap_phase(np.ma.array(wrapped_phzC, mask=holo_mask))
unwrapped_phzC -= np.mean(unwrapped_phzC)

# ABC hologram
wrapped_phzABC = np.angle(ABC_holo)
unwrapped_phzABC = unwrap_phase(np.ma.array(wrapped_phzABC, mask=holo_mask))
unwrapped_phzABC -= np.mean(unwrapped_phzABC)

# ABC true phase for comparison
wrapped_true_phzABC = np.angle(imgABC)
unwrapped_true_phzABC = unwrap_phase(np.ma.array(wrapped_true_phzABC, mask=image_mask))
unwrapped_true_phzABC -= np.mean(unwrapped_true_phzABC)

cmap = 'jet'

## -- Holography Process Demonstrated with all phase plates present --##
plt.figure()
plt.subplot(3, 3, 1)
plt.imshow(Img, cmap='grey')
plt.title("original beam")

plt.subplot(3, 3, 2)
plt.imshow(np.log(cmagsq(RefBeam) - np.min(cmagsq(RefBeam)) + 1), cmap='grey', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("reference beam")

plt.subplot(3, 3, 3)
plt.imshow(np.log(ABC_raw_holo - np.min(ABC_raw_holo) + 1), cmap='grey', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("raw hologram")

plt.subplot(3, 3, 4)
plt.imshow(np.log(cmagsq(ABC_fft_holo) - np.min(cmagsq(ABC_fft_holo)) + 1), cmap='grey')
plt.title("FFT hologram")

plt.subplot(3, 3, 5)
plt.imshow(np.log(cmagsq(ABC_fft_holo[-diam:, -diam:]) - np.min(cmagsq(ABC_fft_holo[-diam:, -diam:])) + 1), cmap='grey')
plt.title("demodulated")

plt.subplot(3, 3, 6)
plt.imshow(np.log(cmagsq(ABC_fft_holo_proc) - np.min(cmagsq(ABC_fft_holo_proc)) + 1), cmap='grey')
plt.title("low pass masked")

plt.subplot(3, 3, 7)
plt.imshow(cmagsq(ABC_holo), cmap='grey', extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("replayed field cmag")

plt.subplot(3, 3, 8)
plt.imshow(unwrapped_phzABC, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("DH phase")
plt.colorbar()

plt.subplot(3, 3, 9)
plt.imshow(unwrapped_true_phzABC, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4 ])
plt.title("true phase")
plt.colorbar()
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle("Holography Sim: 3 Phz Screen (cm) (top 2 rows log scaled)")

## plot the rest
plt.figure()
plt.imshow(get_center(unwrapped_phzA, diam / 3), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Unwrapped phase w/ screen A @ 1/4 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(np.ma.array(A, mask=image_mask), diam), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Masked Screen A prescription (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(unwrapped_phzB, diam / 3), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Unwrapped phase w/ screen B @ 1/2 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(np.ma.array(B, mask=image_mask), diam), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Masked Screen B prescription (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(unwrapped_phzC, diam / 3), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Unwrapped phase w/ screen C @ 3/4 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(np.ma.array(C, mask=image_mask), diam), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Screen C prescription (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(
    unwrapped_phzA +
    unwrapped_phzB +
    unwrapped_phzC, diam / 3), cmap=cmap,
           extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Sum of unwrapped phases (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(unwrapped_phzABC, diam / 3), cmap=cmap, extent=[0, L * 1e-4 / 3, 0, L * 1e-4 / 3])
plt.title("Unwrapped phase w/ all screens present (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(unwrapped_phzABC - (
        unwrapped_phzA +
        unwrapped_phzB +
        unwrapped_phzC), diam / 3), cmap=cmap,
           extent=[0, L * 1e-4/ 3, 0, L * 1e-4/ 3])
plt.title("all screens - sum of individual (cm)")
plt.colorbar()

plt.figure()
plt.imshow(get_center(np.ma.array(
    A +
    B +
    C, mask=image_mask), diam), cmap=cmap, extent=[0, L * 1e-4/ 3, 0, L * 1e-4/ 3])
plt.title("Sum of all screen prescriptions (cm)")
plt.colorbar()


plt.show()
