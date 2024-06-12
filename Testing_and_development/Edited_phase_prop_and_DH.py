import matplotlib.pyplot as plt
import numpy as np
from dh.dhsim import ft_phase_screen, create_reference, detect
from dh.dhdefs import ift2, circ, cmagsq, ft2, lens_phz
from skimage.restoration import unwrap_phase

from Testing_and_development.demo_utils import new_ang_spec_multi_prop

# pixel diameter of the object. Determines resolution

resolution = 1000  # Image resolution pixel/cms (should be no more than 1000)

# Some resolution math: 1/resolution (cm/pixel) * 1000 (microns/cms) = 10000/resolution (microns/pixel) A
# reasonable CCD camera pixel size is 10 microns across. Thus, so long as resolution < 1000 we can easily assume that
# our camera resolution will be better than the simulation grid resolution.


diam = 2 * resolution  # number of pixels in diameter of beam (forced to be 2 cms)

# circle of 10s
xx, yy = np.mgrid[:diam, :diam]
Img = np.array(circ(xx - diam / 2, yy - diam / 2, diam)) * 10

# AFRL sample image
#Img=load_img(None, diam)+1


# Arrays of 100s
# Img=np.ones((diam,diam))*100

# Amount of grid zero padding (not necessary w/o DH)
#padval = diam  # should be at least size of diam

#k = 2*np.pi / wvl
#Dap = Ln        # size of receiving aperture

# real image
#edit
#Img = np.pad(Img, padval)


# make mask for unwrapper
image_mask = (Img == 0)

# determining total image size
N, _ = np.shape(Img)

convertmc2mt = 1e-6  # multiply by to convert microns to meters
convertcm2mt = 0.01

# Field of view of our image in microns
#L = N * 20000 / diam  # Ensuring that centered image is 20 mm in diameter to match the paper
L = 20000 # 0meters
# 20000 microns= 20 mm = 2 centimeters = .02 meters


# dx = L / N  # cm per grid point
# x = np.linspace(-L / 2, L / 2, num=N, endpoint=False)  # np.arange(-L / 2, L / 2, dx)
# fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), num=N, endpoint=False)  # np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / L)

#[FX, FY] = np.meshgrid(fx, fx)

# wavelength (microns)
lam = 0.78

# distance of propagation in microns (26 cm to match article roughly)
z = 260000 * 1

Nmod = diam       # model size
Nimg = Nmod     # final DH image size
Nprop = Nmod    # propagation size
Ndet = 2*Nmod   # detector size (add as user input?)
padsize_prop = (Nprop-Nmod)//2  # pad input for propagation
padsize_det = (Ndet-Nimg)//2    # pad input for detection

Ln = lam*z*N/L    # observation plane grid length [m]
d1 = L/Nprop   # object-plane grid spacing [m]
#d1b = L1/Nimg   # reduced resolution grid spacing in object plane [m]
dn = Ln/Nprop   # detector-plane grid spacing [m]
Dap = Ln
### observation plane Specifications ###
# full resolution coordinate system (source plane)
x1 = np.arange(-Nprop/2,Nprop/2)*d1
x1,y1 = np.meshgrid(x1,x1)
# detector coordinate system
xn = np.arange(-Nprop/2,Nprop/2)*dn
[xn,yn] = np.meshgrid(xn,xn)
rn = np.sqrt(xn**2 + yn**2)

welldepth = 5.0e4  # well depth of the detector
bits = 12  # no. of bits for the A/D
sigma_rn = 40.0  # standard dev of read noise

ap = circ(xn,yn,Dap)    # aperture (circlular mask)
ap_b = ap               # low resolution aperture
Rshift = Ndet/2-Nimg/2  # number of samples to shift object spectrum in the frequency domain
RefBeam = create_reference(N,welldepth,Rshift) # genereates a reference field with a phase tilt
# L_obs = lam*z/dx    # observation plane grid length [m] Can be derived from page 120 eq 7.21
# dx_obs= L_obs

# Ln = lam*z*diam/0.02
# dn=Ln/N
# xn = np.arange(-diam*dx / 2, diam*dx / 2)
# [xn, yn] = np.meshgrid(xn, xn)
# # Ln=dn*diam*1.5
# padsize_det = diam
#
# ap = circ(xn,yn,Ln)
#
x0 = np.linspace(-diam / 2, diam / 2, num=diam, endpoint=False)
[x0, y0] = np.meshgrid(x0, x0)



# wavenumber
# k = 2 * np.pi / lam
# kz = np.sqrt(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2)

# -- Angular spectrum propagation with num_phz_screen phase screen -- #
np.random.seed(2)
# Make Phase screens (ro is 7000 microns (7 mm) to match article)
A = ft_phase_screen(r0=7000 * convertmc2mt, N=N, delta=d1* convertmc2mt)

Img = np.pad(Img,padsize_prop)
# propagate with each phase separately
imgA, _, _ = new_ang_spec_multi_prop(Img, lam * convertmc2mt, d1 * convertmc2mt, dn * convertmc2mt,
                                     np.array([0, z * convertmc2mt / 4, z * convertmc2mt]), np.exp(1j * A))

imgA_col = np.pad(ap*lens_phz(imgA,lam* convertmc2mt,z* convertmc2mt,dn* convertmc2mt),padsize_det)  # collimate field and pad array to size of the FPA
imgA_f = (1/Ndet) * ift2(imgA_col)    # this and above act to apply a focusing lens to produce the image on the FPA

# -- Detect Phase with Holography --##
# imgA = ap*lens_phz(imgA,lam,z,dn)  # collimate field
# imgA = (1/N) * ift2(imgA)    # this and above act to apply a focusing lens to produce the image on the FPA


# !!!Might have to change this one. May require a lot of zero padding
Ndetect = N  # detector Grid size.


# !!!Likely problems with how diam is set here
HoloMask = circ(x0, y0, diam)  # aperture (circlular mask)

# !!!Might have to change this one. Not sure what an appropriate shift should be.
Rshift = Ndetect / 2 - diam / 2  # number of samples to shift object spectrum in the frequency domain

# !!!Not sure how units will be relevant here
# RefBeam = create_reference(Ndetect, welldepth, Rshift)  # generates a reference field with a phase tilt
scale_fac = (2 ** -bits) * welldepth / np.mean(np.sqrt(cmagsq(RefBeam)))  # compute scale factor for data

## !! probably going to have to consider these things
# Ep_col = np.pad(ap*lens_phz(Ep,wvl,Dz,dn),padsize_det)  # collimate field and pad array to size of the FPA
# Ef = (1/Ndet) * ift2(Ep_col)    # this and above act to apply a focusing lens to produce the image on the FPA

# !!!probably going to be issues with units here. Don't know if detect function assumes certain units

#detect A
A_raw_holo, noise_pwr = detect(cmagsq(imgA + RefBeam), welldepth, bits, sigma_rn)  # detect hologram
A_raw_holo = A_raw_holo * scale_fac  # scale digitized hologram data to represent photoelectrons
A_fft_holo = (1 / Ndetect) * ft2(A_raw_holo)  # convert to freq domain
A_fft_holo_proc = ap_b * A_fft_holo[-diam:, -diam:]  # demodulate, low-pass filter, and decimate
# H_proc2 = HoloMask * lens_phz(H_proc,wvl,-Dz,dn)  # apply lense curvature to return to the entrance pupil plane
A_holo = (1 / Ndetect) * ift2(A_fft_holo_proc)  # convert back to image domain

# make mask for unwrapping hologram phase
holo_diam = len(A_fft_holo_proc) * diam / N
holo_mask = 1 - np.array(circ(xx - len(A_fft_holo_proc) / 2, yy - len(A_fft_holo_proc) / 2, holo_diam))

# collect phase from all images

# A hologram
wrapped_phzA = np.angle(A_holo)
unwrapped_phzA = unwrap_phase(np.ma.array(wrapped_phzA, mask=holo_mask))
unwrapped_phzA -= np.mean(unwrapped_phzA)

cmap = 'jet'


## plot the rest
plt.figure()
plt.imshow(unwrapped_phzA, cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Unwrapped phase w/ screen A @ 1/4 distance (cm)")
plt.colorbar()

plt.figure()
plt.imshow(np.ma.array(A, mask=image_mask), cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Screen A prescription (cm)")
plt.colorbar()

plt.figure()
plt.imshow(np.angle(imgA_f), cmap=cmap, extent=[0, L * 1e-4, 0, L * 1e-4])
plt.title("Screen A prescription (cm)")
plt.colorbar()

plt.show()
