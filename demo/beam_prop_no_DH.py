import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.restoration import unwrap_phase

import numpy as np
from dh.dhsim import ang_spec_multi_prop, create_reference, generate_screens, detect
from dh.dhdefs import lens_phz, ift2, circ, cmagsq, ft2, cnormdb, forward, normdb


opt_wavelength = 1.064e-6

prop_dist = .4  # Propogation distance in meters
Num_screens = 1  # number of screens
screen_loc = np.linspace(0,prop_dist,Num_screens+1)  # location of each propagation plane 1->n
object_grid_len = 5
theta0_til = 10.
D_over_r0 = 10.

# set some fixed simulation parameters
welldepth = 5.0e4  # well depth of the detector
bits = 12  # no. of bits for the A/D
sigma_rn = 100  # standard dev of read noise
snr0 = 100

# set grid parameters
Nmod = 256  # model size
Nimg = Nmod  # final DH image size
Nprop = Nmod  # propagation size
Ndet = 2 * Nmod  # detector size (add as user input?)
padsize_prop = (Nprop - Nmod) // 2  # pad input for propagation
padsize_det = (Ndet - Nimg) // 2  # pad input for detection
# qfac = Ndet/Nimg    # q factor

Ln = opt_wavelength * prop_dist * Nmod / object_grid_len  # observation plane grid length [m]
d1 = object_grid_len / Nprop  # object-plane grid spacing [m]
# d1b = L1/Nimg   # reduced resolution grid spacing in object plane [m]
dn = Ln / Nprop  # detector-plane grid spacing [m]
# k = 2*np.pi / wvl
Dap = Ln  # size of receiving aperture

# ---------- Generate coordinate system ------------- #
# full resolution coordinate system (source plane)
x1 = np.arange(-Nprop / 2, Nprop / 2) * d1
x1, y1 = np.meshgrid(x1, x1)
# detector coordinate system
xn = np.arange(-Nprop / 2, Nprop / 2) * dn
[xn, yn] = np.meshgrid(xn, xn)
rn = np.sqrt(xn ** 2 + yn ** 2)
# reduced resolution coordinate system (source plane)
# xb = np.arange(-Nprop/2,Nprop/2)*d1b
# xb,yb = np.meshgrid(xb,xb)

# ---------- Generate aperture, reference, phase screen -----#
ap = circ(xn, yn, Dap)  # aperture (circlular mask)
ap_b = ap  # low resolution aperture
Rshift = Ndet / 2 - Nimg / 2  # number of samples to shift object spectrum in the frequency domain
R = create_reference(Ndet, welldepth, Rshift)  # generates a reference field with a phase tilt
I = np.zeros((256, 256))  # load the target image


#---Make custom screen and layer it a bunch --#

t, _ = generate_screens(D_over_r0, theta0_til, None, Nprop, d1, dn, prop_dist, screen_loc, screen_loc[1:], Dap,
                        opt_wavelength)

# t, _ = generate_screens(D_over_r0, theta0_til, None, Nprop, d1, dn, prop_dist, screen_loc, screen_loc[1:], Dap,
#                          opt_wavelength)
# for i in [2,3,4,5]:
#     t[i]=t[1]


N0 = I.shape[0]
f = I  # ignore quadratic phase in object plane
fpad = np.pad(f, padsize_prop)  # pad reflectance coefficient prior to propagation
# only the energy that makes its way back to us
# propagate
En, _, _ = ang_spec_multi_prop(fpad, opt_wavelength, d1, dn, screen_loc, np.exp(1j * t))
Ep = ap * En
#attempt to unwrap phase before DH
cmap='jet'
plt.figure()
plt.imshow(unwrap_phase(np.angle(Ep)), cmap=cmap);
plt.title('')

#
# # detect hologram
# Ep_col = np.pad(ap * lens_phz(Ep, opt_wavelength, prop_dist, dn),
#                 padsize_det)  # collimate field and pad array to size of the FPA
# Ef = (1 / Ndet) * ift2(Ep_col)  # this and above act to apply a focusing lens to produce the image on the FPA
# h, _ = detect(cmagsq(Ef + R), welldepth, bits, sigma_rn)  # detect hologram
# scale_fac = (2 ** -bits) * welldepth / np.mean(np.sqrt(cmagsq(R)))  # compute scale factor for data
# h = h * scale_fac  # scale digitized hologram data to represent photoelectrons
# H = (1 / Ndet) * ft2(h)  # convert to freq domain
# H_proc = ap_b * H[-Nimg:, -Nimg:]  # demodulate, low-pass filter, and decimate
# H_proc2 = ap_b * lens_phz(H_proc, opt_wavelength, -prop_dist,
#                           dn)  # apply lense curvature to return to the entrance pupil plane
# h_proc = (1 / Nimg)*ift2(H_proc)  # convert back to image domain
#
#
# # these are only used for plot output
# rhat2 = cmagsq(forward(1, H_proc2, ap, opt_wavelength, prop_dist, d1, Ln, screen_loc, t, 1, 1))  # phase corrected DH image
# H[Ndet // 2 - 5:Ndet // 2 + 6,
# Ndet // 2 - 5:Ndet // 2 + 6] = 0  # high-pass filter to remove dc term (for ploting purposes)
#
# # plot various intermediate outputs
#
# vmin = -30
# vmax = 0
# cmap = 'jet'
# # plt.figure()
# # plt.imshow(h, cmap=cmap);
# # plt.title('Hologram')
# #
# # plt.figure()
# # plt.imshow(cnormdb(H), vmin=vmin, vmax=vmax, cmap=cmap);
# # plt.title('Hologram Spectrum')
# #
# # plt.figure()
# # plt.imshow(cnormdb(H_proc), vmin=vmin, vmax=vmax, cmap=cmap);
# # plt.title('Pupil Img')
#
# # plt.figure()
# # plt.imshow(cnormdb(h_proc), vmin=vmin, vmax=vmax, cmap=cmap);
# # plt.title('Blurry DH Image')
#
#
# plt.figure()
# plt.imshow(unwrap_phase(np.angle(h_proc)), cmap=cmap);
# plt.title('Blurry DH Image')
# plt.colorbar(label='[rad]')
#
# # plt.figure()
# # plt.imshow(np.imag(h_proc), cmap=cmap);
# # plt.title('Blurry DH Image2')
#
# plt.figure()
# plt.figure()
# vmin = np.floor(np.min(t))
# vmax = np.ceil(np.max(t))
# vmin, vmax = None, None
# cmap = 'jet'
# # cmap='viridis'
# for tt in range(1, t.shape[0]):
#     alpha = screen_loc[tt] / screen_loc[-1]
#     Delta = (1.0 - alpha) * d1 + alpha * dn
#     xmax = Nprop / 2 * Delta
#     extent = [-xmax, xmax, -xmax, xmax]
#     if t.shape[0] > 2:
#         plt.subplot(1 + (t.shape[0] - 2) // 4, min([4, t.shape[0] - 1]), tt)  # subplot w/ 4 columns max
#     plt.imshow(t[tt], vmin=vmin, vmax=vmax, cmap=cmap);
#     plt.colorbar(label='[rad]')
#     plt.xlabel('[m]');
#     plt.title('phase screen {}, z={:.1f}km'.format(tt, screen_loc[tt] * 1e-3))
#
# print("close figs to continue")
plt.show()
