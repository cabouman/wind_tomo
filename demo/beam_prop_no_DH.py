import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.restoration import unwrap_phase

import numpy as np
from dh.dhsim import ang_spec_multi_prop, create_reference, generate_screens, detect, load_img
from dh.dhdefs import lens_phz, ift2, circ, cmagsq, ft2, cnormdb, forward, normdb


opt_wavelength = 1.064e-6

prop_dist = 50  # Propogation distance in m
Num_screens = 0  # number of screens
screen_loc = np.array([0])  # location of each propagation plane 1->n
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
Ndet = 2*Nmod  # detector size (add as user input?)
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
I = load_img(None,Nmod)  # load the target image




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

Uin=fpad
wvl=opt_wavelength
delta1=d1
deltan=dn
z=screen_loc
exp_jt=np.exp(1j * t)
"""
Propagate field through a series of phase screens
Args:
    Uin (ndarray): Signal to propagate through space
    wvl (float): Wavelength of the light propagating through space [m]
    delta1 (float): Object plane grid spacing [m]
    deltan (float): Pupil plane grid spacing [m]
    z (ndarray): 1-D array containing the locations of each propagation plane [m]
    exp_jt (ndarray): (nplanes,Nprop,Nprop) Phase distortion screens in complex form, e.g. exp(1j*t)
Returns:
    Uout (ndarray): Output signal from propagation
    xn (ndarray): x-coordinate grid [samples]
    yn (ndarray): y-coordinate grid [samples]
"""
N = Uin.shape[0]    # number of grid points
x = np.arange(-N/2,N/2)
nx,ny = np.meshgrid(x,x)
k = 2*np.pi/wvl     # optical wave vector
# super-Gaussian absorbing boundary
#nsq = nx**2 + ny**2
#w = .49*N
#sg = np.exp(np.power(-nsq,8)/w**16)
sg = np.ones((N,N))

n = len(z)
Delta_z = z[1:n] - z[0:n-1] # propagation distances
alpha = z/z[-1]
delta = (1-alpha)*delta1 + alpha*deltan
m = delta[1:n] / delta[0:n-1]
x1 = nx * delta[0]
y1 = ny * delta[0]
r1sq = x1**2 + y1**2
Q1 = np.exp(1j*k/2*(1-m[0])/Delta_z[0]*r1sq)
Uin = Uin * Q1 * exp_jt[0]

# spatial frequencies (of i^th plane)
deltaf = 1 / (N*delta[idx])
fX = nx * deltaf
fY = ny * deltaf
fsq = fX**2 + fY**2
Z = Delta_z[idx]    # propagation distance
Q2 = np.exp(-1j*np.pi**2 * 2*Z / m[idx] / k * fsq)  # quadratic phase factor

# compute the propagated field
Uin = sg * exp_jt[idx+1] * ift2(Q2*ft2(Uin/m[idx],delta[idx]**2),(N*deltaf)**2)

# observation-plane coordinates
xn = nx * delta[-1]
yn = ny * delta[-1]
rnsq = xn**2 + yn**2
Q3 = np.exp(1j*k/2*(m[-1]-1)/(m[-1]*Z)*rnsq)
Uout = Q3 * Uin


Ep = ap * En
#attempt to unwrap phase before DH
plt.figure()
plt.imshow(I)
plt.title('original')


plt.figure()
plt.imshow(np.real(Ep))
cmap='jet'
plt.figure()
plt.imshow(unwrap_phase(np.angle(En)), cmap=cmap);
plt.title('phase')

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
plt.figure()
vmin = np.floor(np.min(t))
vmax = np.ceil(np.max(t))
vmin, vmax = None, None
cmap = 'jet'
# cmap='viridis'
for tt in range(1, t.shape[0]):
    alpha = screen_loc[tt] / screen_loc[-1]
    Delta = (1.0 - alpha) * d1 + alpha * dn
    xmax = Nprop / 2 * Delta
    extent = [-xmax, xmax, -xmax, xmax]
    if t.shape[0] > 2:
        plt.subplot(1 + (t.shape[0] - 2) // 4, min([4, t.shape[0] - 1]), tt)  # subplot w/ 4 columns max
    plt.imshow(t[tt], vmin=vmin, vmax=vmax, cmap=cmap);
    plt.colorbar(label='[rad]')
    plt.xlabel('[m]');
    plt.title('phase screen {}, z={:.1f}km'.format(tt, screen_loc[tt] * 1e-3))

print("close figs to continue")
plt.show()
