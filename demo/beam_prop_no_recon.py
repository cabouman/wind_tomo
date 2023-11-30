import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.restoration import unwrap_phase

import numpy as np
from dh.dhsim import ang_spec_multi_prop, create_reference, generate_screens, detect
from dh.dhdefs import lens_phz, ift2, circ, cmagsq, ft2, cnormdb, forward, normdb


#settings and stuff I don't really understand
Num_screens=1
opt_wavelength = 1.064e-6
prop_dist = 10  # Propagation distance in meters
screen_loc = np.linspace(0,prop_dist,Num_screens+1)

welldepth = 5.0e4  # well depth of the detector
object_grid_len = 5 # don't know what this means
D_over_r0 = 10
theta0_til = 10
Nmod = 256  # model size
Nimg = Nmod  # final DH image size
Nprop = Nmod  # propagation size
Ndet = 2 * Nmod  # detector size (add as user input?)
Ln = opt_wavelength * prop_dist * Nmod / object_grid_len  # observation plane grid length [m]
d1 = object_grid_len / Nprop  # object-plane grid spacing [m]
dn = Ln / Nprop  # detector-plane grid spacing [m]
Dap = Ln  # size of receiving aperture
padsize_prop = (Nprop - Nmod) // 2  # pad input for propagation
padsize_det = (Ndet - Nimg) // 2  # pad input for detection
sigma_rn = 1  # standard dev of read noise
bits = 12  # no. of bits for the A/D


# # detector coordinate system
xn = np.arange(-Nprop / 2, Nprop / 2) * dn
[xn, yn] = np.meshgrid(xn, xn)

# set initial image
I = np.zeros((256, 256))

#determine apeture
ap = circ(xn, yn, Dap)

# Make phase screens
t, _ = generate_screens(D_over_r0, theta0_til, None, Nprop, d1, dn, prop_dist, screen_loc, screen_loc[1:], Dap, opt_wavelength)

#propagate beam
f = I  # ignore quadratic phase in object plane
fpad = np.pad(f, padsize_prop)  # pad reflectance coefficient prior to propagation
En, _, _ = ang_spec_multi_prop(fpad, opt_wavelength, d1, dn, screen_loc, np.exp(1j * t))
Ep = ap * En

cmap='jet'


# reference beam
Rshift = Ndet / 2 - Nimg / 2  # number of samples to shift object spectrum in the frequency domain
R = create_reference(Ndet, welldepth, Rshift)  # generates a reference field with a phase tilt


#get hologram
Ep_col = np.pad(ap * lens_phz(Ep, opt_wavelength, prop_dist, dn),
                padsize_det)  # collimate field and pad array to size of the FPA
Ep=np.pad(Ep,padsize_det)
Ef = (1 / Ndet) * ift2(Ep_col)  # this and above act to apply a focusing lens to produce the image on the FPA

plt.figure()
plt.imshow(cmagsq(Ef))
plt.title('True phase')

h, _ = detect(cmagsq(Ef + R), welldepth, bits, sigma_rn)  # detect hologram


# Convert to frequency
H = (1 / Ndet) * ft2(h)

#window out region we don't want
H_proc = ap * H[-Nimg:, -Nimg:]

#Convert back to image domain
h_proc = (1 / Nimg)*ift2(H_proc)


plt.figure()
plt.imshow(unwrap_phase(np.angle(h_proc)), cmap=cmap)
plt.title('DH recovered phase')
plt.colorbar(label='[rad]')

plt.figure()
for tt in range(1, t.shape[0]):
    alpha = screen_loc[tt] / screen_loc[-1]
    Delta = (1.0 - alpha) * d1 + alpha * dn
    xmax = Nprop / 2 * Delta
    extent = [-xmax, xmax, -xmax, xmax]
    if t.shape[0] > 2:
        plt.subplot(1 + (t.shape[0] - 2) // 4, min([4, t.shape[0] - 1]), tt)  # subplot w/ 4 columns max
    plt.imshow(t[tt], cmap=cmap);
    plt.colorbar(label='[rad]')
    plt.xlabel('[m]');
    plt.title('phase screen {}, z={:.1f}km'.format(tt, screen_loc[tt] * 1e-3))

plt.show()