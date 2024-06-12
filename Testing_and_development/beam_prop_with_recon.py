import dh
import matplotlib.pyplot as plt
import numpy as np
from dh.dhsim import ang_spec_multi_prop, create_reference, generate_screens, detect, load_img
from dh.dhdefs import lens_phz, ift2, circ, cmagsq, ft2, cnormdb, forward, normdb
from skimage.restoration import unwrap_phase

# make_flat_image((256,256), "flat.jpg")
# sim = dh.DH()
# sim.dhsim(N=256,target_fname='flat.jpg', Nscr=1, D_over_r0=10, seed=1, sim_plt_flg=True)

# # Set reconstruction parameters (update from defaults) and call recon()
# # Parameters can be set with both the set_recon_params() and recon()  methods.
# # The parameter settings persist in the class instance in 'recon_params'.
# sim.set_recon_params(bootstrap_it=10, NL1=20, NL2=200, plt_flg=True, nplt=100)
# # set hpdh_mode to 'hybrid' if reconstructing multiple phase planes
# sim.recon(hpdh_mode='fastest', unwrap=True)  # Reconstruct r,phz
#
# # Plot final estimated reflectance/phase
# print("Plotting final estimated reflectance/phase, Strehl")
# plt.figure(); plt.imshow(sim.r,cmap='gray'); plt.title("estimated reflectance")
# plt.figure(); plt.imshow(sim.t[0],cmap='jet'); plt.title("true phase"); plt.colorbar()
# plt.figure(); plt.imshow(sim.phz[0],cmap='jet'); plt.title("estimated phase"); plt.colorbar()
#
# # Plot strehl ratio and rmse vs. iteration number
# plt.figure(); plt.plot(sim.strehl); plt.title('Strehl ratio'); plt.xlabel('iteration')
# plt.figure(); plt.plot(sim.rmse); plt.title('RMSE'); plt.xlabel('iteration')
# print("Close figs to continue")
# plt.show()
sim = dh.DH()

plane='image'
N=256
Dz=5000
L1=5.0
wvl=1.064e-6
t=None
D_over_r0=10.
theta0_til=10.
rytov=None
Nscr=1
s=None
snr0=100
seed=1
sim_plt_flg=True
target_fname=None
"""
Simulate digital holography measurements.

Args:
    out (class DHdata, optional): Existing class instance for destination of simulation output
    target_fname (string, optional): filename for target image (png,jpg,tif)
    plane (str): 'pupil' produces data in pupil-plane (default), 'image' in image-plane
    N (int): [256] output image plane size (NxN)
    Dz (float): propagation distance [m]
    L1 (float): object grid length [m]
    wvl (float): optical wavelength [m]
    t (ndarray): phase screens input array. Use to bypass generating screen internally.
    D_over_r0 (float): turbulence strength, D/r_0 for spherical wave
    theta0_til (float): ratio of isoplanatic angle to diffraction limited angle
    rytov (float): desired Rytov number. NOTE: Either theta0_til or rytov can be set, but not both
    Nscr (int): number of phase screens along propagation path (if s is empty)
    s (list): desired locations in [m] of phase screens along propagation path
    snr0 (float): desired SNR of data
    seed (int): seed for random number generator
    sim_plt_flg (boolean): True/False display plots of phase screens etc.

Returns:
    class DHdata: simulated data and attributes
"""
if seed is not None:
    np.random.seed(int(seed))

# set some fixed simulation parameters
welldepth = 5.0e4   # well depth of the detector
bits = 12           # no. of bits for the A/D
sigma_rn = 40.0     # standard dev of read noise

# if given, parse parameters from phase screen locations
if s is not None:
    s = np.array(s)
    if s[0]==0:
        s = s[1:]   # no phase distortion at object plane, so clip this location if given
    Nscr = s.shape[0]
    Dz = s[-1]

# if given, parse parameters from phase screens
if isinstance(t,np.ndarray):
    if t.ndim == 2:
        t = t[np.newaxis,:,:]   # isoplanatic case
    if (s is not None) and (t.shape[0] != s.shape[0]):
        raise Exception("dhsim: dimensions mismatch t {}, s {}".format(t.shape,s.shape))
    Nscr = t.shape[0]
    N = t.shape[1]
    # internal convention has a phase=0 screen at the object plane, so add here
    t0 = np.zeros((1,t.shape[1],t.shape[2]))
    t = np.concatenate((t0,t),axis=0)

# set propagation plane and screen locations
if s is None:
    z = np.linspace(0,Dz,Nscr+1) # location of each propagation plane 1->n
    s = z[1:]
else:
    z = np.concatenate((np.array([0]),s))  # add propagation plane at object

# set grid parameters
Nmod = N        # model size
Nimg = Nmod     # final DH image size
Nprop = Nmod    # propagation size
Ndet = 2*Nmod   # detector size (add as user input?)
padsize_prop = (Nprop-Nmod)//2  # pad input for propagation
padsize_det = (Ndet-Nimg)//2    # pad input for detection
#qfac = Ndet/Nimg    # q factor

Ln = wvl*Dz*N/L1    # observation plane grid length [m]
d1 = L1/Nprop   # object-plane grid spacing [m]
#d1b = L1/Nimg   # reduced resolution grid spacing in object plane [m]
dn = Ln/Nprop   # detector-plane grid spacing [m]
#k = 2*np.pi / wvl
Dap = Ln        # size of receiving aperture

# ---------- Generate coordinate system ------------- #
# full resolution coordinate system (source plane)
x1 = np.arange(-Nprop/2,Nprop/2)*d1
x1,y1 = np.meshgrid(x1,x1)
# detector coordinate system
xn = np.arange(-Nprop/2,Nprop/2)*dn
[xn,yn] = np.meshgrid(xn,xn)
rn = np.sqrt(xn**2 + yn**2)
# reduced resolution coordinate system (source plane)
#xb = np.arange(-Nprop/2,Nprop/2)*d1b
#xb,yb = np.meshgrid(xb,xb)

# ---------- Generate aperture, reference, phase screen -----#
ap = circ(xn,yn,Dap)    # aperture (circlular mask)
ap_b = ap               # low resolution aperture
Rshift = Ndet/2-Nimg/2  # number of samples to shift object spectrum in the frequency domain
R = create_reference(Ndet,welldepth,Rshift) # genereates a reference field with a phase tilt
r = load_img(target_fname, Nmod)# load the target image
if t is None:
    t,_ = generate_screens(D_over_r0,theta0_til,rytov,Nprop,d1,dn,Dz,z,s,Dap,wvl)

# ------------------- Generate field, propagate, and detect  -------------
I0 = 1.0e4  # inital reflectance scaling actor (somewhat arbitrary)
I = I0*r    # scale reflectance
N0 = I.shape[0]
# generate a random draw of the complex-valued reflection coefficient (at image resolution)
g = np.sqrt(I/2)
f = g   # ignore quadratic phase in object plane
fpad = np.pad(f,padsize_prop)   # pad reflectance coefficient prior to propagation
                                # only the energy that makes its way back to us
# propagate
En,_,_ = ang_spec_multi_prop(fpad, wvl, d1, dn, z, np.exp(1j*t))
Ep = ap*En

#energy1 = np.sum(cmagsq(fpad)) # measure energy in the field. We'll assume this is
#energy2 = np.sum(cmagsq(Ep))   # measure the incident field energy
#Ep2 = Ep*np.sqrt(energy1/energy2)       # renormalize the energy so that we can control the SNR better

# Imaging system loop will readjust input scale to achieve desired snr0
snr_factor=1
snr_error=1
jj=0
while snr_error > .05 and jj<=30:
    jj += 1

    # detect hologram
    Ep = Ep*np.sqrt(snr_factor)     # scale the signal field by factor
    Ep_col = np.pad(ap*lens_phz(Ep,wvl,Dz,dn),padsize_det)  # collimate field and pad array to size of the FPA
    Ef = (1/Ndet) * ift2(Ep_col)    # this and above act to apply a focusing lens to produce the image on the FPA
    h,noise_pwr = detect(cmagsq(Ef+R),welldepth,bits,sigma_rn)    # detect hologram
    scale_fac = (2**-bits)*welldepth/np.mean(np.sqrt(cmagsq(R)))  # compute scale factor for data
    h = h * scale_fac       # scale digitized hologram data to represent photoelectrons
    H = (1/Ndet) * ft2(h)   # convert to freq domain
    H_proc = ap_b * H[-Nimg:,-Nimg:]    # demodulate, low-pass filter, and decimate
    H_proc2 = ap_b * lens_phz(H_proc,wvl,-Dz,dn)  # apply lense curvature to return to the entrance pupil plane
    h_proc = (1/Nimg) * ift2(H_proc)    # convert back to image domain
    # calculate SNR
    sig_with_noise = np.var(H_proc[ap_b==1])  # data with noise and signal
    H_noise = H[:Nimg,-Nimg:]   # data with just noise
    noise = np.var(H_noise)
    SNR = (sig_with_noise-noise)/noise
    snr_error = abs(snr0-SNR)/snr0

    # compute scale factor toward achieving snr0
    if snr0-SNR>0:
        snr_factor = 1/(1-snr_error)
    else:
        if snr_error>1:
            snr_factor = 1/(snr_error)
        else:
            snr_factor = 1-snr_error
    #print("snr iter",jj,snr0,SNR,snr_error,snr_factor)

sig_pwr = np.mean(cmagsq(H_proc[ap_b==1]))
noise_pwr = np.mean(cmagsq(H_noise))
CNR = (sig_pwr-noise_pwr)/noise_pwr

Ib = cmagsq(forward(1,H_proc2,ap,wvl,Dz,d1,Ln,z,np.zeros(t.shape),1,1))  # single shot blurry image
Ib = Ib/np.max(Ib)

if sim_plt_flg is True:
    #plt.ion()

    # these are only used for plot output
    rhat2 = cmagsq(forward(1,H_proc2,ap,wvl,Dz,d1,Ln,z,t,1,1)) # phase corrected DH image
    H[Ndet//2-5:Ndet//2+6,Ndet//2-5:Ndet//2+6]=0    # high-pass filter to remove dc term (for ploting purposes)

    # plot various intermediate outputs
    plt.figure()
    vmin=-30
    vmax=0
    cmap='gray'
    extent1 = [x1[0,0],x1[0,-1],y1[0,0],y1[-1,0]]
    extentn = [xn[0,0],xn[0,-1],yn[0,0],yn[-1,0]]
    plt.subplot(241)
    plt.imshow(r, extent=extent1, cmap=cmap)
    plt.title('Reflectance')
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.subplot(242)
    plt.imshow(cnormdb(fpad), extent=extent1, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title('Speckled Obj')
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    plt.subplot(243)
    plt.imshow(cnormdb(ap * Ep), extent=extentn, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title('Pupil Signal')
    plt.xlabel('[m]')
    plt.ylabel('m')
    plt.subplot(244)
    plt.imshow(h, cmap=cmap)
    plt.title('Hologram')
    plt.subplot(245)
    plt.imshow(cnormdb(H), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title('Hologram Spectrum')
    plt.subplot(246)
    plt.imshow(cnormdb(H_proc), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title('Pupil Img')
    plt.subplot(247)
    plt.imshow(cnormdb(h_proc), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title('Blurry DH Image')
    plt.subplot(248)
    plt.imshow(normdb(rhat2), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title('Phase-Corrected DH Image')

    cmap='jet'
    plt.figure()
    plt.imshow(unwrap_phase(np.angle(h_proc)), cmap=cmap)
    plt.colorbar(label='[rad]')
    plt.title('Unwrapped Phase')


    # plot phase screens
    plt.figure()
    vmin=np.floor(np.min(t))
    vmax=np.ceil(np.max(t))
    vmin,vmax=None,None
    cmap='jet'
    #cmap='viridis'
    for tt in range(1,t.shape[0]):
        alpha = z[tt]/z[-1]
        Delta = (1.0-alpha) * d1 + alpha * dn
        xmax = Nprop/2 * Delta
        extent = [-xmax,xmax,-xmax,xmax]
        if t.shape[0]>2:
            plt.subplot(1+(t.shape[0]-2)//4,min([4,t.shape[0]-1]),tt)   # subplot w/ 4 columns max
        plt.imshow(t[tt],extent=extent,vmin=vmin,vmax=vmax,cmap=cmap); plt.colorbar(label='[rad]')
        plt.xlabel('[m]');plt.title('phase screen {}, z={:.1f}km'.format(tt,z[tt]*1e-3))
        #plt.ylabel('[m]');

    if False:
        plt.figure()
        xl,xh=700,800   # define bounding box for zoomed hologram
        yl,yh=700,800
        extent = [xl-0.5,xh+0.5,yh+0.5,yl-0.5]
        plt.subplot(121); plt.imshow(h,cmap='gray');plt.title('Simulated hologram')
        plt.subplot(122); plt.imshow(h[yl:yh+1,xl:xh+1],extent=extent,cmap='gray');plt.title('(zoomed in)')

        plt.figure(); plt.hist(h.flatten(),100); plt.title('hist of hologram')
        #print("min,max",np.min(h),np.max(h))

    #input("press Enter")
    print("close figs to continue")
    plt.show()

# Assign the fields for returning the data

if plane == 'pupil':
    sim.y = H_proc2  # pupil-plane
elif plane == 'image':
    sim.y = H_proc  # image-plane
else:
    raise Exception("dhsim: 'plane' {} unsupported".format(plane))
sim.plane = plane
sim.yip = H_proc  # yip to be deprecated
sim.ypp = H_proc2  # ypp to be deprecated
sim.ap = ap
sim.wvl = wvl
sim.Dz = Dz
sim.Ln = Ln
sim.dn = dn
sim.L1 = L1
sim.d1 = d1

sim.issim = True
sim.r_true = r
sim.t = t[1:, :, :]  # clip off phz=0 screen at object plane
sim.s = s
sim.z = z

sim.D_over_r0 = D_over_r0
sim.theta0_til = theta0_til
sim.rytov = rytov
sim.SNR = SNR
sim.CNR = CNR
sim.Ib = Ib



sim.set_recon_params(bootstrap_it=10, NL1=20, NL2=200, plt_flg=True, nplt=100)
# set hpdh_mode to 'hybrid' if reconstructing multiple phase planes
sim.recon(hpdh_mode='fastest', unwrap=True)  # Reconstruct r,phz

# Plot final estimated reflectance/phase
print("Plotting final estimated reflectance/phase, Strehl")
plt.figure(); plt.imshow(sim.r,cmap='gray'); plt.title("estimated reflectance")
plt.figure(); plt.imshow(sim.t[0],cmap='jet'); plt.title("true phase"); plt.colorbar()
plt.figure(); plt.imshow(sim.phz[0],cmap='jet'); plt.title("estimated phase"); plt.colorbar()

# Plot strehl ratio and rmse vs. iteration number
plt.figure(); plt.plot(sim.strehl); plt.title('Strehl ratio'); plt.xlabel('iteration')
plt.figure(); plt.plot(sim.rmse); plt.title('RMSE'); plt.xlabel('iteration')
print("Close figs to continue")
plt.show()
