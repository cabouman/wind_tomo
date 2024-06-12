import dh
import matplotlib.pyplot as plt
import numpy as np
from dh.dhsim import load_img, ft_phase_screen
from dh.dhdefs import ift2, circ, cmagsq, ft2, cnormdb, forward, normdb
from skimage.restoration import unwrap_phase


# This demo simulates a beam propagation through a phase screen using angular spectrum propagation

#Parameter affecting grid resolution higher diam means higher resolution.
diam=200

#circle of 100s
xx, yy = np.mgrid[:diam, :diam]
Img = np.array(circ(xx-diam/2,yy-diam/2,diam))*100

#AFRL sample image
Img=load_img(None, diam)+1


#Arrays of 100s
# Img=np.ones((diam,diam))*100


padval=0

#make mask for unwrapper
image_mask=(np.pad(Img, padval)==0)

# real image
Img = np.pad(Img, padval)

# determining total image size
N, _ = np.shape(Img)
print(N)

cmap='jet'

plt.figure()
plt.imshow(cnormdb(Img))
plt.title("cnormdb before propagation")
# Field of view of our image in microns
L = N*22000/diam # Ensuring that centered image is 22 mm in diameter to match the paper

#22000 microns= 22 mm = 2.2 centimeters
#

# get axis
dx = L / N
x = np.linspace(-L / 2, L / 2,num=N, endpoint=False) #np.arange(-L / 2, L / 2, dx)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), num=N, endpoint=False) #np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / L)

[FX, FY] = np.meshgrid(fx, fx)

# wavelength (microns)
lam = 0.5

# distance of propagation in microns (26 cm to match article roughly)
z = 260000

# wavenumber
k = 2 * np.pi / lam

# -- angular spectrum propogation with no phase screen -- ##
kz = np.sqrt(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2)
H = np.exp(1j * z * kz)
Img_prop = ift2(ft2(Img) * H)

#Plotting results
phz_noscreen=np.angle(Img_prop)

plt.figure()
plt.imshow(unwrap_phase(np.ma.array(phz_noscreen,mask=image_mask)),cmap=cmap)
plt.title("phase component after propagation")
plt.colorbar()

plt.figure()
plt.imshow(cnormdb((Img_prop)))
plt.title("cnormdb after propagation")
plt.colorbar()


# Now repeat with phase Screens
# Note ignoring Quadratic phase factor
np.random.seed(1)


# -- Angular spectrum propagation with one phase screen placed halfway-- #
# Make Phase screen (ro is 7000 microns (7 mm) to match article)
phz_screen = ft_phase_screen(r0=7000, N=N, delta=dx)

# perform angular spectrum propagation to half distance and apply phase screen.
H = np.exp(1j * (z/2) * kz)
Img_prop2 = np.exp(1j*phz_screen)*ift2(ft2(Img) * H)

# perform angular spectrum propagation to remaining distance
H = np.exp(1j * (k/2) * kz)
Img_prop2 = ift2(ft2(Img_prop2) * H)


#Plotting results
phz_withscreen=np.angle(Img_prop2)

plt.figure()
plt.imshow(phz_withscreen,cmap=cmap)
plt.title("Raw phase component after propagation")
plt.colorbar()

plt.figure()
plt.imshow(unwrap_phase(np.ma.array(phz_withscreen,mask=image_mask)),cmap=cmap)
plt.title("phase component after propagation w/ screen")
plt.colorbar()

plt.figure()
plt.imshow(np.real(phz_screen),cmap=cmap)
plt.title("phase screen")
plt.colorbar()

plt.figure()
plt.imshow(cmagsq(Img_prop2))
plt.title("cnormdb after propagation w/ screen")
plt.colorbar()

plt.show()