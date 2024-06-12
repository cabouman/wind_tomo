import dh
import numpy as np
from dh.dhsim import ft_phase_screen

Z_total=26 # cm
M=26 #number of plates
del_Z=Z_total/M

V=M*del_Z

convert = 1e-6  # multiply by to convert microns to meters
N = 500
diam = 2 * 300
L = N * 20000 / diam
dx = L / N  # microns per grid point

#autocorrelation is in terms of

# generate phase screen array with initial screen
screens = ft_phase_screen(r0=7000 * convert, N=N, delta=dx * convert)
screens = np.expand_dims(screens, 0)
T_hat= dh.ft2(screens[0])

for i in range(1,M+1,1):
    h_hat = np.exp(-del_Z**2)*T_hat
    mu=np.real(h_hat)
    sigma= max([])
    U = np.random.normal()


