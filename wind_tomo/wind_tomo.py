import numpy as np
def ift3(X,scale=1):
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(X))) * scale


def ft_phase_volume(r0, N, delta, L0=np.inf, l0=0.0):
    """
    Generate phase volume consistent with random draw of atmospheric turbulence.

    Args:
        r0 (float): Fried's coherence length [m]
        N (int): Volume size (NxNxN)
        delta (float): grid sampling interval in [m]
        L0 (float, optional): [inf] von Karman PSD, one over outer scale frequency [m]
        l0 (float, optional): [0] von Karman PSD, one over inner scale frequency [m]

    Returns:
        ndarray (float): NxNxN phase Volume
    """

    # setup the PSD
    del_f = 1/(N*delta)     # frequency grid spacing [1/m]
    fx = np.arange(-N/2,N/2)*del_f
    fx,fy,fz = np.meshgrid(fx,fx,fx)
    f = np.sqrt(fx**2 + fy**2 + fz**2)

    # modified von Karman atmospheric phase PSD
    #fm = 5.92/l0/(2*np.pi)             # inner scale frequency [1/m]
    oneover_fm = l0*(2.0*np.pi)/5.92   # one over inner scale frequency [1/m]^-1 (avoid div by zero)
    f0 = 1/L0                           # outer scale frequency [1/m]

    #PSD_phi = 0.023*r0**(-5/3) * np.exp(-(f*oneover_fm)**2) / (f**2 + f0**2)**(11/6)
    with np.errstate(divide='ignore'):
        PSD_phi = 0.023*r0**(-5/3) * np.divide(np.exp(-(f*oneover_fm)**2), (f**2 + f0**2)**(11/6))
    PSD_phi[N//2,N//2,N//2] = 0

    # random draws of Fourier coefficients
    cn = (np.random.randn(N,N,N) + 1j*np.random.randn(N,N,N)) * np.sqrt(PSD_phi)*del_f
    phz = np.real(ift3(cn,N**3))

    return phz