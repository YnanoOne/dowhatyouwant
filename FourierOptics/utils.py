import numpy as np
import matplotlib.pyplot as plt

def fft2c(g):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))) # centered 2D FFT

def ifft2c(G):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G)))

def make_squareGrid(N, dx):
    x = (np.arange(N) - N//2) * dx 
    X, Y = np.meshgrid(x, x)
    return x, X, Y

def fresnel_propagate(u0, lam, z, dx): 
    N = u0.shape[0]
    fx = (np.arange(N) - N//2) / (N * dx) # frequency space
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * lam * z * (FX**2 + FY**2)) # Fresnel kernel in freq
    uz = ifft2c(fft2c(u0) * H) * np.exp(1j * 2*np.pi/lam * z) # Fresnel convolution & phase
    return uz

def helmholtz_propagate(u0, lam, z, dx):
    N = u0.shape[0]
    fx = (np.arange(N) - N//2) / (N * dx) # frequency space
    FX, FY = np.meshgrid(fx, fx)
    FZ = np.sqrt(1/lam**2 - FX**2 - FY**2 + 0j)
    H = np.exp(1j * 2 * np.pi * z * FZ) # This includes phase
    uz = ifft2c(fft2c(u0) * H) 
    return uz
