import numpy as np 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import scipy as sp
from scipy.ndimage import zoom
import os


def Pmag_func(DH, alpha0, ff, HH0, Ku2, Ms):
	'''

	Parametrized function of the absorved power in FMR experiments. In funtion of 
	Ms, alpha, DH (inomogeneous linewidth) and Hu (anisotropic uniaxial field)

	The expression has been obtained from Mathematica derivation.

	'''
    term1 = 1e-9 * DH
    term2 = alpha0 * ff
    term3 = 1j * 28 * np.abs(HH0 + (2 * Ku2) / Ms + 1.25664e-6 * Ms)
    numerator = 400 * np.pi**2 * np.imag(
        1j * 2.8e19 * ff * (term1 + term2 + term3)
    )

    absH = np.abs(HH0 + (2 * Ku2) / Ms)
    absH_full = np.abs(HH0 + (2 * Ku2) / Ms + 1.25664e-6 * Ms)

    denom = (
        DH**2
        + 2e9 * alpha0 * DH * ff
        + 1e18 * ff**2
        + 1e18 * alpha0**2 * ff**2
        + absH * (
            1j * 2.8e10 * DH
            + 1j * 2.8e19 * alpha0 * ff
            - 7.84e20 * absH_full
        )
        + (
            1j * 2.8e10 * DH
            + 1j * 2.8e19 * alpha0 * ff
        ) * absH_full
    )
    final = - numerator / denom
    return np.real(final), np.imag(final)


def param_norm(x,x_min,x_max):
    x_norm = (x - x_min)/(x_max - x_min)
    return x_norm

def param_denorm(x_norm, x_min, x_max):
    x = x_norm * (x_max - x_min) + x_min
    return x

def A_creation(x_grid,y_grid):
    A = np.zeros((y_grid,x_grid))
    for i in range(y_grid):
        rand = np.random.rand()
        for j in range(x_grid):
            A[i][j] = rand
    return A

def normalization(size,array):
    grid_normalized = np.zeros((size,size))

    for i in range(size):
        fila = array[i, :]
        average = np.average(fila)
        maxim = max(fila)
        minim = min(fila)
        
        if (average + maxim) > (average - minim):
            fila_norm = fila/maxim
    
        #---------------------------------------------------------
        elif average == 0:
            fila_norm = fila*0
        #---------------------------------------------------------
        else:
            fila_norm = fila/np.abs(minim)
        
        grid_normalized[i, :] = fila_norm 
    return grid_normalized
    
def A_creation_noise(x_grid,y_grid,noise):
    A = np.zeros((y_grid,x_grid))  
    
    for i in range(x_grid):
        rand = np.random.random()
        
        for j in range(y_grid):
            A[i][j] = 1 - (rand * noise)
            
    return A    



#Parameters ------------------------
np.random.seed(seed = 2024)
N_noise = 1
N_samples = 20000
Sample_size = 64
noise = 0
gyro = 28.024*10**9
margins_grid = 30
pregrid = Sample_size + 2*margins_grid
# ----------------------------------

# Set the range for the parameter training

xMs = np.random.uniform(low = 5.30, high=6.1461, size = N_samples)
xalpha0 = np.random.uniform(low = -4, high= -1.3010, size = N_samples)
xDH = np.random.uniform(low = np.log10(gyro*10**(-5)) , high= np.log10(gyro*10**(-3)), size = N_samples)
xHu = np.random.uniform(low = -4, high = -1, size = N_samples)
xKu = np.log10(0.5*10**(xMs)*10**(xHu))
Pmag = []
param_train = np.zeros((N_samples*N_noise,4))

fmax = 15 ; fmin = 2.5
hmax = 0.2 ; hmin = 0
# margins 

f_step = (fmax-fmin)/Sample_size
fmax = f_step*margins_grid + fmax
fmin = - f_step*margins_grid + fmin
h_step = (hmax-hmin)/Sample_size
hmax = h_step*margins_grid + hmax
hmin = hmin - h_step*margins_grid



for k in tqdm(range(N_samples)):
    _xMs = xMs[k]
    _xalpha0 = xalpha0[k]
    _xDH = xDH[k]
    _xKu = xKu[k]
    
    Pmag_re = np.zeros((pregrid,pregrid))
    Pmag_im = np.zeros((pregrid,pregrid))
    
    for i in range(pregrid):
        '''
        i: Magnetic Field 
        j: frequency
        '''
        ff = (fmin + (fmax-fmin)*i/(pregrid))
        for j in range(pregrid):
            
            HH0 = (hmin + (hmax-hmin)*j/(pregrid))
            Real_part, Imag_part = Pmag_func(10**(_xDH),10**(_xalpha0),
                                             ff,HH0,10**(_xKu),10**(_xMs))
            Pmag_re[i][j] = Imag_part
            Pmag_im[i][j] = Real_part

    A = A_creation_noise(pregrid,pregrid,noise)
    _Pmag = A * Pmag_re + (1 - A) * Pmag_im
    _Pmag= normalization(pregrid,-_Pmag)
    Pmag.append(_Pmag[margins_grid:(margins_grid+64),margins_grid:(margins_grid+64)])
    param_train[k] = ([param_norm(_xMs,5.30,6.1461),
                       param_norm(_xalpha0,-4,-1.3010), 
                       param_norm(_xDH,np.log10(gyro*10**(-5)),np.log10(gyro*10**(-3))), 
                       param_norm(_xKu,1,4.84)])

Pmag = np.array(Pmag)
# save the data
np.save('./train_data/Pmag_20000_ani.npy',Pmag)
np.save('./train_data/params_20000_ani.npy',param_train)
