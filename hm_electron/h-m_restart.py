import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sf
import math as mt
import random
import glob

f = sorted(glob.glob('./Checkpoints/*.npz'))

# Change as needed
# figure out an efficient way to automate
data = np.load(f[-1])

zetahst = data['zetahst']
phifhst = data['phifhst']
phihst = data['phihst']

phi = phihst[-1,:,:]

import warnings
warnings.simplefilter('error', RuntimeWarning)
def HM(nx,ny,lx,ly,nt,dt,Z,E,R,Q,mu,phi,isav):
    global KX,KY,KX2,KY2,KXD,KYD

    # Define dx and dy:
    dx=lx/nx; dy=ly/ny

    ### define grids ###
    # np.r_: concatenate any number of array slices along row (row-wise merging).
    # np.arange(start,stop,step): return evenly spaced vales within a given interval. 
    # kx = 2*pi*n/lx; ky = 2*pi*m/ly, where n and m are integers. 
    # so actually L = 41.88...
    # kx =2*np.pi/lx*np.r_[np.arange(nx/2),np.arange(-nx/2,0)]
    # ky =2*np.pi/ly*np.r_[np.arange(ny/2),np.arange(-ny/2,0)]
    b =np.arange(0,19.25,0.15)
    a=np.arange(-19.2,-0.15,0.15)
    kx=np.concatenate((b,a))
    ky=kx

    # for de-aliasing (not sure what this does. Need to research)
    kxd=np.r_[np.ones(nx//3),np.zeros(nx//3+nx%3),np.ones(nx//3)]
    kyd=np.r_[np.ones(ny//3),np.zeros(ny//3+ny%3),np.ones(ny//3)]

    # Create meshgrid and also define kx^2 and ky^2
    kx2=kx**2; ky2=ky**2
    KX, KY =np.meshgrid(kx ,ky )
    KX2,KY2 =np.meshgrid(kx2,ky2)
    KXD,KYD =np.meshgrid(kxd,kyd)

    # Define Fourier transform of phi
    # fft2: 2-D discrete Fourier transform
    # ifft2: 2-D discrete inverse Fourier transform of real or complex sequence.

    # Allocate space to store phi, zeta, and n. In this case phi = n.
    # phi(time,x,y) etc.
    phihst =np.zeros((nt//isav,nx,ny))
    zetahst =np.zeros((nt//isav,nx,ny))

    # Allocate space for phi in fourier-space:
    phifhst =np.zeros((nt//isav,nx,ny))
    # phif[:128,:128] = np.conj(phif[128:,128:])
    
    phi=phi
    
    phif = sf.fft2(phi) # phi is a real function!
    
    phif = np.fft.fftshift(phif)
    # phif[:128,:] = np.conj(np.flip(phif[128:,:]))
    phifhst[0,:,:] = abs(phif)
    phif = np.fft.ifftshift(phif)

    zetaf=-(KX2+KY2)*phif    

    # Define initial condition at t=0.
    phihst[0,:,:] =phi

    zetahst[0,:,:] = np.real(sf.ifft2(zetaf)) # not in calculation

    for it in range(1,nt):

        #---Numerical Method: 4th-order Runge-Kutta
        # time adv. in spectral space (phif). 

        # Previous method has an integrating factor method involve due to hyperviscosity term. 
        # phif=np.exp(-mu*(KX2+KY2)**2*dt)*phif
        # But in this case mu=0 and therefore there is no hyperviscosity term involved. 

        gw1 = adv(phif)
        gw2 = adv(phif+0.5*dt*gw1)
        gw3 = adv(phif+0.5*dt*gw2)
        gw4 = adv(phif+dt*gw3)

        phif=phif+dt*(gw1+2*gw2+2*gw3+gw4)/6

        # Stores values every it%isav==0
        if(it%isav==0):
            # print(it//isav)
            # In previous code. Seems to be a boundary condition in spectral space. 
            # phif[0,0]=0
            phif = np.fft.fftshift(phif)
            # Maybe need to comment this out below? Try tomorrow.
            # phif[:128,:] = np.conj(np.flip(phif[128:,:]))
            phifhst[it//isav,:,:] = abs(phif)
            phif = np.fft.ifftshift(phif)

            # Reality condition (Needs to be enforced):
            # if needed the size can be checked using np.shape(phif)
            # Not sure if this is right...
            # phif[128:,128:] = -np.conj(phif[:128,:128])


            zetaf = -(KX2+KY2)*phif

            # Transorm phi and zeta into real space using IFFT.
            phi=np.real(sf.ifft2(phif))
            zeta=np.real(sf.ifft2(zetaf))

            # Record phi and zeta values in allocated phi and zeta array at instance in time.
            phihst[it//isav,:,:]=phi
            zetahst[it//isav,:,:]=zeta
	    
    return locals()

def adv(phif):
    # phif[0,0]=0. # originally it's not small.
    # phif = phif # /nx

    zetaf=-(KX2+KY2)*phif
    # kconstf=1./(1.+KX2+KY2)
    kconstf = 1./(1.+Z*(KX2+KY2))

    # Define spatial derivatives \partial_x phi, etc. for Poisson bracket.
    # This was written originally for Hasegawa-Wakatani.
    phixf = 1j*KX*phif;  phix =np.real(sf.ifft2(phixf*KXD*KYD))
    phiyf = 1j*KY*phif;  phiy =np.real(sf.ifft2(phiyf*KXD*KYD))
    zetaxf= 1j*KX*zetaf; zetax=np.real(sf.ifft2(zetaxf*KXD*KYD))
    zetayf= 1j*KY*zetaf; zetay=np.real(sf.ifft2(zetayf*KXD*KYD))

    # FFT2 real-space calculation of -phix*zetay+phiy*zetax-kap*phiy
    derivative = sf.fft2(Q*(phix*zetay-phiy*zetax)+R*np.real(sf.ifft2(phiyf))+E*np.real(sf.ifft2(zetayf)))
    # derivative = sf.fft2(-kap*np.real(sf.ifft2(phiyf))) # This works but slowly

    # Multiply by kconstf.
    advff = kconstf*(derivative)

    return advff

nx=ny=257
mx=16
my=8
lx=2*np.pi/0.15; ly=lx
dx=lx/nx; dy=ly/ny
nt=10000; isav=50; dt=1e-5
kap=0.1
Z=1
Q=10000
E=R=1.5

data=HM(nx,ny,lx,ly,nt,dt,Z,E,R,Q,0,phi,isav) # mu=1e-4 for the previous code. 
locals().update(data)

for h in range(len(f)):
    if f[h] is f[-1]:
        numbering = h+2
    else:
        pass

if numbering < 10:
	np.savez('./Checkpoints/hm_00'+ str(numbering) +'.npz',zetahst=zetahst,phifhst=phifhst,phihst=phihst)
elif 10 <= numbering < 100:
	np.savez('./Checkpoints/hm_0'+ str(numbering) +'.npz',zetahst=zetahst,phifhst=phifhst,phihst=phihst)
else:
	np.savez('./Checkpoints/hm_'+ str(numbering) +'.npz',zetahst=zetahst,phifhst=phifhst,phihst=phihst)
