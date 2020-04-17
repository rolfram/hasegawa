import numpy as np
import math as mt
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import random


nx=ny=257
mx=16
my=8
lx=2*np.pi/0.15; ly=lx
dx=lx/nx; dy=ly/ny
nt=10000; isav=50; dt=1e-5
kap=0.1
Z=1
R=1.5
E=1.5
Q=10000

x = np.arange(nx)*dx
y = np.arange(ny)*dy
X,Y = np.meshgrid(x,y)
px=np.linspace(-mx/2,mx/2,mx)*2.*np.pi/lx
px[8]=0
px[7]=0
print(px)
py=np.arange(my)*2.*np.pi/ly
A=np.zeros((mx,my))+0.*1j
phi=np.zeros((nx,ny))+0.*1j
# had at 0.3-0.5 loop for 64 nx.
# had 0-8 for both modes

for k in range(mx):
    rand=random.uniform(0,2*np.pi)
    A[int(random.uniform(4,12)),int(random.uniform(0,8))]=random.uniform(0.08,0.15)*np.exp(1j*rand)
    
A[8,int(random.uniform(3*my/8,4*my/8))] = 0.11*np.exp(1j*random.uniform(0,2*np.pi))

rand=random.uniform(0,2*np.pi)
A[8,int(my/4)]=1.*np.exp(1j*rand)  


# Actual IC

for i in range(nx):
    for j in range(ny):
        for m1 in range(mx):
            for m2 in range(my):
                phi[i,j]=phi[i,j]+A[m1,m2]*np.exp(1j*px[m1]*x[i]+1j*py[m2]*y[j])
phi=np.real(phi)
phi=np.transpose(phi)

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



data=HM(nx,ny,lx,ly,nt,dt,Z,E,R,Q,0,phi,isav) # mu=1e-4 for the previous code. 
locals().update(data)


np.savez('./Checkpoints/hm_001.npz',zetahst=zetahst,phifhst=phifhst,phihst=phihst)
