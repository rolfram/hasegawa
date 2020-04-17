import numpy as np
import matplotlib.pyplot as plt
import glob

file_sav = sorted(glob.glob('./Checkpoints/*.npz'))
sav_size = len(file_sav)

frames = 49
print(sav_size)
k = 0
sizetuple = (10,8)
fig, ax = plt.subplots(2, 2, figsize=sizetuple)

for i in range(sav_size):
    print(file_sav[i])
    data = np.load(file_sav[i])
    zetahst = data['zetahst']
    phifhst = data['phifhst']
    phihst = data['phihst']
    
    for ra in range(frames):
        if ra%2==0:
        	im1=ax[0,0].imshow(zetahst[ra,:,:]            ,aspect='auto',origin='lower',cmap='gnuplot');ax[0,0].axis('off');ax[0,0].set_title(r'$\zeta\ (vorticity)$')
        	im2=ax[0,1].imshow(phifhst[ra,:,:]               ,aspect='auto',origin='lower',extent=[-128,128,-128,128],cmap='viridis');ax[0,1].axis('on');ax[0,1].set_title(r'$\phi_k \ $(Potiential in k-space)')
        	ax[0,1].set_xlim(-15,15)
        	ax[0,1].set_ylim(-15,15)
        	im3=ax[1,0].imshow(phihst[ra,:,:]             ,aspect='auto',origin='lower',cmap='RdYlBu_r');ax[1,0].axis('on');ax[1,0].set_xlabel('x');ax[1,0].set_title(r'$\phi\ (potential)$')
        	im4=ax[1,1].imshow(np.log(phifhst[ra,:,:]),aspect='auto',origin='lower',cmap='jet');ax[1,1].axis('off');ax[1,1].set_title(r'Log($\phi_k$)')
        	ax[1,1].set_xlim(113,143)
        	ax[1,1].set_ylim(113,143)
        	plt.draw()
        
        	if k < 10:
            		savefile = './snapshots/snap_00000'+str(k)+'.png'
        	elif 10 <= k < 100:
            		savefile = './snapshots/snap_0000'+str(k)+'.png'
        	elif 100 <= k < 1000:
            		savefile = './snapshots/snap_000'+str(k)+'.png'
        	else:
            		savefile = './snapshots/snap_00'+str(k)+'.png'
        	k += 1
        	plt.savefig(savefile)
        else:
            pass
    plt.cla()
