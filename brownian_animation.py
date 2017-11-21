## BROWNIAN MOTION - PLOT
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# load data
runid = 0
path = 'data/'
d = np.load(path+'dat%03i.npy' % runid).all()
d['Nhalf'] = int(d['N']/2)


## plotting
for i in range(d['R'].shape[-1]):

    if i == 0:
        #TODO scatter size
        fig,ax = plt.subplots(1,1,figsize=(4*d['L'],4*d['H']))
        plt.tight_layout()
        S = np.diff(ax.transData.transform([0,1.8*d['s']]))[0]
        Q = ax.scatter(d['R'][0,:,i],d['R'][1,:,i],S**2,['C0']*(d['N']-1)+['k'])
        ax.set_xlim(0,d['L'])
        ax.set_ylim(0,d['H'])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.pause(1)
    else:
        Q.set_offsets(d['R'][:,:,i].T)
        plt.pause(0.0001)
