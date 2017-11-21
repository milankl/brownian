## BROWNIAN MOTION - MIXING
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse,interpolate

#TODO make it work for an arbitrary number of runs
#TODO probably won't run right away, change after TODO further down

path = '/Users/milan/Dropbox/phd/maths/brownian/data/'
runids = [0] # all runs to average over

# time axis for interpolation
# adjust depending on time of simulation
taxis = np.linspace(0,17,1000)


# functions
def Dmat(N):
    """ Dmat(N) creates a difference matrix D to compute all relative positions (or velocities) of a set of N particles. Return D as sparse csr matrix, for fast matrix-vector multiplication. Didx provides a vector which elements are the particle indices that correspond to a certain difference in the resulting vector of D.dot(x)."""

    D = np.zeros((int(N*(N-1)/2),N)).astype(np.int)

    for i in range(N-1):
        j = sum(range(N-i,N))
        D[j:j+N-i-1,i] = 1
        D[range(j,j+N-i-1),range(i+1,N)] = -1

    # index to particles n,m
    idx2nm = np.where(D)[1].reshape(-1,2)
    return sparse.csr_matrix(D),idx2nm

M = np.empty((len(runids),len(taxis)))

for ir,runid in enumerate(runids):
    d = np.load(path+'dat%03i.npy' % runid).all()
    d['Nhalf'] = int(d['N']/2)

    print(d['t'].max())

    D,idx2nm = Dmat(d['N'])
    Dred,idx2nm_red = Dmat(d['Nhalf'])
    Dblue,idx2nm_blue = Dmat(d['N']-d['Nhalf'])

    red_blue = np.squeeze(np.asarray(D[:,:d['Nhalf']].sum(axis=1))).astype(np.bool)
    Dred_blue = D[red_blue,:]

    ## estimate mixing
    """ the idea is to estimate the mixing m via

            m = 1/2 * ( xy_red + xy_blue) / xy_red_blue

        where

            xy_red,xy_blue = the average distance between all red/blue particles

            xy_red_blue = the average distance between all combinations of red and blue
            particles but not between either red-red or blue-blue."""

    x_red = Dred.dot(d['R'][0,:d['Nhalf'],:])
    y_red = Dred.dot(d['R'][1,:d['Nhalf'],:])
    xy_red = np.nanmean(np.sqrt(x_red**2 + y_red**2),axis=0)
    del x_red,y_red

    x_blue = Dblue.dot(d['R'][0,d['Nhalf']:,:])
    y_blue = Dblue.dot(d['R'][1,d['Nhalf']:,:])
    xy_blue = np.nanmean(np.sqrt(x_blue**2 + y_blue**2),axis=0)
    del x_blue,y_blue

    x_red_blue = Dred_blue.dot(d['R'][0,...])
    y_red_blue = Dred_blue.dot(d['R'][1,...])
    xy_red_blue = np.nanmean(np.sqrt(x_red_blue**2 + y_red_blue**2))
    del x_red_blue,y_red_blue

    m = 0.5*(xy_red + xy_blue) / xy_red_blue

    M[ir,:] = interpolate.interp1d(d['t'],m)(taxis)
    print(ir)

##

#TODO make this work for arbitrary number of runs
Mm = np.nanmean(M.reshape((int(len(runids)/10),10,1000)),axis=1)
Ms = np.nanstd(M.reshape((int(len(runids)/10),10,1000)),axis=1)

cls = ['C'+str(i) for i in range(Mm.shape[0])]
NN = [0,0.5,1.,1.5]

fig,ax = plt.subplots(1,1)

for im in range(int(len(runids)/10)):
    ax.plot(taxis,Mm[im,:],cls[im],label='g = %1.1f' % NN[im])
    ax.fill_between(taxis,Mm[im,:]-Ms[im,:],Mm[im,:]+Ms[im,:],facecolor=cls[im],alpha=.5)

ax.set_xlim(0,17)
ax.set_xlabel('time')
ax.set_ylabel('mixing')
ax.legend(loc=4)
plt.show()
