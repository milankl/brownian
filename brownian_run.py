## BROWNIAN MOTION
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import glob

""" BROWNIAN MOTION, physical basis:

elastic collision

    vr = v1-v2
    xr = x1-x2

    v1new = v1 - 2*m2/(m1+m2)*vr.dot(vr)/xr.dot(xr)*xr
    v2new = v2 + 2*m1/(m1+m2)*vr.dot(vr)/xr.dot(xr)*xr

detect collision - adaptive timestep

    s   particle size (radius)
    |xr|**2 + 2*(xr.dot(vr))*t + |vr|**2*t**2 = 4s**2

solve for tmin, but tmin > 0

collision with wall

    t = (L - x0)/u, (H - y0)/v, -x0/u, -y0/v

find smallest t.

collision with wall with gravity

    t = v/g +- sqrt((v/g)**2 + 2/g*(y0-H))

"""

# parameters
N = 100           # number of particles
Nt = 3000       # number of time steps
dtmax = 1e-1    # maximum time step
g = 0          # gravity acceleration
s = 1e-2          # size of particles
ef = 1e-15      # tolerance error around boundaries

# periodic in x,y? only 1,0 allowed
periodicx = 0
periodicy = 0

# domain
L = 2.
H = 2.
ez = np.array([0,1])

# initial conditions
r = np.random.rand(2,N)
r[0,:] *= L
r[1,:] *= H

# separate in two groups of equal size
Nhalf = int(N/2)
r[0,:Nhalf] *= 0.5
r[0,Nhalf:] *= 0.5
r[0,Nhalf:] += (L*0.5)

uv = np.random.rand(2,N)-.5

# set the last particle ('seed') in the middle of the domain and set velocity=0
r[:,-1] = [L/2,H/2]
uv[:,-1] = 0

# a flag vector with information whether particles are dead/alive
# i.e. whether the become 'dead' when they touch the seed
alive = np.ones(N)
alive[-1] = 0   # initially only the seed particle is dead

# storing options
storing = 1     # 1 for storing 0 for no storing
path = 'data/'  # relative path
fstore = 5     # storing frequency

## difference operator
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

D,idx2nm = Dmat(N)

def nm2idx(n,m):
    return sum(range(N-n,N))+m-n-1

## functions
def wallcollision(x,y,u,v):
    """ Computes new positions and velocity for particles that collide with a wall.
    Due to adaptive time stepping, particles will sit directly on the wall. However,
    allow for rounding errors by setting ef larger than machine precision."""

    # boolean array to determine particles close to a wall
    xL = (x >= (L-ef))
    x0 = (x <= ef)
    yH = (y >= (H-ef))
    y0 = (y <= ef)

    # store previous vertical coordinates for energy fix computation
    y0pre = y[y0]
    yHpre = y[yH]

    # mirror position across wall or apply periodic boundaries
    # but place at least 2*ef away from the wall
    x[xL] = (2.*L - x[xL] + periodicx*(2*x[xL] - 3*L)).clip(2*ef,L-2*ef)
    x[x0] = (-x[x0] + periodicx*L).clip(2*ef,L-2*ef)
    y[yH] = (2.*H - y[yH] + periodicy*(2*y[yH] - 3*H)).clip(2*ef,H-2*ef)
    y[y0] = (-y[y0] + periodicy*H).clip(2*ef,H-2*ef)

    # reverse momentum or not for periodic boundaries
    u[xL+x0] = u[xL+x0]*(-1 + 2*periodicx)
    v[yH+y0] = v[yH+y0]*(-1 + 2*periodicy)

    # energy fix
    v[y0] = np.sign(v[y0])*(1-periodicy)*np.sqrt(v[y0]**2 - 2*g*(y[y0] - y0pre))
    v[yH] = np.sign(v[yH])*(1-periodicy)*np.sqrt(v[yH]**2 - 2*g*(y[yH] - yHpre))

    return np.vstack((x,y)),np.vstack((u,v))

def mom_exchange(r1,r2,uv1,uv2):
    """ change momentum based on velocity. """

    dr = r1-r2
    duv = uv1-uv2
    uv1 = uv1 - duv.dot(dr) / dr.dot(dr) * dr
    uv2 = uv2 + duv.dot(dr) / dr.dot(dr) * dr

    return uv1,uv2

## functions for timestepping
def timestep(x,y,u,v,col_nm):
    """ computes the smallest time step till next collision with a wall.
    In case this is larger than dtmax, return dtmax instead. This function
    ignores any effect of g."""

    # time till wall collision
    txL = (L - s - x)/u
    tx0 = (s-x)/u

    if g > 0: # the gravity case
        tyH = pq(-2*v/g,-2/g*(y-H+s))
        ty0 = pq(-2*v/g,-2/g*(y-s))

    else:
        tyH = (H - s - y)/v
        ty0 = (s-y)/v

    # concatenate and find positive minimum
    tw = np.array((txL,tx0,tyH,ty0))
    twmin = tw[tw>0].clip(0,dtmax).min()

    # time till next particle collision
    tc = time_to_particle_collision(x,y,u,v)

    # if previous collison set their time to dtmax
    if len(col_nm):
        tc[nm2idx(*col_nm)] = dtmax

    if N > 1:
        tcmin = tc.min()
    else:   # there are no collision for one particle
        tcmin = 2*dtmax

    # find colliding particles
    if tcmin < twmin:
        col_nm = idx2nm[np.where(tc == tcmin)[0][0]]
    else:
        col_nm = np.empty(0)

    return min(twmin,tcmin),col_nm

def pq(p,q):
    """ Solves a quadratic equation of the form
        x**2 + p*x + q = 0. Fills nans and negatives with dtmax."""
    x = -p/2. + np.outer(np.array([1.,-1]),np.sqrt((p/2.)**2 - q))
    x[np.logical_or(np.isnan(x),x < ef)] = dtmax
    return x.min(axis=0)

def time_to_particle_collision(x,y,u,v):
    """ Computes the time for each particle pair of size s till collision.
    Replaces all times for particles being closer than 2*s by dtmax, to avoid
    them being caught within each other."""
    xr = D.dot(x)
    yr = D.dot(y)
    ur = D.dot(u)
    vr = D.dot(v)

    xrm = xr**2 + yr**2
    vrm = ur**2 + vr**2

    # solve quadratic equation
    p = 2*(xr*ur + yr*vr) / vrm
    q = (xrm - 4*s**2) / vrm
    t = pq(p,q)

    # this line is important to avoid stuck particle pairs
    t[xrm <= 4*s**2] = dtmax
    return t

## time loop - euler forward
# collision indices n,m indicating which particles did collide
col_nm = np.empty(0)

# preallocate storage matrix
R = np.empty((2,N,int(Nt/fstore)+1))
tvec = np.empty((int(Nt/fstore)+1))
dt = 0.
t = 0.

# store initial conditions
istore = 0
R[:,:,istore] = r
tvec[istore] = t

for i in range(Nt-1):

    dt,col_nm = timestep(*r,*uv,col_nm)

    r += (dt*uv.T - g/2.*dt**2*ez).T
    uv[1,:] -= dt*g

    if len(col_nm):
        n,m = col_nm
        # in case one particle of the two colliding is dead, set both to be dead
        if (alive[m] == 0) or (alive[n] == 0):
            alive[[n,m]] = 0
            uv[:,n],uv[:,m] = 0,0
        else: # collision of two particles that are still alive
            uv[:,n],uv[:,m] = mom_exchange(r[:,n],r[:,m],uv[:,n],uv[:,m])
    else:
        r,uv = wallcollision(*r,*uv)

    t += dt

    # feedback on integration progress
    if ((i+1)/Nt*100 % 5) < (i/Nt*100 % 5):
        print(str(int((i+1)/Nt*100.))+'%')

    # store positions in matrix that is later stored to file
    if i % fstore == 0:
        istore += 1
        R[:,:,istore] = r
        tvec[istore] = t

# storage in file
if storing:
    allfiles = glob.glob(path+'dat*.npy')
    if allfiles:
        fileid = '%03i' % (max([int(file[-7:-4]) for file in allfiles])+1)
    else:
        fileid = '000'
    np.save(path+'/dat'+fileid+'.npy',dict(R=R,t=tvec,s=s,L=L,H=H,g=g,N=N,Nt=Nt))
    print('Data saved in dat'+fileid)
else:
    print('Data not stored.')
