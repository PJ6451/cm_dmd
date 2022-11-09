import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cm_dmd import *
import pickle
import os

def lorenz(t, X, sigma=10, beta=2.667, rho=28):
        u, v, w = X
        up = -sigma * (u - v)
        vp = rho * u - v - u * w
        wp = -beta * w + u * v
        return up, vp, wp

dt = .05
t0 = 0.
tf = 20.
NT = int((tf-t0)/dt)
tvals = np.linspace(t0,tf,NT+1)
numiconds = 80
num_dim = 3

data_fname = 'lorenz_data.pkl'
if os.path.exists(data_fname):
    # Load data from file
    rawdata = pickle.load(open(data_fname, 'rb'))
    initconds = rawdata[:,:,0]
else:
    icx = np.random.uniform(-15, 15, numiconds)
    icy = np.random.uniform(-20, 20, numiconds)
    icz = np.random.uniform(0, 40, numiconds)
    tspan = np.array([0, tf])
    dts = np.arange(0, tf, dt)
    rawdata = np.zeros(shape=(num_dim, numiconds, NT))
    initconds = np.array([icx,icy,icz])

    for ii, ic in enumerate(initconds.T):
        tmp = solve_ivp(lorenz, t_span=tspan, y0=ic, method='RK45', t_eval=dts)
        rawdata[:, ii, :] = tmp.y
    
    pickle.dump(rawdata, open(data_fname, 'wb'))



thrshhld = 15.
recon = cm_dmd(rawdata, num_dim, numiconds, NT, thrshhld)
#recon = path_reconstruction(phim, initconds, num_dim, numiconds, NT)

fig = plt.figure(figsize = (10, 7))
recon_1 = recon[:,0,:]
traj = rawdata[:,0,:]
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot3D(traj[0,:], traj[1,:], traj[2,:],label='rk4')
ax.legend()
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot3D(recon_1[0,:], recon_1[1,:], recon_1[2,:],label='cmdmd')
ax.legend()
#fig.savefig("lorentz96_hdmd_234", dpi=200)
plt.show()