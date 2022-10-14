import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cm_dmd import *

def lorenz(t, X, sigma=10, beta=2.667, rho=28):
        u, v, w = X
        up = -sigma * (u - v)
        vp = rho * u - v - u * w
        wp = -beta * w + u * v
        return up, vp, wp

dt = .05
t0 = 0.
tf = 30.
NT = int((tf-t0)/dt)
tvals = np.linspace(t0,tf,NT+1)
numiconds = 80
num_dim = 3

icx = np.random.uniform(-15, 15, numiconds)
icy = np.random.uniform(-20, 20, numiconds)
icz = np.random.uniform(0, 40, numiconds)
tspan = np.array([0, tf])
dts = np.arange(0, tf, dt)
rawdata = np.zeros(shape=(num_dim, numiconds, NT))
initconds = np.array([icx,icy,icz]).T

for ii, ic in enumerate(initconds):
    tmp = solve_ivp(lorenz, t_span=tspan, y0=ic, method='RK45', t_eval=dts)
    rawdata[:, ii, :] = tmp.y

thrshhld = 15.
evls, phim, kvecs = cm_dmd(rawdata, num_dim, numiconds, NT)
recon = path_reconstruction(phim, initconds, num_dim, numiconds, NT)
recon = np.reshape(recon, (num_dim, numiconds))

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
traj = rawdata[:,0,:]
ax.plot3D(traj[0,:], traj[1,:], traj[2,:],label='rk4')
ax.plot3D(recon[0,:], recon[1,:], recon[2,:],label='cmdmd')
ax.legend()
#fig.savefig("lorentz96_hdmd_234", dpi=200)
plt.show()