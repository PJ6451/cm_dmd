import numpy as np
from scipy.linalg import toeplitz as toep
import matplotlib.pyplot as plt

def cm_dmd(rawdata, NT, thrshhld, tvals, dt):
    # assign values for regression
    y = rawdata[:,-1]
    X = rawdata[:,:-1]

    #svd and building companion matrix
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
        
    c = vr @ np.diag(1. / sr) @ np.conj(ur.T) @ y
    comp_mat = np.diag(np.array([1.]*(NT-2)), k = -1)
    comp_mat[:,-1] = c

    #calculating eigenvalues/vectors/modes
    evals, evecs = np.linalg.eig(comp_mat)
    mag_evals = np.abs(evals)
    #plot_eigs(evals)

    #refactor m based on eigenvalues
    ind = np.where(mag_evals < .99)
    m = ind[0][0]
    evals = evals[:m]
    evecs = evecs[:m,:m]
    modes = np.dot(X[:,:m], evecs)
    
    recon = np.zeros([rawdata.shape[0], m])
    for i in range(m):
        recon[:,i] = np.real(modes @ evals**i)

    return recon, modes, m

def path_reconstruction(phim, window, initconds):
    phimat = phim[:, ::(window - 1)]
    u, s, vh = np.linalg.svd(phimat.T, full_matrices=False)
    kmat = np.conj(vh.T) @ np.diag(1. / s) @ np.conj(u.T) @ initconds
    recon = np.real(kmat.T @ phim)
    return recon

def plot_eigs(evals):
    fig = plt.figure(figsize = (10, 7))
    plt.scatter(evals.real, evals.imag)
    plt.xlabel("real")
    plt.ylabel("imag")
    plt.show()