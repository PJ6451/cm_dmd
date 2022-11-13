import numpy as np
from scipy.linalg import toeplitz as toep
import matplotlib.pyplot as plt

def cm_dmd(rawdata, NT, thrshhld):
    # assign values for regression
    y_end = rawdata[:,-1]
    Y_arma = rawdata[:,:-1]

    #svd and building companion matrix
    u, s, vh = np.linalg.svd(Y_arma, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
        
    c = vr @ np.diag(1. / sr) @ np.conj(ur.T) @ y_end
    comp_mat = np.diag(np.array([1.]*(NT-2)), k = -1)
    comp_mat[:,-1] = c

    #calculating eigenvalues/vectors/modes
    evals, evecs = np.linalg.eig(comp_mat)
    evls_pwr = np.abs(evals**(NT-1))
    
    #refactor m based on eigenvalues
    ind = np.where(evls_pwr > 60)
    m = ind[0][0]
    evals = evals[:m]
    evecs = evecs[:m,:m]
    modes = Y_arma[:,:m] @ evecs
    modes = modes[:,:m]
    #plot_eigs(evals)
    
    recon = np.zeros([rawdata.shape[0], m])
    for i in range(m):
        recon[:,i] = np.real(modes @ evals**i)
    #Vandermonde Matrix
    #v_m = np.vander(evals, N = m, increasing=True)

    return recon, m

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