import numpy as np
from scipy.linalg import toeplitz as toep
import tensorflow as tf

def cm_dmd(rawdata, num_dim, numiconds, NT):
    # assign values for regression
    y_end = rawdata[:,:,-1]
    Y_arma = rawdata[:,:,:-1]
    
    #svd and building companion matrix
    u, s, vh = np.linalg.svd(Y_arma, full_matrices=False)
    v = np.zeros((num_dim, NT-1, numiconds))
    uh = np.zeros((num_dim,numiconds,numiconds))
    sig = np.zeros((num_dim,numiconds,numiconds))
    comp_mat = np.zeros([num_dim,NT-1,NT-1])
    comp_mat_diag = np.array([1]*(NT-2))
    comp_mat_ones = np.diag(comp_mat_diag, k = -1)
    for i in range(num_dim):
        v[i,:,:] = np.conj(vh[i,:,:].T)
        uh[i,:,:] = np.conj(u[i,:,:].T)
        sig[i,:,:] = np.diag(1/s[i])
        comp_mat[i,:,:] = comp_mat_ones
        
    c = v @ sig @ uh @ y_end[...,None]
    comp_mat[:,:,-1] = c[:,:,0]

    #calculating eigenvalues/vectors/modes
    kmat = comp_mat
    evls, evcs = np.linalg.eig(kmat)
    phim = np.linalg.solve(evcs, np.reshape(Y_arma,[num_dim, NT-1, numiconds]))
    return evls, phim, evcs


def path_reconstruction(phim, initconds, num_dim, numiconds, NT):
    #constructing phimat
    phimat_t = np.zeros([num_dim, numiconds, NT-1])
    for i in range(num_dim):
        phimat_t[i,:,:] = phim[i,:,:].T
    
    #svd to construct kmat
    u, s, vh = np.linalg.svd(phimat_t, full_matrices=False)
    v = np.zeros((num_dim, NT-1, numiconds))
    uh = np.zeros((num_dim, numiconds, numiconds))
    sig = np.zeros((num_dim, numiconds, numiconds))
    for i in range(num_dim):
        v[i,:,:] = np.conj(vh[i,:,:].T)
        uh[i,:,:] = np.conj(u[i,:,:].T)
        sig[i,:,:] = np.diag(1/s[i])
    
    #kmat and reconstruction of trajectories
    kmat = v @ sig @ uh @ initconds.T[...,None]
    kmat_t = np.zeros([num_dim, 1, NT-1])
    for i in range(num_dim):
        kmat_t[i,:,:] = kmat[i,:,:].T
    
    recon = np.real(kmat_t @ phim)
    return recon


def path_test(query_pts, initconds, phim, evls, window):
    Nobs = np.shape(phim)[0]
    Nqp = np.shape(query_pts)[0]
    Ns = np.shape(query_pts)[1]
    numiconds = int(np.shape(phim)[1] / (window - 1))
    iconphim = np.zeros((Nobs, numiconds), dtype=np.complex128)

    for ll in range(numiconds):
        iconphim[:, ll] = phim[:, ll * (window - 1)]

    u, s, vh = np.linalg.svd(iconphim, full_matrices=False)
    Kmat = (initconds.T) @ (np.conj(vh)).T @ np.diag(1. / s) @ (np.conj(u)).T
    err = np.linalg.norm(initconds.T - Kmat @ iconphim)
    print("Error in first fit: %1.2e" % err)

    uk, sk, vhk = np.linalg.svd(Kmat, full_matrices=False)
    phi_query = np.conj(vhk).T @ np.diag(1. / sk) @ np.conj(uk).T @ query_pts.T
    err = np.linalg.norm(query_pts.T - Kmat @ phi_query)
    print("Error in second fit: %1.2e" % err)

    test_paths = np.zeros((Nqp, Ns, window-1), dtype=np.float64)
    test_paths[:, :, 0] = query_pts
    eveciter = evls
    for ll in range(1, window-1):
        test_paths[:, :, ll] = np.real((Kmat @ np.diag(eveciter) @ phi_query)).T
        eveciter = eveciter * evls
        print(np.abs(eveciter))
    return test_paths