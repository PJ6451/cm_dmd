import numpy as np
from scipy.linalg import toeplitz as toep

def cm_dmd(rawdata, num_dim, numiconds, NT):
    # assign values for regression
    y_end = rawdata[:,:,-1]
    Y_arma = rawdata[:,:,:-1]
    print(np.linalg.cond(Y_arma))
    
    #svd and building companion matrix
    u, s, vh = np.linalg.svd(Y_arma, full_matrices=False)

    v = batch_conj_transpose(vh,[num_dim, NT-1, numiconds])
    uh = batch_conj_transpose(u,[num_dim,numiconds,numiconds])
    sig = batch_build_sigma(s, [num_dim,numiconds,numiconds])
    comp_mat = build_comp_mat(num_dim, NT)
        
    c = v @ sig @ uh @ y_end[...,None]
    comp_mat[:,:,-1] = c[:,:,0]

    #calculating eigenvalues/vectors/modes
    kmat = comp_mat
    evals, evecs = np.linalg.eig(kmat)
    phim = Y_arma @ evecs

    #Vandermonde Matrix
    v_m = np.zeros((num_dim,NT-1,NT-1))
    for j in range(NT-1):
        v_m[:,:,j] = evals**j    
    print(np.linalg.cond(v_m))
    
    return evals, phim, evecs

def batch_conj_transpose(A: np.ndarray, dim: list):
    A_t = np.zeros(dim)
    for i in range(dim[0]):
        A_t[i,:,:] = np.conj(A[i,:,:].T)    
    return A_t

def batch_build_sigma(s: np.ndarray, dim: list):
    sig = np.zeros(dim)
    for i in range(dim[0]):
        sig[i,:,:] = np.diag(1/s[i])
    return sig

def build_comp_mat(num_dim, NT):
    comp_mat = np.zeros([num_dim,NT-1,NT-1])
    comp_mat_diag = np.array([1]*(NT-2))
    comp_mat_ones = np.diag(comp_mat_diag, k = -1)
    for i in range(num_dim):       
        comp_mat[i,:,:] = comp_mat_ones
    return comp_mat

def path_reconstruction(phim, initconds, num_dim, numiconds, NT):
    #constructing phimat
    phimat_t = batch_conj_transpose(phim,[num_dim, NT-1, numiconds])
    
    #svd to construct kmat
    u, s, vh = np.linalg.svd(phimat_t, full_matrices=False)
    v = batch_conj_transpose(vh,[num_dim, numiconds, numiconds])
    uh = batch_conj_transpose(u,[num_dim, numiconds, NT-1])
    sig = batch_build_sigma(s,[num_dim, numiconds, numiconds])
    
    #kmat and reconstruction of trajectories
    kmat =  v @ sig @ uh @ initconds[...,None]
    kmat_t = batch_conj_transpose(kmat,[num_dim, numiconds, NT-1])
    recon = np.real(kmat_t @ phim)

    return recon

def path_reconstruction(phim, window, initconds):
    phimat = phim[:, ::(window - 1)]
    u, s, vh = np.linalg.svd(phimat.T, full_matrices=False)
    kmat = np.conj(vh.T) @ np.diag(1. / s) @ np.conj(u.T) @ initconds
    recon = np.real(kmat.T @ phim)
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