import numpy as np

def path_reconstruction_arnoldi(evecs, evals, initconds, num_dim, numiconds, NT):
    a_m = np.zeros((num_dim,NT-1))
    d_a = np.zeros((num_dim,NT-1,NT-1))
    d_a_inv = np.zeros((num_dim,NT-1,NT-1))
    v_m = np.zeros((num_dim,NT-1,NT-1))

    for j in range(NT-1):
        v_m[:,:,j] = evals**j
        a_m[:,j] = np.linalg.norm(evecs[:,j,:],2)

    for i in range(num_dim):
        d_a[i,:,:] = np.diag(a_m[i])
        d_a_inv[i,:,:] = np.diag(1/a_m[i])

    w_m = np.real(evecs @ d_a_inv)
    recon = np.real(w_m @ d_a @ v_m)
    f_NT = np.real(w_m @ d_a @ (evals**NT)[...,None])
    recon = np.concatenate((recon, f_NT),axis=2)
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

def batch_conj_transpose(A: np.ndarray, dim: list):
    if np.size(dim) > 0:
        A_t = np.zeros(dim)
        for i in range(dim[0]):
            A_t[i,:,:] = np.conj(A[i,:,:].T)    
        return A_t
    else:
        return np.conj(A.T)

def batch_build_sigma(s: np.ndarray, dim: list):
    if np.size(dim) > 0:
        sig = np.zeros(dim)
        for i in range(dim[0]):
            sig[i,:,:] = np.diag(1/s[i])
        return sig
    else:
        return np.diag(1/s)

def build_comp_mat(num_dim, NT):
    if num_dim > 0:
        comp_mat = np.zeros([num_dim,NT-1,NT-1])
        comp_mat_diag = np.array([1]*(NT-2))
        comp_mat_ones = np.diag(comp_mat_diag, k = -1)
        for i in range(num_dim):       
            comp_mat[i,:,:] = comp_mat_ones
        return comp_mat
    else:
        comp_mat = np.diag(np.array([1.]*(NT-2)), k = -1)
        return comp_mat

def build_vandermonde(num_dim, NT, evals):
    evals = evals.T
    if num_dim > 0:
        v_m = np.zeros((num_dim,NT-1,NT-1)) 
        for i in range(num_dim):
            v_m[i,:,:] = np.vander(evals[:,i], N = NT-1, increasing=True)
        return v_m
    else:
        return np.vander(evals, N = NT-1, increasing=True)
