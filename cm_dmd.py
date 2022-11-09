import numpy as np
from scipy.linalg import toeplitz as toep

def cm_dmd(rawdata, num_dim, numiconds, NT, thrshhld):
    recon = np.zeros([num_dim,numiconds,NT])
    # assign values for regression
    y_end = rawdata[:,:,-1]
    Y_arma = rawdata[:,:,:-1]

    #svd and building companion matrix
    u, s, vh = np.linalg.svd(Y_arma, full_matrices=False)

    v = batch_conj_transpose(vh,[num_dim, NT-1, numiconds])
    uh = batch_conj_transpose(u,[num_dim, numiconds,numiconds])
    sig = batch_build_sigma(s, [num_dim, numiconds,numiconds])
    comp_mat = build_comp_mat(num_dim, NT)
        
    c = v @ sig @ uh @ y_end[...,None]
    comp_mat[:,:,-1] = c[:,:,0]

    #calculating eigenvalues/vectors/modes
    evals, evecs = np.linalg.eig(comp_mat)
    modes = Y_arma @ evecs
    
    #Vandermonde Matrix
    #v_m = build_vandermonde(0, NT, evals)
    #c_mat_test = np.real(evecs @ np.diag(evals) @ v_m)
    #aaa = np.allclose(comp_mat, c_mat_test, rtol=1e-05)
    
    recon[:,:,:-1] = modes @ np.linalg.pinv(evecs)
    recon[:,:,-1] = (recon[:,:,:-1] @ c)[:,:,0]

    return recon

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
    if num_dim > 0:
        v_m = np.zeros((num_dim,NT-1,NT-1)) 
        for i in range(num_dim):
            v_m[i,:,:] = np.vander(evals[i,:], N = NT-1, increasing=True)
        return v_m
    else:
        return np.vander(evals, N = NT-1, increasing=True)