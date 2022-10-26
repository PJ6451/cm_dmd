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