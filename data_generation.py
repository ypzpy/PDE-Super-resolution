import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from tqdm.std import trange
from torch.utils.data import dataloader


def create_T(N):
    """
    Create the inner blocks of the A matrix (excluding boundaries)
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
    """
    m = np.zeros((N,N))
    m[0,0] = 1.
    m[1,0] = 1.
    m[N-1,N-1] = 1.
    m[N-2,N-1] = 1.
    
    for i in range(1,N-1):
        m[i,i] = -4.
        
    for i in range(1,N-2):
        m[i,i+1] = 1.
        m[i+1,i] = 1.
    
    return m


def create_rest(N):
    """
    Create the outer blocks of the A matrix (at boundaries)
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
    """
    m = np.zeros((N,N))
    m[1:N-1,1:N-1] = np.eye(N-2)
    
    return m


def generate_data(N):
    """
    Generate the PDE data u which follows the equation Au = b using finite difference method
    
    Args
    ----------
    N : int
        The size of PDE domain is (N,N)
        
    Outputs
    ----------
    w : ndarray of size (N,N)
        The PDE solution u for the equation Au = b 
    r : ndarray of size (N**2,1)
        The forcing term b
    A : ndarray of size (N**2,N**2)
        The matrix A
    x,y : ndarray of size (N,1)
        x and y coordinates
    """
    
    # Initialisation
    A = np.zeros((N**2,N**2))
    h = 1/(N-1)
    x = np.arange(0,1.0001,h)
    y = np.arange(0,1.0001,h)
    N2 = (N-2)*(N-2)
    
    # Set the top left and bottom right N*N block to identity matrix
    A[0:N,0:N] = np.eye(N)
    A[N*(N-1):N*(N-1)+N,N*(N-1):N*(N-1)+N] = np.eye(N)

    # Set the inner blocks of matrix A
    for i in range(1,N-1):
        A[N*i:N*i+N,N*i:N*i+N] = create_T(N)
    
    # Set the boundary blocks of matrix A
    for i in range(0,N-2):
        A[N*(i+1):N*(i+1)+N,N*i:N*i+N] = create_rest(N)
        A[N*(i+1):N*(i+1)+N,N*(i+2):N*(i+2)+N] = create_rest(N)
        
    # Work out the forcing term r
    r_middle = np.zeros(N2)

    for i in range (0,N-2):
        for j in range (0,N-2):
            r_middle[i+(N-2)*j] = -(8*np.pi**2) * np.sin(2*np.pi*x[i+1]) * np.sin(2*np.pi*y[j+1])*h**2
                
    r = np.zeros(N**2)
    for i in range(N-2):
        r[N*(i+1)+1:N*(i+1)+N-1] = r_middle[(N-2)*i:(N-2)*i+(N-2)]
        
    # Work out w where Aw = r
    w = np.linalg.solve(A,r).reshape((N,N))
    
    return w, r, A, x, y


def generate_pairs_from_ulow(num, l, sigma, N_low, N_high, file_name):
    """
    Generate and save high-resolution & low-resolution pairs by super-resolving the forcing term of low-res field
    
    Args
    ----------
    num: int
        The number of training labels required
    l, sigma: float
        hyperparameters of the covariance kernal
    N_low: int
        The size of low-res PDE domain is (N_low,N_low)
    N_high: int
        The size of high-res PDE domain is (N_high,N_high)
    file_name: string
        Name of the h5 file to be saved
    """
    w_low, r_low, A_low, x_low, y_low = generate_data(N_low)
    w_high, r_high, A_high, x_high, y_high = generate_data(N_high)
    
    h_low = 1/(N_low-1)
    h_high = 1/(N_high-1)

    G = gaussian_kernal(x_low,y_low,l,sigma,N_low)
    
    for i in tqdm(range(num)):
        r_low_sample = np.random.multivariate_normal(r_low.ravel(),G)
        r_high_sample = cv2.resize(r_low_sample.reshape(N_low,N_low), dsize=(N_high,N_high), interpolation=cv2.INTER_CUBIC)/(h_low**2) * (h_high**2)
        r_high_sample = r_high_sample.reshape(-1,1)
        
        C = np.linalg.solve(A_high,r_high_sample)
        w_high_sample = C.reshape((N_high,N_high))
        
        C = np.linalg.solve(A_low,r_low_sample)
        w_low_sample = C.reshape((N_low,N_low))
        
        if i == 0:
            total_high = w_high_sample.reshape(1,w_high_sample.shape[0],-1)
            total_low = w_low_sample.reshape(1,w_low_sample.shape[0],-1)
        else:
            total_high = np.concatenate([total_high,w_high_sample.reshape(1,w_high_sample.shape[0],-1)],axis=0)
            total_low = np.concatenate([total_low,w_low_sample.reshape(1,w_low_sample.shape[0],-1)],axis=0)
            
    # Save to h5py file
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset("high_res",  data=total_high)
        hf.create_dataset("low_res",  data=total_low)
        
    return


def generate_pairs_from_uhigh(num, l, sigma, N_low, N_high, file_name):
    """
    Generate and save high-resolution & low-resolution pairs by downscaling the forcing term of high-res field
    
    Args
    ----------
    num: int
        The number of training labels required
    l, sigma: float
        hyperparameters of the covariance kernal
    N_low: int
        The size of low-res PDE domain is (N_low,N_low)
    N_high: int
        The size of high-res PDE domain is (N_high,N_high)
    file_name: string
        Name of the h5 file to be saved
    """
    w_low, r_low, A_low, x_low, y_low = generate_data(N_low)
    w_high, r_high, A_high, x_high, y_high = generate_data(N_high)
    
    h_low = 1/(N_low-1)
    h_high = 1/(N_high-1)
    scale = int(h_low/h_high)

    G = gaussian_kernal(x_high,y_high,l,sigma,N_high)
    
    for k in tqdm(range(num)):
        r_high_sample = np.random.multivariate_normal(r_high.ravel(),G)
        r_high_sample = r_high_sample.reshape(N_high,N_high)
        r_low_sample = np.zeros((N_low,N_low))
    
        for i in range(0,N_low-1):
            for j in range(0,N_low-1):
                r_low_sample[i][j] = r_high_sample[scale*i][scale*j]/(h_high**2) * (h_low**2)
                
        C = np.linalg.solve(A_high,r_high_sample.reshape(-1,1))
        w_high_sample = C.reshape((N_high,N_high))
        
        C = np.linalg.solve(A_low,r_low_sample.reshape(-1,1))
        w_low_sample = C.reshape((N_low,N_low))
        
        if k == 0:
            total_high = w_high_sample.reshape(1,w_high_sample.shape[0],-1)
            total_low = w_low_sample.reshape(1,w_low_sample.shape[0],-1)
        else:
            total_high = np.concatenate([total_high,w_high_sample.reshape(1,w_high_sample.shape[0],-1)],axis=0)
            total_low = np.concatenate([total_low,w_low_sample.reshape(1,w_low_sample.shape[0],-1)],axis=0)
            
    # Save to h5py file
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset("high_res",  data=total_high)
        hf.create_dataset("low_res",  data=total_low)
        
    return


def gaussian_kernal(x,y,l,sigma,N):
    """
    Work out the covariance of GP for the forcing term
    
    Args
    ----------
    x, y: ndarray
        x and y coordinates
    l, sigma: float
        hyperparameters of the covariance kernal
    N : int
        The size of PDE domain is (N,N)
    """
    m = N*N
    n = N*N
    dist_matrix = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = (y[i%N]-y[j%N])**2 + (x[i//N]-x[j//N])**2
    
    return sigma ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)


def u_prior(l,sigma,N):
    """
    Work out the mean and covariance matrix of the prior of u
    
    Args
    ----------
    l, sigma: float
        hyperparameters of the covariance kernal
    N : int
        The size of PDE domain is (N,N)
    """
    w, r, A, x, y = generate_data(N)
    mean_u = w
    G = gaussian_kernal(x,y,l,sigma,N)
    covariance_u = np.matmul(np.linalg.solve(A,G),np.linalg.inv(A).T)
    
    return mean_u, covariance_u