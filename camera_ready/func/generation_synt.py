#%%
import torch
import numpy as np
import itertools
import datetime
import os

def generate_x(d, n):
    """
    Generates a matrix X with multivariate normal distribution.

    Args:
        d (int): Number of dimensions.
        n (int): Number of samples.

    Returns:
        x (numpy.ndarray): Generated matrix X.
    """
    mu_x = torch.abs(torch.randn(d))
    cov_x = torch.randn(d, d) / torch.sqrt(torch.tensor(d).float())
    cov_x = cov_x@cov_x.T + torch.eye(d) * 0.5
    x = torch.distributions.MultivariateNormal(mu_x, cov_x).sample((n,))
    return x.T

def generate_z(k, d):
    """
    Generates a matrix Z with absolute normal distribution.

    Args:
        k (int): Number of rows.
        d (int): Number of columns.

    Returns:
        z (numpy.ndarray): Generated matrix Z.
    """
    return torch.abs(torch.randn(k, d))



def project_highest(y, s,thres = None, mode = 'value'):
    """
    Projects a vector `y` to a `s`-sparse vector with highest values.

    Args:
        y (torch.Tensor): Input vector.
        s (int): Number of non-zero elements in the output vector.

    Returns:
        z (torch.Tensor): Projected sparse vector.
    """
    ytopk = torch.topk(y, s, axis=0)
    ind,val = ytopk.indices,ytopk.values
    z = torch.zeros_like(y)
    if mode == 'value':
        for i in range(s):
            z[ytopk.indices[i], range(z.shape[1])] = ytopk.values[i]
        if thres is not None:
            z[z < thres] = 0
    elif mode == 'binary':
        if thres is not None:
            for i in range(y.shape[1]):
                mask = (y[ind[:, i], i] >= thres)
                z[ind[mask, i], i] = 1
        else:
            for i in range(s):
                z[ytopk.indices[i], range(z.shape[1])] = 1

    return z





def store_data(X, Y, Z, file_name):
    """
    Stores the matrices X, Y, Z into files.

    Args:
        X (torch.Tensor): Matrix X.
        Y (torch.Tensor): Matrix Y.
        Z (torch.Tensor): Matrix Z.
        file_name (str): File name prefix.
    """
    d, n, k = X.shape[0], X.shape[1], Y.shape[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'{file_name}_{timestamp}'
    torch.save(X, f'{file_name}_{d}_{n}_{k}_X.pt')
    torch.save(Y, f'{file_name}_{d}_{n}_{k}_Y.pt')
    torch.save(Z, f'{file_name}_{d}_{n}_{k}_Z.pt')




def generate_synt(d=100, n=300, k=200, s=3, it_num=10, snr=0.032, seed=None,project_thres = None, project_mode = 'value'):
    """
    Generates synthetic data.

    Args:
        d (int, optional): Number of dimensions. Defaults to 100.
        n (int, optional): Number of samples. Defaults to 300.
        k (int, optional): Number of rows. Defaults to 200.
        s (int, optional): Number of non-zero elements in the output vector. Defaults to 3.
        it_num (int, optional): Number of iterations. Defaults to 10.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        None
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.datetime.now()
    X = generate_x(d, n).to(device)
    x_time = datetime.datetime.now()
    print('X generated:', x_time - start_time)
    Z = generate_z(k, d).to(device)
    Z_time = datetime.datetime.now()
    print('Z generated:', Z_time - x_time)
    Y = project_highest(Z @ X, s, project_thres, project_mode)
    for _ in range(it_num):
        it_time = datetime.datetime.now()
        Z_hat =  torch.linalg.lstsq(X.T, Y.T).solution.T
        Z = Z_hat.clone()
        Y = project_highest(Z @ X, s, project_thres, project_mode)
        print('Iteration time:', datetime.datetime.now() - it_time)
    condition_y = Z @ X
    infty_norm = torch.max(torch.abs(condition_y), dim=0).values
    noise = torch.randn_like(condition_y) * torch.sqrt(infty_norm) * snr
    Y = project_highest(condition_y + noise, s, project_thres, project_mode)
    store_data(X, Y, Z, f'data/synt_data/gaussian_{snr}_{s}_{it_num}_{project_thres}_{project_mode}'+str(seed))
# %%
