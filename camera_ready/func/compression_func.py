#%%
import numpy as np
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit, ElasticNet
from sklearn.model_selection import train_test_split
import time, sys,os,datetime
from func.generation_synt import project_highest

def load_data_subg(filename, dtype='data'):
    """
    Loads data from a file.

    Parameters:
    - filename (str): Name of the file.
    - dtype (str): Type of data ('data' or 'label').

    Returns:
    - tensor: Loaded data.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, filename)

    
    snr = filename.split('_')[2]
    s = filename.split('_')[3]
    seed, timing, d, n , k = filename.split('_')[4], filename.split('_')[-4], filename.split('_')[-3], filename.split('_')[-2], filename.split('_')[-1]
    print(f'SNR: {snr}, Timing: {timing}, d: {d}, n: {n}, k: {k}, seed: {seed}, s: {s}')

        

    if dtype == 'data':
        true_filename = f"{file_path}_X.pt"
        if not os.path.exists(true_filename):
            print(true_filename)
            print('File not found')
            return None
        x = torch.load(true_filename)

        if x.shape[0] != int(d) or x.shape[1] != int(n):
            print('Data shape mismatch')
            return None
        return x, snr, timing, d, n, k
    
    elif dtype == 'label':
        y = torch.load(f"{file_path}_Y.pt")
        if y.shape[0] != int(k) or y.shape[1] != int(n):
            print('Label shape mismatch')
            return None
        return y
    elif dtype == 'regressor':
        z = torch.load(f"{file_path}_Z.pt")
        if z.shape[0] != int(k) or z.shape[1] != int(d):
            print('Regressor shape mismatch')
            return None
        return z
    else:
        print('Type not found')
        return None


def compression_matrix(compression_s, k, delta=0.3, epsilon=0.1, mode='gaussian', m=None):
    """
    Returns a compression matrix that satisfies (s, delta)-RIP with high probability.

    Parameters:
    - s (int): Sparsity level.
    - k (int): Number of columns.
    - delta (float): RIP constant.
    - epsilon (float): Probability bound.
    - mode (str): Type of matrix ('gaussian' or 'hadamard').
    - m (int, optional): Number of rows. If None, calculate the lower bound.

    Returns:
    - np.ndarray: Compression matrix.
    """
    if m is not None:
        print(f'Given m: {m}')
    else:
        output = compression_s * torch.log(9) + compression_s * torch.log(torch.e * (k / compression_s)) + torch.log(2 / epsilon)
        m = int(16 * output / (delta ** 2))
        print(f'Lower bound of m: {m}')
        if m > k / 3:
            print('m is too large')
            m = int(k / 3)
            print(f'Changed m to {m}')

    if mode == 'gaussian':
        phi = torch.randn(m, k) / torch.sqrt(torch.tensor(m))
    elif mode == 'hadamard':
        print('NOTICE: THE COLUMN NUMBER OF HADAMARD MATRIX IS NOT CALCULATED')
        return None
    else:
        print('Mode not found')
        return None

    return phi



def calculate_Wopt(X, Y, phi, lamb=0, r=None):
    """
    Calculates the optimal W matrix.

    Parameters:
    - X (tensor): Dictionary matrix.
    - Y (tensor): Signal matrix.
    - lamb (float): Regularization parameter.
    - r (optional): Additional parameter for calculation.

    Returns:
    - tensor: Optimal W matrix.
    """
    if r != None:
        print('r is not None, haven\'t implemented yet')
        sys.exit(1)
    compressed_Y = phi @ Y

    output = torch.linalg.lstsq(X.T,compressed_Y.T).solution.T
    # output = compressed_Y @ torch.linalg.pinv(X)
    return output, compressed_Y

def auto_calculate_s(Y):
    """
    Automatically calculates the sparsity level.

    Parameters:
    - Y (tensor): Signal matrix.

    Returns:
    - int: Sparsity level.
    """
    avg = torch.count_nonzero(Y) / Y.shape[1]
    print('auto_calculate_s:',avg)
    return int(avg) + 1 if avg % 1 != 0 else int(avg)

def preprocessing_subg(filename='synt_data/gaussian_0.032_20240727091135_100_300_200', device=torch.device("cpu")):
    X, snr, timing, d, n, k = load_data_subg(filename, dtype='data')
    X = X.to(device)
    Y = load_data_subg(filename, dtype='label').to(device)
    X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2)
    X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T
    s = auto_calculate_s(Y_train)
    return filename, X_train, X_test, Y_train, Y_test, s

def omp(X, y, s):
    """
    Performs Orthogonal Matching Pursuit.

    Parameters:
    - X (tensor): Dictionary matrix.
    - y (tensor): Signal vector.
    - s (int): Sparsity level.

    Returns:
    - tensor: Coefficients after performing OMP.
    """
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
    omp.fit(X.cpu(), y.cpu())
    coefficients = torch.tensor(omp.coef_).to(X.device).T
    return coefficients


def MOR_pgd(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'value'
    pgd_thres = None

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)

def MOR_pgd_positive(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'value'
    pgd_thres = 0


    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)

def MLC_pgd(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'binary'
    pgd_thres = 0.5

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)

def MLC_pgd_loose(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'binary'
    pgd_thres = 1e-3

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)


def MLC_pgd_LOOSE(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'binary'
    pgd_thres = None

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)

def MLC_pgd_val(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'value'
    pgd_thres = 0.5

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)


def MLC_pgd_valloose(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'value'
    pgd_thres = 1e-3

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)


def MLC_pgd_valLOOSE(phi, x, s, device, eta=0.9, T=60):
    """
    Performs projected gradient descent (PGD) optimization.

    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """

    project_mode = 'value'
    pgd_thres = None

    starting = time.time()
    V = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    V2 = torch.ones_like(V).to(device)
    for i in range(T):
        V1 = V - eta * phi.T @ (phi @ V - x)
        V = project_highest(V1, s, mode=project_mode,thres = pgd_thres)
        if i % 2 == 0:
            diff = torch.linalg.norm(V - V2)
            nv = torch.linalg.norm(V)
            V2 = V.clone()
            if diff / (1e-6 + nv) < 1e-6:
                break

        if i % 10 == 2:
            print(f'{i+1}/ {T}')
            print(f'difference between V and V2: {diff**2/V.shape[1]}')
            print(f'square norm of V: {nv**2/V.shape[1]}')

        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return V, (i+1)

def pgd_fromfista(phi, x, s, device, eta=1, T=60, project_mode='value',pgd_thres = 0.5,fista_mode = 'fixed'):
    """
    Performs FISTA optimization.
    Args:
        phi (torch.Tensor): The phi tensor.
        x (torch.Tensor): The x tensor.
        s (int): The value of s.
        device (str): The device to perform the computation on.
        eta (float, optional): The learning rate. Defaults to 0.8.
        T (int, optional): The maximum number of iterations. Defaults to 60.
        project_mode (str, optional): The mode for projection. Defaults to 'value'.

    Returns:
        torch.Tensor: The optimized tensor V.
        int: The number of iterations performed.
    """
    project_mode = 'value'
    pgd_thres = 0

    starting = time.time()
    Y1 = torch.zeros((phi.shape[1], x.shape[1])).to(device)
    X1 = torch.zeros_like(Y1).to(device)
    X0 = torch.zeros_like(Y1).to(device)
    Y0 = torch.zeros_like(Y1).to(device)
    t = torch.tensor(1).to(device)
    for i in range(T):
        X0 = X1.clone()
        X1 = Y1 - eta * phi.T @ (phi @ Y1 - x)
        X1 = project_highest(X1, s, mode=project_mode,thres = pgd_thres)
        Y1 = X1 + (t-1)/((1+torch.sqrt(1+4*t**2))/2)*(X1-X0)
        t = (1+torch.sqrt(1+4*t**2))/2
        
        if i % 2 == 0:
            diff = torch.linalg.norm(Y0 - Y1)
            nv = torch.linalg.norm(Y1)
            Y0 = Y1.clone()
            if diff / (1e-6 + nv) < 1e-3:
                break



        print(f'time: {time.time()-starting}', f'iteration: {i+1}')
        starting = time.time()
    return Y1, (i+1)




def compression(X_train, Y_train, s,m = None, compression_delta = 0.3, compression_epsilon = 0.1,device = torch.device("cpu")):
    """
    Compresses the training data using a compression matrix and calculates the compressed training loss.

    Args:
        X_train (torch.Tensor): The input training data.
        Y_train (torch.Tensor): The target training data.
        s (int): The number of columns in the compression matrix.
        compression_delta (float, optional): The delta parameter for the compression matrix. Defaults to 0.3.
        compression_epsilon (float, optional): The epsilon parameter for the compression matrix. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the compression matrix, the compressed weight matrix, and the compressed training loss.
    """
    phi = compression_matrix(compression_s=  s, k=Y_train.shape[0], delta = compression_delta, epsilon = compression_epsilon, mode='gaussian',m = m)
    
    # W_hat,compressed_Y_train = calculate_Wopt(X_train, Y_train, phi)
    phi = phi.numpy()
    compressed_Y_train = phi@Y_train
    compressed_Y_train = torch.tensor(compressed_Y_train,dtype=torch.float32).to(device)
    # print('shape of X_train and Compressed_Y_train:',X_train.shape,compressed_Y_train.shape)
    W_hat = torch.linalg.lstsq(X_train.T,compressed_Y_train.T).solution.T
    del compressed_Y_train, X_train
    phi = torch.tensor(phi,dtype=torch.float32).to(device)
    compressed_training_loss = 0

    # compressed_training_loss = torch.norm(compressed_Y_train - W_hat @ X_train) ** 2
    return phi, W_hat, compressed_training_loss

def prediction(W_hat,X_test,Y_test,phi,s,reg_method = 'CD', prediction_mode = 'value',prediction_thres = 0,EN_alpha = 0.1, EN_l1_ratio = 0.5):
    WX = W_hat @ X_test
    print('wx.device:',WX.device)
    prediction_time_start = datetime.datetime.now()
    if reg_method == 'OMP':
        res = omp(phi,WX,s)
        others = {}

    elif reg_method == 'CD':
        res = phi.T@WX
        others = {}
        
    elif reg_method == 'EN':
        regr = ElasticNet(alpha=EN_alpha, l1_ratio=EN_l1_ratio,fit_intercept=False,max_iter=10000)
        regr.fit(phi.cpu().numpy(),WX.cpu().numpy())
        res = torch.tensor(regr.coef_.T,dtype=torch.float32).to(X_test.device)
        others = {'alpha':EN_alpha,'l1_ratio':EN_l1_ratio}

    elif reg_method == 'PGD_MOR':
        res,iter_num = MOR_pgd(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    elif reg_method == 'PGD_MOR_positive':
        res,iter_num = MOR_pgd_positive(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    
    elif reg_method == 'PGD_MLC':
        res,iter_num = MLC_pgd(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    elif reg_method == 'PGD_MLC_loose':
        res,iter_num = MLC_pgd_loose(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    elif reg_method == 'PGD_MLC_LOOSE':
        res,iter_num = MLC_pgd_LOOSE(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    elif reg_method == 'PGD_MLC_val':
        res,iter_num = MLC_pgd_val(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    elif reg_method == 'PGD_MLC_valloose':
        res,iter_num = MLC_pgd_valloose(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}
    elif reg_method == 'PGD_MLC_valLOOSE':
        res,iter_num = MLC_pgd_valLOOSE(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}

    # elif reg_method == 'FISTA':
    #     res,iter_num = fista(phi,WX,s,X_test.device,project_mode = project_mode)
    #     others = {'iter_num':iter_num}
    elif reg_method == 'PGD_from_FISTA':
        res,iter_num = pgd_fromfista(phi,WX,s,X_test.device)
        others = {'iter_num':iter_num}

    else:
        print('Method not found')
        return None
    res = project_highest(res,s,mode=prediction_mode, thres = prediction_thres)

    prediction_time_end = datetime.datetime.now()
    prediction_time = prediction_time_end - prediction_time_start
    precision = torch.sum((res > 0) & (Y_test > 0)) / (torch.sum(res > 0))
    print('total number of positive:',torch.sum(res>0),torch.sum(Y_test > 0))
    others['num_positive'] = torch.sum(res>0)
    others['num_positive_true'] = torch.sum(Y_test > 0)
    output_distance = torch.norm(Y_test - res) ** 2/Y_test.shape[1]
    prediction_loss = torch.norm(phi@res - WX) ** 2/WX.shape[1]
    print('precision:',precision,'output_distance:',output_distance,'prediction_loss:',prediction_loss)
    print('prediction time:',prediction_time)
    return precision, output_distance, prediction_loss,others, prediction_time

def store_res(filename,precision_list, output_distance_list, prediction_loss_list,others_list, cs_training_loss_list, training_loss, prediction_time_list, phi_time_list):
    if not os.path.exists('results'):
        os.mkdir('results')
    with open(f'results/{filename}.txt','w+') as f:
        f.write(f'Precision: {precision_list}\n')
        f.write('\n')
        f.write(f'Output distance: {output_distance_list}\n')
        f.write('\n')
        f.write(f'Prediction loss: {prediction_loss_list}\n')
        f.write('\n')
        f.write(f'Others: {others_list}\n')
        f.write('\n')
        f.write(f'Compressed training loss: {cs_training_loss_list}\n')
        f.write('\n')
        f.write(f'Training loss: {training_loss}\n')
        f.write('\n')
        f.write(f'Prediction time: {prediction_time_list}\n')
        f.write('\n')
        f.write(f'Phi time: {phi_time_list}\n')
        f.write('\n')
    print('Results stored')

def special_923(filename, device):
    filename, X_train, X_test, Y_train, Y_test, compression_s = preprocessing_subg(filename, device)
    Z = load_data_subg(filename, dtype='Z').to(device)
    return filename, X_train, X_test, Y_train, Y_test, Z, compression_s



def synt_func(filename = 'synt_data/gaussian_0.032_20240727091135_100_300_200',X_train = None, X_test = None, Y_train = None, Y_test = None,
              device = torch.device("cpu"),prediction_s = 1,compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10, m = None,
                     predicting_data = 'test_data',project_mode = 'value',reg_method = 'CD',EN_alpha = 0.1, EN_l1_ratio = 0.5, prediction_thres = 0):
    if (filename == None)&(Y_train != None):
        compression_s = auto_calculate_s(Y_train)
        print('s:',compression_s)
    elif (filename != None)&(Y_train == None):
        filename, X_train, X_test, Y_train, Y_test, compression_s = preprocessing_subg(filename, device)
    else:
        print('Data not found')
        return None
    Z_hat = torch.linalg.lstsq(X_train.T,Y_train.T).solution.T

    ## calculate loss
    training_loss = torch.norm(Y_train - Z_hat @ X_train) ** 2
    cs_training_loss_list = []
    output_list = []

    for _ in range(phi_iter):
        phi_time_start = datetime.datetime.now()
        phi, W_hat, compressed_training_loss = compression(X_train = X_train,  Y_train = Y_train, s = compression_s, device=device,
                                                           compression_delta = compression_delta, compression_epsilon = compression_epsilon, m = m)
        phi = phi.to(device)
        phi_time_end = datetime.datetime.now()
        phi_time = phi_time_end - phi_time_start
        cs_training_loss_list.append(compressed_training_loss)
        if predicting_data == 'training_data':
            precision, output_distance, prediction_loss,others,prediction_time = prediction(W_hat,X_train,Y_train,phi,prediction_s,reg_method = reg_method,prediction_mode = project_mode,EN_alpha = EN_alpha, EN_l1_ratio = EN_l1_ratio,prediction_thres = prediction_thres) 
        elif predicting_data == 'test_data':
            precision, output_distance, prediction_loss,others,prediction_time = prediction(W_hat,X_test,Y_test,phi,prediction_s,reg_method = reg_method,prediction_mode=  project_mode,EN_alpha = EN_alpha, EN_l1_ratio = EN_l1_ratio,prediction_thres = prediction_thres)
        else:
            print('Prediction mode not found')
            return None
        output_list.append((precision, output_distance, prediction_loss,others,prediction_time,phi_time))
    precision_list = [x[0] for x in output_list]
    output_distance_list = [x[1] for x in output_list]
    prediction_loss_list = [x[2] for x in output_list]
    others_list = [x[3] for x in output_list]
    prediction_time_list = [x[4] for x in output_list]
    phi_time_list = [x[5] for x in output_list]
    print(filename)
    res_filename = filename.split('/')[-1]+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+f'_s{prediction_s}_m{m}_{predicting_data}_reg_{reg_method}_project_{project_mode}+prediction_thres_{prediction_thres}_EN_alpha_{EN_alpha}_EN_l1_ratio_{EN_l1_ratio}'
    store_res(res_filename,precision_list, output_distance_list, prediction_loss_list,others_list, cs_training_loss_list, training_loss, prediction_time_list, phi_time_list)
    return precision_list
    


def MLC_func(filename = None,X_train = None, X_test = None, Y_train = None, Y_test = None,
              device = torch.device("cpu"),prediction_s = 1, compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10, m = None,
                     predicting_data = 'test_data',project_mode = 'binary',reg_method = 'CD',mlc_name = 'mlc',EN_alpha = 0.1, EN_l1_ratio = 0.5,prediction_thres = 0.5):
    if (filename == None)&(Y_train != None):
        # compression_s = auto_calculate_s(Y_train)
        # print('suggested s:',compression_s)
        compression_s = 19
    elif (filename != None)&(Y_train == None):
        filename, X_train, X_test, Y_train, Y_test, compression_s = preprocessing_subg(filename, device)
    else:
        print('Data not found')
        return None

    training_loss = 0
    cs_training_loss_list = []
    output_list = []

    for _ in range(phi_iter):
        phi_time_start = datetime.datetime.now()
        phi, W_hat, compressed_training_loss = compression(X_train=X_train, Y_train=Y_train, s=compression_s, device = device,
                                                           compression_delta=compression_delta, compression_epsilon=compression_epsilon, m = m)
        
        phi = phi.to(device)
        phi_time_end = datetime.datetime.now()
        phi_time = phi_time_end - phi_time_start
        cs_training_loss_list.append(compressed_training_loss)

        X_test_c,Y_test_c = X_test.to(device),Y_test.to(device)

        if predicting_data == 'training_data':
            precision, output_distance, prediction_loss,others,prediction_time = prediction(W_hat,X_train,Y_train,phi,prediction_s,reg_method = reg_method,prediction_mode = project_mode,EN_alpha = EN_alpha, EN_l1_ratio = EN_l1_ratio,prediction_thres = prediction_thres)
        elif predicting_data == 'test_data':
            precision, output_distance, prediction_loss,others,prediction_time = prediction(W_hat,X_test_c,Y_test_c,phi,prediction_s,reg_method = reg_method,prediction_mode = project_mode,EN_alpha = EN_alpha, EN_l1_ratio = EN_l1_ratio,prediction_thres = prediction_thres)
        else:
            print('Prediction mode not found')
            return None
        output_list.append((precision, output_distance, prediction_loss,others,prediction_time,phi_time))
        del phi, W_hat,X_test_c,Y_test_c
    precision_list = [x[0] for x in output_list]
    output_distance_list = [x[1] for x in output_list]
    prediction_loss_list = [x[2] for x in output_list]
    others_list = [x[3] for x in output_list]
    prediction_time_list = [x[4] for x in output_list]
    phi_time_list = [x[5] for x in output_list]
    print(filename)
    res_filename = mlc_name+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+f'_s{prediction_s}_m{m}_{predicting_data}_reg_{reg_method}_project_{project_mode}+prediction_thres_{prediction_thres}_EN_alpha_{EN_alpha}_EN_l1_ratio_{EN_l1_ratio}'
    store_res(res_filename,precision_list, output_distance_list, prediction_loss_list,others_list, cs_training_loss_list, training_loss, prediction_time_list, phi_time_list)
    return precision_list

    
# %%
