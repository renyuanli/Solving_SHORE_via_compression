import numpy as np
import torch,os
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import warnings


def read_data(filename, header=True, dtype='float32', zero_based=True):
    """Read data in sparse format

    Arguments
    ---------
    filename: str
        output file name
    header: bool, default=True
        If header is present or not
    dtype: str, default='float32'
        data type of values
    zero_based: boolean, default=True
        zwero based indices?

    Returns
    --------
    features: csr_matrix
        features matrix
    labels: csr_matix
        labels matrix
    num_samples: int
        #instances
    num_feat: int
        #features
    num_labels: int
        #labels
    """
    with open(filename, 'rb') as f:
        _l_shape = None
        if header:
            line = f.readline().decode('utf-8').rstrip("\n")
            line = line.split(" ")
            num_samples, num_feat, num_labels = int(
                line[0]), int(line[1]), int(line[2])
            _l_shape = (num_samples, num_labels)
        else:
            num_samples, num_feat, num_labels = None, None, None
        features, labels = load_svmlight_file(f, n_features=num_feat, multilabel=True, zero_based=zero_based)
        labels = ll_to_sparse(
            labels, dtype=dtype, zero_based=zero_based, shape=_l_shape)
    return features, labels, num_samples, num_feat, num_labels



def ll_to_sparse(X, shape=None, dtype='float32', zero_based=True):
    """Convert a list of list to a csr_matrix; All values are 1.0
    Arguments:
    ---------
    X: list of list of tuples
        nnz indices for each row
    shape: tuple or none, optional, default=None
        Use this shape or infer from data
    dtype: 'str', optional, default='float32'
        datatype for data
    zero_based: boolean or "auto", default=True
        indices are zero based or not

    Returns:
    -------
    X: csr_matrix
    """
    indices = []
    indptr = [0]
    offset = 0
    for item in X:
        if len(item) > 0:
            indices.extend(item)
            offset += len(item)
        indptr.append(offset)
    data = [1.0]*len(indices)
    _shape = gen_shape(indices, indptr, zero_based)
    if shape is not None:
        assert _shape[0] <= shape[0], "num_rows_inferred > num_rows_given"
        assert _shape[1] <= shape[1], "num_cols_inferred > num_cols_given"
        indptr = expand_indptr(_shape[0], shape[0], indptr)
    return csr_matrix(
        (np.array(data, dtype=dtype), np.array(indices), np.array(indptr)),
        shape=shape)

def gen_shape(indices, indptr, zero_based=True):
    _min = min(indices)
    if not zero_based:
        indices = list(map(lambda x: x-_min, indices))
    num_cols = max(indices)
    num_rows = len(indptr) - 1
    return (num_rows, num_cols)


def expand_indptr(num_rows_inferred, num_rows, indptr):
    """Expand indptr if inferred num_rows is less than given
    """
    _diff = num_rows - num_rows_inferred
    if _diff > 0:  # Fix indptr as per new shape
        # Data is copied here
        warnings.warn("Header mis-match from inferred shape!")
        return np.concatenate((indptr, np.repeat(indptr[-1], _diff)))
    elif _diff == 0:  # It's fine
        return indptr
    else:
        raise NotImplementedError("Unknown behaviour!")
 
# def read_eur_train():
#     filename = 'data/real_data/eurlex/eurlex_train.txt'
#     features, labels, num_samples, num_feat, num_labels = read_data(filename)
#     features,labels = torch.tensor(features.toarray(),dtype=torch.float32).T, labels.transpose()
#     return features, labels, num_samples, num_feat, num_labels

def read_eur_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, '..', 'data', 'real_data', 'eurlex', 'eurlex_train.txt')
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features, labels = torch.tensor(features.toarray(), dtype=torch.float32).T, labels.transpose()
    return features, labels, num_samples, num_feat, num_labels

def read_eur_test():
    # filename = 'data/real_data/eurlex/eurlex_test.txt'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, '..', 'data', 'real_data', 'eurlex', 'eurlex_test.txt')
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = torch.tensor(features.toarray(),dtype=torch.float32).T, torch.tensor(labels.toarray(),dtype=torch.float32).T
    return features, labels, num_samples, num_feat, num_labels

def read_wiki_train():
    filename = 'data/real_data/wiki10/train.txt'
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = torch.tensor(features.toarray(),dtype=torch.float32).T, labels.transpose()
    # features,labels = torch.tensor(features.transpose()), torch.tensor(labels.transpose())
    return features, labels, num_samples, num_feat, num_labels

def read_wiki_test():
    filename = 'data/real_data/wiki10/test.txt'
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = torch.tensor(features.toarray(),dtype=torch.float32).T, torch.tensor(labels.toarray(),dtype=torch.float32).T
    # features,labels = torch.tensor(features.transpose()), torch.tensor(labels.transpose())
    return features, labels, num_samples, num_feat, num_labels

# def read_wiki_train1():
#     filename = 'data/real_data/wiki10/train.txt'
#     features, labels, num_samples, num_feat, num_labels = read_data(filename)
#     features,labels = features.transpose(), labels.transpose()
#     return features, labels, num_samples, num_feat, num_labels

# def read_wiki_test1():
#     filename = 'data/real_data/wiki10/test.txt'
#     features, labels, num_samples, num_feat, num_labels = read_data(filename)
#     features,labels = features.transpose(), labels.transpose()
#     return features, labels, num_samples, num_feat, num_labels

def read_ama_train1():
    filename = 'data/real_data/amazon131k/train.txt'
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = features.transpose()[:,:100000], labels.transpose()[:,:100000]
    return features, labels, num_samples, num_feat, num_labels

def read_ama_train2():
    filename = 'data/real_data/amazon131k/train.txt'
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = features.transpose()[:,:200000], labels.transpose()[:,:200000]
    return features, labels, num_samples, num_feat, num_labels

def read_ama_train3():
    filename = 'data/real_data/amazon131k/train.txt'
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = features.transpose(), labels.transpose()
    return features, labels, num_samples, num_feat, num_labels

def read_ama_test():
    filename = 'data/real_data/amazon131k/test.txt'
    features, labels, num_samples, num_feat, num_labels = read_data(filename)
    features,labels = features.transpose(), labels.transpose()
    return features, labels, num_samples, num_feat, num_labels
    
# def load_res_eur():
#     file_lst = ['resultseur/sol_pgd_reg_loose2.pt','resultseur/sol_omp2.pt','resultseur/sol_cd2.pt']
#     name_lst = ['Prop.Algo.','OMP','CD']
#     res = []
#     for i in range(len(file_lst)):
#         res.append(torch.load(file_lst[i]))
#     return res,name_lst


# def load_res_delic():
#     file_lst = ['results/delic_sol_pgd_reg_loose3.pt','results/delic_sol_omp3.pt','results/delic_sol_cd3.pt']
#     name_lst = ['Prop.Algo.','OMP','CD']
#     res = []
#     for i in range(len(file_lst)):
#         res.append(torch.load(file_lst[i]))
#     return res,name_lst

# def load_res_wiki():
#     file_lst = ['results/wiki_sol_pgd_reg_loose2.pt','results/wiki_sol_omp2.pt','results/wiki_sol_cd2.pt']
#     name_lst = ['Prop.Algo.','OMP','CD']
#     res = []
#     for i in range(len(file_lst)):
#         res.append(torch.load(file_lst[i]))
#     return res,name_lst

# def load_res_wiki2():
#     file_lst = ['results/wiki_sol_pgd_reg_loose3.pt','results/wiki_sol_omp3.pt','results/wiki_sol_cd3.pt']
#     name_lst = ['Prop.Algo.','OMP','CD']
#     res = []
#     for i in range(len(file_lst)):
#         res.append(torch.load(file_lst[i]))
#     return res,name_lst







            

    