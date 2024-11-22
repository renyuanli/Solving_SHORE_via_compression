### Test on real data Eurlex ###
#%%
import sys
import os
import torch,datetime,numpy

starting_time = datetime.datetime.now()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)


import func.real_data as real_data
import func.compression_func as comp_func
from func.analysis_drawing import draw_m_value,parse_data_file



eur_train_x, eur_train_y, eur_train_n, eur_train_d, eur_train_k = real_data.read_eur_train()
eur_test_x, eur_test_y, eur_test_n, eur_test_d, eur_test_k = real_data.read_eur_test()

print('shape of x and y:', eur_train_x.shape, eur_train_y.shape)
print('n, d, k:', eur_train_n, eur_train_d, eur_train_k)
print('cuda:', torch.cuda.is_available())
cuda_time = datetime.datetime.now()
c_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
eur_train_x,eur_train_y = eur_train_x.to(c_device),eur_train_y


m_lst = [100,500]
for m in m_lst:
    for prediction_s in [1]:
        for reg_methods in ['PGD_MLC_LOOSE','CD','PGD_from_FISTA']:
            precision_list = comp_func.MLC_func(
                filename=None,
                X_train=eur_train_x,
                Y_train=eur_train_y,
                X_test=eur_test_x,
                Y_test=eur_test_y,
                prediction_s=prediction_s,
                device = c_device,
                m=m,
                phi_iter=1,
                reg_method= reg_methods,
                mlc_name= 'eur',
                project_mode='binary',
                prediction_thres=0,
                predicting_data='test_data'
            )

#%% EUR from synt

methods_lst = ['PGD_MLC_LOOSE','CD','PGD_from_FISTA']
s_lst = ['s1']
m_lst_name = ['m100_','m500']

eur_results_dictionary = {}
for method in methods_lst:
    eur_results_dictionary[method] = {}
    for s in s_lst:
        eur_results_dictionary[method][s] = {}
        for m in m_lst_name:
            eur_results_dictionary[method][s][m] = {}

# parent_dir = os.path.dirname(os.getcwd())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
snr1_results = os.path.join(parent_dir, 'results')

# snr1_results = os.path.join(parent_dir, 'nips2024/results/eur_results')
filelist = os.listdir(snr1_results)
print('snr1_results:', snr1_results)
res_dict = []
for filename in filelist:
    if filename == '.DS_Store':
        continue
    # parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(snr1_results, filename)
    with open(file_path, 'r') as file:
        file_content = file.read()
    result = parse_data_file(file_content)
    for method in methods_lst:
        if method in filename:
            for s in s_lst:
                if s in filename:
                    for m in m_lst_name:
                        if m in filename:
                            eur_results_dictionary[method][s][m] = result
                            break
                    break
            break


criterion_lst = ['Prediction time','Output distance','Precision']
s = 1
for criterion in criterion_lst:
    name_criterion = criterion+' vs number of rows '

    if criterion == 'Precision':
        res_dict = {}
        for method in methods_lst:
            if method == 'PGD_MLC_LOOSE':
                name_method = 'Prop.Algo'
            elif method == 'PGD_from_FISTA':
                name_method = 'FISTA'
            else:
                name_method = method

            res_dict[name_method] = []
            for m in m_lst_name:
                list1 = eur_results_dictionary[method]['s'+str(s)][m][criterion]
                list2 = eur_results_dictionary[method]['s'+str(s)][m]['num_positive']
                n_snr1 = 3809
                true_precision = [(a*b)/(n_snr1*s) for a, b in zip(list1, list2)]
                res_dict[name_method].append(true_precision) 
        draw_m_value(res_dict,m_lst,s,name_criterion,', Eurlex','eurlex_results_'+criterion)
    else:
        res_dict = {}
        for method in methods_lst:
            if method == 'PGD_MLC_LOOSE':
                name_method = 'Prop.Algo'
            elif method == 'PGD_from_FISTA':
                name_method = 'FISTA'
            else:
                name_method = method

            res_dict[name_method] = []
            for m in m_lst_name:
                res_dict[name_method].append(eur_results_dictionary[method]['s'+str(s)][m][criterion])
        draw_m_value(res_dict,m_lst,s,name_criterion,', Eurlex','eurlex_results_'+criterion)


