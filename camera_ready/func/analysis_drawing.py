# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import torch,datetime,numpy
import os

def combing_results(res_lst,name):
    res_dict = {}
    res_dict[name] = [res for res in res_lst]
    return res_dict



def draw_m_value(res_dict,x_lst,s,name,title,figname,lines = False):
    markers = ['o','s','^','x','*','+']
    color_lst = [ 'C0','C1','C3','C4','C5','C6']

    plt.plot()
    i = 0
    for res_name,res_val in res_dict.items():
        mean_lst = []
        std_lst = []
        for values in res_val: 
            mean = np.mean(values)
            std = np.std(values)
            mean_lst.append(mean)
            std_lst.append(std)
        plt.plot(x_lst,mean_lst,label = res_name,marker = markers[i],color = color_lst[i])
        plt.grid(visible=True)
        plt.fill_between(x_lst, np.array(mean_lst)-np.array(std_lst), np.array(mean_lst)+np.array(std_lst), alpha=0.2, color = color_lst[i])
        i = i + 1
        plt.legend()
        if lines:
            plt.axhline(y=1, linestyle='dotted', color='green')
        
        plt.xlabel('Number of rows')
        plt.ylabel(name)
        plt.xlim(left = 0)
    plt.title(f'{name} when s = {s}'+title)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, 'drawing', figname+name+str(s)+'.png'))
    plt.show()

def load_fista(filename):
    res_dict = {}
    with open(filename) as f:
        for line in f:
            name = line.split(':')[0]
            if name in ['Precision','Output distance','Prediction loss','Compressed training loss']:
                val = line.split(':')[1]
                values = val.strip().strip('[]').split(', ')
                values = [float(v.split('tensor(')[1].strip(')')) for v in values]
                res_dict[name] = values
            elif name == 'Prediction time':
                val = line.split(':')[1]
                time_strings = val.strip().strip('[]').split('datetime.timedelta(seconds=')
                times = []
                for t in time_strings[1:]:
                    # Extract seconds and microseconds from the string
# Split the string to isolate the 'seconds' and 'microseconds' parts
                    parts = t.split(', microseconds=')
                    print(parts)
                    seconds = int(parts[0])
                    microseconds = int(parts[1].split(')')[0])
                    # seconds = int(parts[0])
                    # microseconds = int(parts[1])                   # Create timedelta object
                    td = datetime.timedelta(seconds=seconds, microseconds=microseconds).total_seconds()
                    times.append(td)
                res_dict[name] = times
    return res_dict


import re
from typing import Dict, List, Union

def parse_time_delta(time_str: str) -> float:
    total_seconds = 0.0
    
    # Extract time components using more specific patterns
    # Use word boundaries \b to ensure we match exact unit names
    patterns = [
        (r'\bdays=(\d+)', 86400),           # days to seconds
        (r'\bhours=(\d+)', 3600),           # hours to seconds
        (r'\bminutes=(\d+)', 60),           # minutes to seconds
        (r'\bmicroseconds=(\d+)', 0.000001),  # microseconds to seconds
        (r'\bmilliseconds=(\d+)', 0.001),   # milliseconds to seconds
        (r'(?<!micro)\bseconds=(\d+)', 1),  # seconds (negative lookbehind for micro)
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, time_str)
        if match:
            value = int(match.group(1))
            total_seconds += value * multiplier
    
    return total_seconds





def extract_tensor_values(line: str) -> List[float]:
    values = re.findall(r'tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
    return [float(v) for v in values]

def parse_others_dict(others_str: str) -> Dict[str, List[Union[int, float]]]:
    result_dict = {
        'iter_num': [],
        'num_positive': [],
        'num_positive_true': []
    }
    
    dict_matches = re.findall(r'{[^}]+}', others_str)
    
    for dict_str in dict_matches:
        iter_match = re.search(r"'iter_num': (\d+)", dict_str)
        if iter_match:
            result_dict['iter_num'].append(int(iter_match.group(1)))
        
        tensor_matches = re.findall(r"'(num_positive(?:_true)?)':\s*tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", dict_str)
        for key, value in tensor_matches:
            result_dict[key].append(float(value))
    
    return result_dict

def parse_data_file(file_content: str) -> Dict[str, List[Union[int, float]]]:
    data_dict = {}
    
    lines = file_content.strip().split('\n\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        if key == 'Others':
            others_data = parse_others_dict(value)
            data_dict.update(others_data)
        elif key in ['Prediction time', 'Phi time']:
            time_list = re.findall(r'datetime\.timedelta\([^)]+\)', value)
            data_dict[key] = [parse_time_delta(t) for t in time_list]
        elif key == 'Training loss':
            try:
                data_dict[key] = float(value)
            except ValueError:
                match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', value)
                if match:
                    data_dict[key] = float(match.group(1))
        else:
            data_dict[key] = extract_tensor_values(value)
    
    return data_dict





# #%%
# parent_dir = os.path.dirname(os.getcwd())

# # Specify the folder you want to open
# eur_results = os.path.join(parent_dir, 'results/eur_results')
# filelist = os.listdir(eur_results)
# res_dict = []
# filename = filelist[0]
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# file_path = os.path.join(parent_dir, 'results/eur_results/' + filename)
# print(file_path)

# with open(file_path, 'r') as file:
#     file_content = file.read()

# # Parse the data
# result = parse_data_file(file_content)


# for filename in filelist:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     file_path = os.path.join(parent_dir, 'results/eur_results/' + filename)
#     print(file_path)
#     res_dict.append()

#%% Synt_data with snr = 1
# methods_lst = ['PGD_MOR_positive','CD','PGD_from_FISTA']
if '__name__' == '__main__':
    methods_lst = ['PGD_MOR_positive']
    s_lst = ['s1','s3']
    m_lst = ['m100_','m300','m500','m700','m1000','m2000']
    snr1_results_dictionary = {}
    for method in methods_lst:
        snr1_results_dictionary[method] = {}
        for s in s_lst:
            snr1_results_dictionary[method][s] = {}
            for m in m_lst:
                snr1_results_dictionary[method][s][m] = {}

    parent_dir = os.path.dirname(os.getcwd())
    snr1_results = os.path.join(parent_dir, 'results/snr1_results')
    filelist = os.listdir(snr1_results)
    res_dict = []
    for filename in filelist:
        if filename == '.DS_Store':
            continue
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/snr1_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                snr1_results_dictionary[method][s][m] = result
                                break
                        break
                break


    # criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss','Ratio']
    criterion_lst = ['Prediction time']
    s = 3
    for criterion in criterion_lst:
        name_criterion = criterion+' vs number of rows '
        if criterion == 'Ratio':
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    list1 = snr1_results_dictionary[method]['s'+str(s)][m]['Compressed training loss']
                    train_loss = snr1_results_dictionary[method]['s'+str(s)][m]['Training loss']
                    true_precision = [item/train_loss for item in list1]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 1','snr1_results_'+criterion,lines = True)

        elif criterion == 'Precision':
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    list1 = snr1_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = snr1_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_snr1 = 6000
                    true_precision = [(a*b)/(n_snr1*s) for a, b in zip(list1, list2)]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 1','snr1_results_'+criterion)
        else:
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                        res_dict[name_method].append([])
                    else:
                        res_dict[name_method].append(snr1_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 1','snr1_results_'+criterion)


    #%% Synt_data with snr = 0.32
    methods_lst = ['PGD_MOR_positive','CD','PGD_from_FISTA']
    s_lst = ['s1','s3']
    m_lst = ['m100_','m300','m500','m700','m1000','m2000']
    snr032_results_dictionary = {}
    for method in methods_lst:
        snr032_results_dictionary[method] = {}
        for s in s_lst:
            snr032_results_dictionary[method][s] = {}
            for m in m_lst:
                snr032_results_dictionary[method][s][m] = {}

    parent_dir = os.path.dirname(os.getcwd())
    snr1_results = os.path.join(parent_dir, 'results/snr032_results')
    filelist = os.listdir(snr1_results)
    res_dict = []
    for filename in filelist:
        if filename == '.DS_Store':
            continue
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/snr032_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                snr032_results_dictionary[method][s][m] = result
                                break
                        break
                break


    criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss','Ratio']
    s = 3
    for criterion in criterion_lst:
        name_criterion = criterion+' vs number of rows '
        if criterion == 'Ratio':
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    list1 = snr032_results_dictionary[method]['s'+str(s)][m]['Compressed training loss']
                    train_loss = snr032_results_dictionary[method]['s'+str(s)][m]['Training loss']
                    true_precision = [item/train_loss for item in list1]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 0.32','snr032_results_'+criterion,lines = True)

        elif criterion == 'Precision':
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    list1 = snr032_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = snr032_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_snr1 = 6000
                    true_precision = [(a*b)/(n_snr1*s) for a, b in zip(list1, list2)]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 0.32','snr032_results_'+criterion)
        else:
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                        res_dict[name_method].append([])
                    else:
                        res_dict[name_method].append(snr032_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 0.32','snr032_results_'+criterion)



    #%% Synt_data with snr = 0.032
    methods_lst = ['PGD_MOR_positive','CD','PGD_from_FISTA']
    s_lst = ['s1','s3']
    m_lst = ['m100_','m300','m500','m700','m1000','m2000']
    snr0032_results_dictionary = {}
    for method in methods_lst:
        snr0032_results_dictionary[method] = {}
        for s in s_lst:
            snr0032_results_dictionary[method][s] = {}
            for m in m_lst:
                snr0032_results_dictionary[method][s][m] = {}

    parent_dir = os.path.dirname(os.getcwd())
    snr1_results = os.path.join(parent_dir, 'results/snr0032_results')
    filelist = os.listdir(snr1_results)
    res_dict = []
    for filename in filelist:
        if filename == '.DS_Store':
            continue
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/snr0032_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                snr0032_results_dictionary[method][s][m] = result
                                break
                        break
                break


    criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss','Ratio']
    s = 3
    for criterion in criterion_lst:
        name_criterion = criterion+' vs number of rows '
        if criterion == 'Ratio':
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    list1 = snr0032_results_dictionary[method]['s'+str(s)][m]['Compressed training loss']
                    train_loss = snr0032_results_dictionary[method]['s'+str(s)][m]['Training loss']
                    true_precision = [item/train_loss for item in list1]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 0.032','snr0032_results_'+criterion,lines = True)

        elif criterion == 'Precision':
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    list1 = snr0032_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = snr0032_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_snr1 = 6000
                    true_precision = [(a*b)/(n_snr1*s) for a, b in zip(list1, list2)]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 0.032','snr0032_results_'+criterion)
        else:
            res_dict = {}
            for method in methods_lst:
                if method == 'PGD_MOR_positive':
                    name_method = 'Prop.Algo'
                elif method == 'PGD_from_FISTA':
                    name_method = 'FISTA'
                else:
                    name_method = method

                res_dict[name_method] = []
                for m in m_lst:
                    if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                        res_dict[name_method].append([])
                    else:
                        res_dict[name_method].append(snr0032_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,300,500,700,1000,2000],s,name_criterion,', 1/SNR = 0.032','snr0032_results_'+criterion)


    #%% EUR from synt

    # methods_lst = ['PGD_MLC_LOOSE','OMP','CD','PGD_from_FISTA']
    methods_lst = ['PGD_MLC_LOOSE']
    s_lst = ['s1','s3']
    m_lst = ['m100_','m200_','m300','m400','m500','m700','m1000','m2000']

    eur_results_dictionary = {}
    for method in methods_lst:
        eur_results_dictionary[method] = {}
        for s in s_lst:
            eur_results_dictionary[method][s] = {}
            for m in m_lst:
                eur_results_dictionary[method][s][m] = {}

    parent_dir = os.path.dirname(os.getcwd())
    snr1_results = os.path.join(parent_dir, 'results/eur_results')
    filelist = os.listdir(snr1_results)
    res_dict = []
    for filename in filelist:
        if filename == '.DS_Store':
            continue
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/eur_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                eur_results_dictionary[method][s][m] = result
                                break
                        break
                break


    # criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss']
    criterion_lst = ['Prediction time','iter_num']
    s = 3
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
                for m in m_lst:
                    list1 = eur_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = eur_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_snr1 = 3809
                    true_precision = [(a*b)/(n_snr1*s) for a, b in zip(list1, list2)]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,name_criterion,', Eurlex','eurlex_results_'+criterion)
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
                for m in m_lst:
                    # if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                    #     res_dict[name_method].append([])
                    # else:
                    res_dict[name_method].append(eur_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,name_criterion,', Eurlex','eurlex_results_'+criterion)







    # %% EUR original

    methods_lst = ['PGD_MLC_LOOSE','OMP','CD','PGD_from_FISTA']
    s_lst = ['s1','s3']
    m_lst = ['m100_','m200_','m300','m400','m500','m700','m1000','m2000']

    eur_results_dictionary = {}
    for method in methods_lst:
        eur_results_dictionary[method] = {}
        for s in s_lst:
            eur_results_dictionary[method][s] = {}
            for m in m_lst:
                eur_results_dictionary[method][s][m] = {}



    parent_dir = os.path.dirname(os.getcwd())
    eur_results = os.path.join(parent_dir, 'results/eur_results')
    filelist = os.listdir(eur_results)
    res_dict = []
    for filename in filelist:
        if filename == '.DS_Store':
            continue
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/eur_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                eur_results_dictionary[method][s][m] = result
                                break
                        break
                break
        

    criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss']
    s = 1
    for criterion in criterion_lst:
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
                for m in m_lst:
                    list1 = eur_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = eur_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_eur = 3809
                    true_precision = [(a*b)/(n_eur*s) for a, b in zip(list1, list2)]

                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,criterion,' for Eurlex','eur_results_'+criterion+'_s1')
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
                for m in m_lst:
                    # if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                    #     res_dict[name_method].append([])
                    # else:
                    res_dict[name_method].append(eur_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,criterion,' for EurLex','eur_results_'+criterion+'_s1')




    #%% WIKI from synt

    methods_lst = ['PGD_MLC_LOOSE','CD','PGD_from_FISTA']
    s_lst = ['s1','s3','s5']
    m_lst = ['m100_','m200_','m300','m400','m500','m700','m1000','m2000']

    wiki_results_dictionary = {}
    for method in methods_lst:
        wiki_results_dictionary[method] = {}
        for s in s_lst:
            wiki_results_dictionary[method][s] = {}
            for m in m_lst:
                wiki_results_dictionary[method][s][m] = {}

    parent_dir = os.path.dirname(os.getcwd())
    snr1_results = os.path.join(parent_dir, 'results/wiki_results')
    filelist = os.listdir(snr1_results)
    res_dict = []
    for filename in filelist:
        if filename == '.DS_Store':
            continue
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/wiki_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                wiki_results_dictionary[method][s][m] = result
                                break
                        break
                break


    # criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss']
    criterion_lst = ['Prediction time','iter_num']
    s = 3
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
                for m in m_lst:
                    list1 = wiki_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = wiki_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_snr1 = 6616 
                    true_precision = [(a*b)/(n_snr1*s) for a, b in zip(list1, list2)]
                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,name_criterion,', wiki','wiki_results_'+criterion)
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
                for m in m_lst:
                    # if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                    #     res_dict[name_method].append([])
                    # else:
                    res_dict[name_method].append(wiki_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,name_criterion,', wiki','wiki_results_'+criterion)






    # %% WIKI

    methods_lst = ['PGD_MLC_LOOSE','CD','PGD_from_FISTA']
    s_lst = ['s1','s3']
    m_lst = ['m100_','m200_','m300','m400','m500','m700','m1000','m2000']

    wiki_results_dictionary = {}
    for method in methods_lst:
        wiki_results_dictionary[method] = {}
        for s in s_lst:
            wiki_results_dictionary[method][s] = {}
            for m in m_lst:
                wiki_results_dictionary[method][s][m] = {}



    parent_dir = os.path.dirname(os.getcwd())
    wiki_results = os.path.join(parent_dir, 'results/wiki_results')
    filelist = os.listdir(wiki_results)
    res_dict = []
    for filename in filelist:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/wiki_results/' + filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
        result = parse_data_file(file_content)
        for method in methods_lst:
            if method in filename:
                for s in s_lst:
                    if s in filename:
                        for m in m_lst:
                            if m in filename:
                                wiki_results_dictionary[method][s][m] = result
                                break
                        break
                break
        


    #%%
    criterion_lst = ['Precision','Output distance','Prediction loss','iter_num','num_positive','num_positive_true','Prediction time','Phi time','Compressed training loss']
    s = 3
    for criterion in criterion_lst:
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
                for m in m_lst:
                    list1 = wiki_results_dictionary[method]['s'+str(s)][m][criterion]
                    list2 = wiki_results_dictionary[method]['s'+str(s)][m]['num_positive']
                    n_wiki = 6616
                    true_precision = [(a*b)/(n_wiki*s) for a, b in zip(list1, list2)]

                    res_dict[name_method].append(true_precision) 
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,criterion,' for wiki','wiki_results_'+criterion+'_s1')
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
                for m in m_lst:
                    # if method == 'PGD_from_FISTA' and (m == 'm100_' or m == 'm300'):
                    #     res_dict[name_method].append([])
                    # else:
                    res_dict[name_method].append(wiki_results_dictionary[method]['s'+str(s)][m][criterion])
            draw_m_value(res_dict,[100,200,300,400,500,700,1000,2000],s,criterion,' for wiki','wiki_results_'+criterion+'_s1')


    # Parse the data



    # fista_filelist = ['gaussian_1_20240802155843_10000_30000_20000_20240803201834_s3_m100_predicting_test_data_reg_FISTA_project_value.txt',
    #  'gaussian_1_20240802155843_10000_30000_20000_20240803205954_s3_m200_predicting_test_data_reg_FISTA_project_value.txt',
    #  'gaussian_1_20240802155843_10000_30000_20000_20240803211200_s3_m300_predicting_test_data_reg_FISTA_project_value.txt',
    #  'gaussian_1_20240802155843_10000_30000_20000_20240803212457_s3_m500_predicting_test_data_reg_FISTA_project_value.txt',
    #  'gaussian_1_20240802155843_10000_30000_20000_20240803213813_s3_m700_predicting_test_data_reg_FISTA_project_value.txt',
    #  'gaussian_1_20240802155843_10000_30000_20000_20240803195417_s3_m1000_predicting_test_data_reg_FISTA_project_value.txt']

    # fista_res_dict = []
    # for filename in fista_filelist:
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     parent_dir = os.path.dirname(current_dir)
    #     file_path = os.path.join(parent_dir, 'results/eur_results/'+filename)
    #     fista_res_dict.append(load_fista(file_path))

    # # combine precision
    # fista_precision = combing_results([res['Precision'] for res in fista_res_dict],'FISTA')
    # print(fista_precision)



    #%%
    fista_filelist = [
        'gaussian_0.032_20240802155025_10000_30000_20000_20240804011231_s3_m200_predicting_test_data_reg_FISTA_project_value.txt',
        'gaussian_0.032_20240802155025_10000_30000_20000_20240804012435_s3_m300_predicting_test_data_reg_FISTA_project_value.txt',
        'gaussian_0.032_20240802155025_10000_30000_20000_20240804013702_s3_m500_predicting_test_data_reg_FISTA_project_value.txt',
        'gaussian_0.032_20240802155025_10000_30000_20000_20240804014539_s3_m700_predicting_test_data_reg_FISTA_project_value.txt'
    ]
    fista_res_dict = []
    for filename in fista_filelist:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/'+filename)
        fista_res_dict.append(load_fista(file_path))

    # combine precision
    fista_precision = combing_results([res['Precision'] for res in fista_res_dict],'FISTA')
    print(fista_precision)

    # %%
    PGD_filelist =  ['gaussian_0.032_20240802155025_10000_30000_20000_20240804091751_s3_m200_predicting_test_data_reg_PGD_from_FISTA_project_value.txt',
                            'gaussian_0.032_20240802155025_10000_30000_20000_20240804092316_s3_m300_predicting_test_data_reg_PGD_from_FISTA_project_value.txt',
                            'gaussian_0.032_20240802155025_10000_30000_20000_20240804092830_s3_m500_predicting_test_data_reg_PGD_from_FISTA_project_value.txt',
                            'gaussian_0.032_20240802155025_10000_30000_20000_20240804093342_s3_m700_predicting_test_data_reg_PGD_from_FISTA_project_value.txt']


    PGD_res_dict = []
    for filename in PGD_filelist:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/'+filename)
        PGD_res_dict.append(load_fista(file_path))

    # combine precision
    PGD_precision = combing_results([res['Precision'] for res in PGD_res_dict],'PGD')
    print(PGD_precision)

    total_precision = {**fista_precision,**PGD_precision}
    draw_m_value(total_precision,[200,300,500,700],3,'Precision',',      1/SNR = 0.032','gaussian_0.032_20240802155025')
    # %%

    timepgd_filelist = ['gaussian_1_20240802155843_10000_30000_20000_20240803042210_s3_m100_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803024037_s3_m200_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803055944_s3_m300_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803073718_s3_m500_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803091432_s3_m700_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803150717_s3_m1000_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803151323_s3_m3000_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240803152113_s3_m5000_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240806000915_s3_m7000_predicting_test_data_reg_PGD_project_value.txt',
                    'gaussian_1_20240802155843_10000_30000_20000_20240806002116_s3_m10000_predicting_test_data_reg_PGD_project_value.txt']


    PGD_res_dict = []
    for filename in timepgd_filelist:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        file_path = os.path.join(parent_dir, 'results/'+filename)
        PGD_res_dict.append(load_fista(file_path))
    # combine time
    PGD_time = combing_results([res['Prediction time'] for res in PGD_res_dict],'PGD')
    print(PGD_time)

    draw_m_value(PGD_time,[100,200,300,500,700,1000,3000,5000,7000,10000],3,'Prediction time',',        1/SNR = 1','gaussian_1_20240802155843'+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    # %%
