# %%
import sys
import os
import torch,datetime,numpy

starting_time = datetime.datetime.now()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import func.generation_synt as gen_synt
import func.real_data as real_data

import func.compression_func as comp_func






#%% Original Experiment

device = torch.device("cuda")
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=30, snr=0.032, project_mode='value', project_thres=0)

# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=30, snr=1, project_mode='value', project_thres=0)

# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=30, snr=0.32, project_mode='value', project_thres=0)

# gen_synt.generate_synt(d=100, n=300, k=200, s=3, it_num=30, snr=0.1, project_mode='value', project_thres=0)


#%%
# file_namelst = [
#     'gaussian_0.32_3_30_0_valueNone_20241001115448_10000_30000_20000',
#     'gaussian_1_3_30_0_valueNone_20241001130359_10000_30000_20000',
#     'gaussian_0.032_3_30_0_valueNone_20241001114740_10000_30000_20000'
# ]

# for filename in file_namelst:
#     for m in [100,300,500,700,1000,2000]:  #100, 300,1000
#         for prediction_s in [3]:
#             # alpha_list = [0.01]
#             # l1_ratio_list = [1]
#             # for alpha in alpha_list:
#             #     for l1_ratio in l1_ratio_list:
#             #         precision_list = comp_func.synt_func(filename = 'data/synt_data/'+filename,device = torch.device("cuda"),predicting_data='test_data',
#             #                                             reg_method='EN', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=m,prediction_s=prediction_s,EN_alpha=alpha,EN_l1_ratio=l1_ratio)

#             precision_list = comp_func.synt_func(filename = 'data/synt_data/'+filename,device = torch.device("cuda"),predicting_data='test_data',
#                                                     reg_method='PGD_MOR_positive', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=m,prediction_s=prediction_s)

#             # precision_list = comp_func.synt_func(filename = 'data/synt_data/'+filename,device = torch.device("cuda"),predicting_data='test_data',
#             #                                         reg_method='OMP', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=m,prediction_s=prediction_s)

#             precision_list = comp_func.synt_func(filename = 'data/synt_data/'+filename,device = torch.device("cuda"),predicting_data='test_data',
#                                                     reg_method='CD', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=m,prediction_s=prediction_s)

#             precision_list = comp_func.synt_func(filename = 'data/synt_data/'+filename,device = torch.device("cuda"),predicting_data='test_data',
#                                                     reg_method='PGD_from_FISTA', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=m,prediction_s=prediction_s)

#             precision_list = comp_func.synt_func(filename = 'data/synt_data/'+filename,device = torch.device("cuda"),predicting_data='test_data',
#                                                     reg_method='PGD_MOR', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=m,prediction_s=prediction_s)


#%%


# eur_train_x, eur_train_y, eur_train_n, eur_train_d, eur_train_k = real_data.read_eur_train()
# eur_test_x, eur_test_y, eur_test_n, eur_test_d, eur_test_k = real_data.read_eur_test()

# print('shape of x and y:', eur_train_x.shape, eur_train_y.shape)
# print('n, d, k:', eur_train_n, eur_train_d, eur_train_k)
# print('cuda:', torch.cuda.is_available())
# cuda_time = datetime.datetime.now()
# c_device = torch.device("cuda")
# eur_train_x,eur_train_y = eur_train_x.to(c_device),eur_train_y
# # eur_test_x,eur_test_y = eur_test_x.to(c_device),eur_test_y.to(c_device)
# alpha_list = [0.01 * (1000 ** (i/4)) for i in range(5)]
# l1_ratio_list = torch.arange(0.1,1,0.3)


# # for m in [100,200,300,400,500,700,1000,2000]: 
# for m in [100,200]:
#     for prediction_s in [1,3]:
#         for reg_methods in ['PGD_MLC_LOOSE','PGD_MLC_valLOOSE','OMP','CD','PGD_from_FISTA']:
#             precision_list = comp_func.MLC_func(
#                 filename=None,
#                 X_train=eur_train_x,
#                 Y_train=eur_train_y,
#                 X_test=eur_test_x,
#                 Y_test=eur_test_y,
#                 prediction_s=prediction_s,
#                 device = c_device,
#                 m=m,
#                 phi_iter=10,
#                 reg_method= reg_methods,
#                 mlc_name= 'eur',
#                 project_mode='binary',
#                 prediction_thres=0,
#                 predicting_data='test_data'
#             )

#%% Specialized for wiki

wiki_train_x, wiki_train_y, wiki_train_n, wiki_train_d, wiki_train_k = real_data.read_wiki_train()
wiki_test_x, wiki_test_y, wiki_test_n, wiki_test_d, wiki_test_k = real_data.read_wiki_test()

print('shape of x and y:', wiki_train_x.shape, wiki_train_y.shape)
print('n, d, k:', wiki_train_n, wiki_train_d, wiki_train_k)
print('cuda:', torch.cuda.is_available())
cuda_time = datetime.datetime.now()
c_device = torch.device("cuda")
wiki_train_x,wiki_train_y = wiki_train_x.to(c_device),wiki_train_y
# wiki_test_x,wiki_test_y = wiki_test_x.to(c_device),wiki_test_y.to(c_device)
alpha_list = [0.01 * (1000 ** (i/4)) for i in range(5)]
l1_ratio_list = torch.arange(0.1,1,0.3)


for m in [300,400,500,700,1000,2000]: 
    for prediction_s in [1,3]:
        for reg_methods in ['PGD_MLC_LOOSE','PGD_MLC_valLOOSE','CD','PGD_from_FISTA']:
            precision_list = comp_func.MLC_func(
                filename=None,
                X_train=wiki_train_x,
                Y_train=wiki_train_y,
                X_test=wiki_test_x,
                Y_test=wiki_test_y,
                prediction_s=prediction_s,
                device = c_device,
                m=m,
                phi_iter=10,
                reg_method= reg_methods,
                mlc_name= 'wiki',
                project_mode='binary',
                prediction_thres=0,
                predicting_data='test_data'
            )





#%%


file_namelst = [
    'gaussian_0.1_3_1_0_valueNone_20240919233000_10000_30000_20000',
    'gaussian_0.1_3_1_None_valueNone_20240919232527_10000_30000_20000',
    'gaussian_0.1_3_10_0_valueNone_20240919233230_10000_30000_20000',
    'gaussian_0.1_3_10_None_valueNone_20240919232755_10000_30000_20000',
    'gaussian_0.032_3_1_0_valueNone_20240919232050_10000_30000_20000',
    'gaussian_0.032_3_1_None_valueNone_20240919231612_10000_30000_20000',
    'gaussian_0.032_3_10_0_valueNone_20240919232322_10000_30000_20000',
    'gaussian_0.032_3_10_None_valueNone_20240919231844_10000_30000_20000'
]
device = torch.device("cuda")
for filename in file_namelst:
    files = 'data/synt_data/'+filename,
    print('files:',files)
    X_train = torch.load(files[0]+'_X.pt')
    Y_train = torch.load(files[0]+'_Y.pt')
    Z = torch.load(files[0]+'_Z.pt')
    sparsity_Y = Y_train.count_nonzero()/Y_train.shape[1]
    calculated_y = Z@X_train
    calculated_sparsity = calculated_y.count_nonzero()/calculated_y.shape[1]
    difference = torch.norm(calculated_y-Y_train)/torch.norm(Y_train)
    print('filename:',filename)
    print('sparsity_Y:',sparsity_Y,'calculated_sparsity:',calculated_sparsity,'difference:',difference)
    print('')








# print('shape of x and y:', eur_train_x.shape, eur_train_y.shape)
# print('n, d, k:', eur_train_n, eur_train_d, eur_train_k)
# print('cuda:', torch.cuda.is_available())
# cuda_time = datetime.datetime.now()
# c_device = torch.device("cuda")
# eur_train_x,eur_train_y = eur_train_x.to(c_device),eur_train_y.to(c_device)
# eur_test_x,eur_test_y = eur_test_x.to(c_device),eur_test_y.to(c_device)
# alpha_list = [0.01 * (1000 ** (i/4)) for i in range(5)]
# l1_ratio_list = torch.arange(0.1,1,0.3)


# for prediction_s in [1,3]:
#     for reg_methods in ['PGD_MOR','PGD_MOR_positive','PGD_MLC','PGD_MLC_loose','PGD_MLC_LOOSE','PGD_MLC_val',
#                     'PGD_MLC_valloose','PGD_MLC_valLOOSE']:
#         precision_list = comp_func.MLC_func(
#             filename=None,
#             X_train=eur_train_x,
#             Y_train=eur_train_y,
#             X_test=eur_test_x,
#             Y_test=eur_test_y,
#             prediction_s=prediction_s,
#             m=1000,
#             phi_iter=10,
#             device=c_device,
#             reg_method= reg_methods,
#             mlc_name= 'eur',
#             project_mode='binary',
#             prediction_thres=0
#         )
#         precision_list = comp_func.MLC_func(
#             filename=None,
#             X_train=eur_train_x,
#             Y_train=eur_train_y,
#             X_test=eur_test_x,
#             Y_test=eur_test_y,
#             prediction_s=prediction_s,
#             m=1000,
#             phi_iter=10,
#             device=c_device,
#             reg_method= reg_methods,
#             mlc_name= 'eur',
#             project_mode='binary',
#             prediction_thres=0.5
#         )
#     print('1000,precision_list:', precision_list)
#     print('cuda_time:', datetime.datetime.now()-cuda_time)

# timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# ending_time = datetime.datetime.now()
# print('ending_time:', ending_time,'starting_time:', starting_time)


#%%


# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=1, snr=0.032, project_mode='value', project_thres=None)
# print('done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=0.032, project_mode='value', project_thres=None)
# print('2nd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=1, snr=0.032, project_mode='value', project_thres=0)
# print('3rd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=0.032, project_mode='value', project_thres=0)
# print('4th done')

# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=1, snr=0.1, project_mode='value', project_thres=None)
# print('done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=0.1, project_mode='value', project_thres=None)
# print('2nd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=1, snr=0.1, project_mode='value', project_thres=0)
# print('3rd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=0.1, project_mode='value', project_thres=0)
# print('4th done')


# see elastic net
#alpha = from 0.01 to 10, l1_ratio = 1





# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_4_20240804181052_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='PGD', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=1000)
# print('1000,precision_list:', precision_list)
# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_4_20240804181052_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='FISTA', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=1000)
# print('1000,precision_list:', precision_list)


# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_1_20240804180158_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='PGD', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=1000)
# print('1000,precision_list:', precision_list)
# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_1_20240804180158_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='FISTA', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=300)
# print('1000,precision_list:', precision_list)


# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_0.032_20240804175302_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='PGD', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=500)
# print('1000,precision_list:', precision_list)
# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_0.032_20240804175302_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='FISTA', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=700)
# print('1000,precision_list:', precision_list)


# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_1_20240802155843_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='PGD', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=3000)
# print('3000,precision_list:', precision_list)

# precision_list = comp_func.synt_func(filename = 'data/synt_data/gaussian_1_20240802155843_10000_30000_20000',device = torch.device("cpu"),predicting_data='test_data',
#                                               reg_method='PGD', compression_delta = 0.3, compression_epsilon = 0.1,phi_iter = 10,m=5000)
# print('5000,precision_list:', precision_list)


# file_namelst = [
#     'gaussian_0.1_3_1_0_valueNone_20240919233000_10000_30000_20000',
#     'gaussian_0.1_3_1_None_valueNone_20240919232527_10000_30000_20000',
#     'gaussian_0.1_3_10_0_valueNone_20240919233230_10000_30000_20000',
#     'gaussian_0.1_3_10_None_valueNone_20240919232755_10000_30000_20000',
#     'gaussian_0.032_3_1_0_valueNone_20240919232050_10000_30000_20000',
#     'gaussian_0.032_3_1_None_valueNone_20240919231612_10000_30000_20000',
#     'gaussian_0.032_3_10_0_valueNone_20240919232322_10000_30000_20000',
#     'gaussian_0.032_3_10_None_valueNone_20240919231844_10000_30000_20000'
# ]
# for file in file_namelst:
#     for prediction_s in [1,3]:
#         for reg_methods in ['PGD_MOR','PGD_MOR_positive']:
#             precision_list = comp_func.synt_func(
#                 filename = 'data/synt_data/'+file,
#                 device = torch.device("cuda"),
#                 prediction_s = prediction_s,
#                 compression_delta=0.3,
#                 compression_epsilon=0.1,
#                 phi_iter= 10,
#                 m = 1000,
#                 predicting_data='test_data',
#                 project_mode='value',
#                 reg_method= reg_methods,
#                 prediction_thres=None,
#             )
#             print('thres is none,precision_list:', precision_list)
#             precision_list = comp_func.synt_func(
#                 filename = 'data/synt_data/'+file,
#                 device = torch.device("cuda"),
#                 prediction_s = prediction_s,
#                 compression_delta=0.3,
#                 compression_epsilon=0.1,
#                 phi_iter= 10,
#                 m = 1000,
#                 predicting_data='test_data',
#                 project_mode='value',
#                 reg_method= reg_methods,
#                 prediction_thres=0,
#             )
#             print('thres is 0,precision_list:', precision_list)


# eur_train_x, eur_train_y, eur_train_n, eur_train_d, eur_train_k = real_data.read_eur_train()
# eur_test_x, eur_test_y, eur_test_n, eur_test_d, eur_test_k = real_data.read_eur_test()



# print('shape of x and y:', eur_train_x.shape, eur_train_y.shape)
# print('n, d, k:', eur_train_n, eur_train_d, eur_train_k)
# print('cuda:', torch.cuda.is_available())
# cuda_time = datetime.datetime.now()
# c_device = torch.device("cuda")
# eur_train_x,eur_train_y = eur_train_x.to(c_device),eur_train_y.to(c_device)
# eur_test_x,eur_test_y = eur_test_x.to(c_device),eur_test_y.to(c_device)
# alpha_list = [0.01 * (1000 ** (i/4)) for i in range(5)]
# l1_ratio_list = torch.arange(0.1,1,0.3)


# for prediction_s in [1,3]:
#     for reg_methods in ['PGD_MOR','PGD_MOR_positive','PGD_MLC','PGD_MLC_loose','PGD_MLC_LOOSE','PGD_MLC_val',
#                     'PGD_MLC_valloose','PGD_MLC_valLOOSE']:
#         precision_list = comp_func.MLC_func(
#             filename=None,
#             X_train=eur_train_x,
#             Y_train=eur_train_y,
#             X_test=eur_test_x,
#             Y_test=eur_test_y,
#             prediction_s=prediction_s,
#             m=1000,
#             phi_iter=10,
#             device=c_device,
#             reg_method= reg_methods,
#             mlc_name= 'eur',
#             project_mode='binary',
#             prediction_thres=0
#         )
#         precision_list = comp_func.MLC_func(
#             filename=None,
#             X_train=eur_train_x,
#             Y_train=eur_train_y,
#             X_test=eur_test_x,
#             Y_test=eur_test_y,
#             prediction_s=prediction_s,
#             m=1000,
#             phi_iter=10,
#             device=c_device,
#             reg_method= reg_methods,
#             mlc_name= 'eur',
#             project_mode='binary',
#             prediction_thres=0.5
#         )
#     print('1000,precision_list:', precision_list)
#     print('cuda_time:', datetime.datetime.now()-cuda_time)

# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=1, it_num=10, snr=0.32, seed=20)
# print('done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=1, it_num=10, snr=0.032, seed=10)
# print('2nd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=1, it_num=10, snr=1, seed=30)
# print('3rd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=1, it_num=10, snr=4, seed=40)
# print('4th done')

# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=0.32, seed=20)
# print('done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=0.032, seed=10)
# print('2nd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=1, seed=30)
# print('3rd done')
# gen_synt.generate_synt(d=10000, n=30000, k=20000, s=3, it_num=10, snr=4, seed=40)
# print('4th done')
# cpu_time = datetime.datetime.now()
# c_device = torch.device("cpu")
# eur_train_x,eur_train_y = eur_train_x.to(c_device),eur_train_y.to(c_device)
# eur_test_x,eur_test_y = eur_test_x.to(c_device),eur_test_y.to(c_device)
# precision_list = comp_func.MLC_func(
#     filename=None,
#     X_train=eur_train_x,
#     Y_train=eur_train_y,
#     X_test=eur_test_x,
#     Y_test=eur_test_y,
#     s=3,
#     m=1000,
#     phi_iter=10,
#     device=c_device,
#     predicting_data='test_data',
#     reg_method='PGD',
#     mlc_name= 'eur'
# )
# print('1000,precision_list:', precision_list)
# print('cpu_time:', datetime.datetime.now()-cpu_time)


timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# comp_func.store_res(
#     filename=f'ama_{timestamp}.pt',
#     precision_list=precision_list,
#     output_distance_list=output_distance_list,
#     prediction_loss_list=prediction_loss_list,
#     others_list=others_list,
#     cs_training_loss_list=cs_training_loss_list,
#     training_loss=training_loss
# )

# ama_train_x, ama_train_y, ama_train_n, ama_train_d, ama_train_k = real_data.read_ama_train()
# ama_test_x, ama_test_y, ama_test_n, ama_test_d, ama_test_k = real_data.read_ama_test()

# print('shape of x and y:', ama_train_x.shape, ama_train_y.shape)
# print('n, d, k:', ama_train_n, ama_train_d, ama_train_k)
# # print('cuda:', torch.cuda.is_available())

# precision_list, output_distance_list, prediction_loss_list, others_list, cs_training_loss_list, training_loss = comp_func.MLC_func(
#     filename=None,
#     X_train=ama_train_x,
#     Y_train=ama_train_y,
#     X_test=ama_test_x,
#     Y_test=ama_test_y,
#     s=1,
#     m=3000,
#     phi_iter=3,
#     device=torch.device("cpu"),
#     predicting_data='test_data',
#     reg_method='PGD'
# )

# timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

ending_time = datetime.datetime.now()
print('ending_time:', ending_time,'starting_time:', starting_time)
# %%
