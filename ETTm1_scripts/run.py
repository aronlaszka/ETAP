import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Intel Math Kernel Library
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS

import torch

TAG = False
w_momentum = False
Name = 'ITA_Approx_Avg'
if TAG:
    if Name == 'ITA': #fifty's affinity score
        from exp.exp_calculate_ITA import Exp_Main
    elif Name == 'ITA_Approx': #our affinity score
        from exp.exp_calculate_affinity import Exp_Main
else:
    from exp.exp_main import Exp_Main #regular MTL training

import random
import numpy as np
import pandas as pd

from itertools import combinations
def combine(temp_list, n):
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1_scripts', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, '
                         'S:univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# sample_mask
parser.add_argument('--frac',type=int, default=0)
parser.add_argument('--frac_num',type=int,default=8)
parser.add_argument('--dataset', type=str, default='ETTm1_scripts',
                    help='choose dataset')

# model define
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='1', help='device ids of multiple gpus')


if __name__ == '__main__':


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # args.gpu = 1

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]



    print('Args in experiment:')
    print(args)
    # ETTm1_scripts
    task_num = 7
    print(task_num)
    tasks = list(range(task_num))
    all_tasks = []
    for i in range(task_num):
        all_tasks.extend(combine(tasks, i+1))

    print(f'all_tasks = {len(all_tasks)}')
    print(f'all_tasks = {all_tasks}')
    # count = 0
    # for each_combi in all_tasks:
    #
    #     if len(each_combi) >3:
    #         count += 1
    #     else:
    #         final_idx = all_tasks.index(each_combi)
    # print(f'count = {count}, final_idx = {final_idx}')
    # exit(0)
    Exp = Exp_Main
    # selected_tasks = [tasks]
    frac = args.frac
    frac_num = args.frac_num
    print(f'all_tasks = {all_tasks}')
    # selected_tasks = all_tasks[ frac*32 : (frac + 1)*32]
    # selected_tasks = [tasks]
    # selected_tasks = all_tasks[2:7]
    '''63 for all_groups>=4'''
    # all_tasks = all_tasks[63:126]
    # print(all_tasks)
    # print(f'len(all_tasks) = {len(all_tasks)}')
    selected_tasks = all_tasks[frac_num:frac_num + 32]
    # exit(0)
    # selected_tasks = all_tasks[frac_num:frac_num+35]
    # selected_tasks = all_tasks[-1:]
    print(f'selected_tasks = {selected_tasks}')
    print(f'total selected_tasks = {len(selected_tasks)}')


    # total_train_loss_mse = []
    # total_train_perf_mse = []
    # total_val_perf_mae = []
    # total_val_perf_mse = []
    # total_test_perf_mae = []
    # total_test_perf_mse = []
    # Cut_off_epoch = []
    # total_val_mse = []
    # total_test_mse = []
    # Tasks = []
    # Time_Required = []

    for run_idx in range(1,4):
        # if run_idx == 1:
        #     w_momentum = False
        # else:
        #     w_momentum = True
        total_train_loss_mse = []
        total_train_perf_mse = []
        total_val_perf_mae = []
        total_val_perf_mse = []
        total_test_perf_mae = []
        total_test_perf_mse = []
        Cut_off_epoch = []
        total_val_mse = []
        total_test_mse = []
        Tasks = []
        Time_Required = []

        # mask[np.array(all_tasks[-1])] = 1
        for selected_task in selected_tasks:
            print(f'selected_task = {selected_task}')
            Tasks.append(selected_task)
            mask = np.zeros((task_num),)
            mask[np.array(selected_task)] = 1
            print(f'mask = {mask} before training')
            print(f'args.itr = {args.itr}')
            if args.is_training:
                for ii in range(args.itr):
                    # setting record of experiments
                    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}'.format(
                        args.model_id,
                        args.model,
                        args.data,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.d_model,
                        args.n_heads,
                        args.e_layers,
                        args.d_layers,
                        args.d_ff,
                        args.factor,
                        args.embed,
                        args.distil,
                        args.des,
                        args.frac,
                        args.frac_num,
                         ii)

                    exp = Exp(args)  # set experiments
                    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

                    if TAG:
                        if Name == 'ITA':
                            trained_model, cut_off_epoch, gradient_metrics, TimeTaken = exp.train_model(setting, mask, run_idx)
                            filename = f'ITA/gradient_metrics_{Name}_run_{run_idx}_SGD_new.csv'
                            with open(filename, 'w') as f:
                                for key in gradient_metrics.keys():
                                    f.write("%s,%s\n" % (key, gradient_metrics[key]))
                            f.close()
                            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                            train_per_mae, train_per_mse, val_per_mae, val_per_mse, test_per_mae, test_per_mse = exp.test(
                                setting, mask, run_idx)

                        if Name == 'ITA_Approx' and not w_momentum:
                            trained_model, cut_off_epoch, gradient_metrics, TimeTaken = exp.train_model(setting,mask,run_idx)
                            filename = f'ITA/gradient_metrics_{Name}_run_{run_idx}_wo_momentum_SGD_new_wo_loss.csv'
                            with open(filename, 'w') as f:
                                for key in gradient_metrics.keys():
                                    f.write("%s,%s\n" % (key, gradient_metrics[key]))
                            f.close()
                            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                            train_per_mae, train_per_mse, val_per_mae, val_per_mse, test_per_mae, test_per_mse = exp.test(
                                setting, mask, run_idx, w_momentum=False)

                        if Name == 'ITA_Approx' and w_momentum:
                            trained_model, cut_off_epoch, gradient_metrics, TimeTaken = exp.train_model(setting,
                                                                                                        mask,
                                                                                                        run_idx,
                                                                                                        w_momentum=True)
                            filename = f'ITA/gradient_metrics_{Name}_run_{run_idx}_w_momentum_SGD_new_wo_loss.csv'
                            with open(filename, 'w') as f:
                                for key in gradient_metrics.keys():
                                    f.write("%s,%s\n" % (key, gradient_metrics[key]))
                            f.close()
                            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                            train_per_mae, train_per_mse, val_per_mae, val_per_mse, test_per_mae, test_per_mse = exp.test(
                                setting, mask, run_idx, w_momentum=True)


                    else:

                        import time
                        start = time.time()
                        trained_model, cut_off_epoch = exp.train_model(setting,mask, run_idx)


                        end = time.time()
                        TimeTaken = end - start

                        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        train_per_mae, train_per_mse, val_per_mae, val_per_mse, test_per_mae, test_per_mse = exp.test(setting,mask, run_idx)

                    total_test_perf_mae.append(test_per_mae)
                    total_test_perf_mse.append(test_per_mse)
                    total_val_perf_mae.append(val_per_mae)
                    total_val_perf_mse.append(val_per_mse)
                    total_train_perf_mse.append(train_per_mse)
                    total_train_loss_mse.append(np.sum(train_per_mse))
                    total_val_mse.append(np.sum(val_per_mse))
                    total_test_mse.append(np.sum(test_per_mse))
                    Cut_off_epoch.append(cut_off_epoch)
                    # if TAG:
                    Time_Required.append(TimeTaken)

                    path = f'new_gain/{args.dataset}/'
                    # if not os.path.exists(path):
                    #     os.makedirs(path)
                    # np.save(path + 'valmae_' + str(selected_task)+'.npy',total_test_perf_mae)
                    # np.save(path + 'testmae_' + str(selected_task)+'.npy',total_val_perf_mae)
                    # np.save(path + 'valmse_' + str(selected_task)+'.npy',total_test_perf_mse)
                    # np.save(path + 'testmse_' + str(selected_task)+'.npy',total_val_perf_mse)

                    if args.do_predict:
                        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                        exp.predict(setting, True)



                    print('>>>>>>>finished single experiment>>>>>>>>')
                    print(len(total_test_perf_mae), len(total_val_perf_mae), len(total_test_perf_mse),
                          len(total_val_perf_mse), len(total_test_mse), len(total_val_mse), len(total_train_loss_mse), len(total_train_perf_mse))

                    res_df = pd.DataFrame({'Task': Tasks,
                                           'total_train_loss_mse': total_train_loss_mse,
                                           # 'total_train_loss_mae': total_train_loss_mae,
                                           'total_val_loss_mse': total_val_mse,
                                           'total_test_loss_mse': total_test_mse,
                                           'total_train_perf_mse': total_train_perf_mse,
                                           'total_val_perf_mse': total_val_perf_mse,
                                           'total_test_perf_mse': total_test_perf_mse,
                                           'total_test_perf_mae': total_test_perf_mae,
                                           'total_val_perf_mae': total_val_perf_mae,
                                            'Cut_off_epoch': Cut_off_epoch,
                                            'Time_Required': Time_Required
                                           })
                    # res_df.to_csv(f'new_gain/{args.dataset}/results_STL.csv', index=False)
                    # res_df.to_csv(f'new_gain/{args.dataset}/results_Sample.csv', index=False)
                    if TAG:
                        if not w_momentum:
                            if Name == 'ITA':
                                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_ALL_{run_idx}_{Name}_SGD_newRes.csv', index=False)
                            else:
                                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_ALL_{run_idx}_{Name}_wo_momentum_SGD_newRes.csv', index=False)
                        else:
                            res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_ALL_{run_idx}_{Name}_w_momentum_SGD_newRes.csv', index=False)
                    else:
                        if args.frac_num == 0:
                            suffix = 'STL'
                        elif args.frac_num == 7:
                            suffix = 'Pairs'
                        elif args.frac_num == 28:
                            suffix = 'G3'
                        elif args.frac_num == 126:
                            suffix = 'ALL'
                        else:
                            suffix = 'Groups'
                        res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{frac_num}_{run_idx}_{len(Tasks)}_{suffix}_SGD.csv', index=False)

                        if os.path.exists(f'new_gain/{args.dataset}/results_ETTm1_{frac_num}_{run_idx}_{len(Tasks)-1}_{suffix}_SGD.csv'):
                            os.remove(f'new_gain/{args.dataset}/results_ETTm1_{frac_num}_{run_idx}_{len(Tasks)-1}_{suffix}_SGD.csv')

                    torch.cuda.empty_cache()

            else:
                ii = 0
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}'.format(args.model_id,
                                                                                                            args.model,
                                                                                                            args.data,
                                                                                                            args.features,
                                                                                                            args.seq_len,
                                                                                                            args.label_len,
                                                                                                            args.pred_len,
                                                                                                            args.d_model,
                                                                                                            args.n_heads,
                                                                                                            args.e_layers,
                                                                                                            args.d_layers,
                                                                                                            args.d_ff,
                                                                                                            args.factor,
                                                                                                            args.embed,
                                                                                                            args.distil,
                                                                                                            args.des,
                                                                                                              args.frac,
                                                                                                              args.frac_num,
                                                                                                              ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting,mask, test=1)
                torch.cuda.empty_cache()

        print('>>>>>>>finished all experiments>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(len(total_test_perf_mae), len(total_val_perf_mae), len(total_test_perf_mse), len(total_val_perf_mse), len(total_test_mse), len(total_val_mse), len(total_train_loss_mse), len(total_train_perf_mse))
        #
        # for each_val_perf_mse in total_val_perf_mse:
        #     total_val_mse.append(np.sum(each_val_perf_mse))
        #
        # for each_test_perf_mse in total_test_perf_mse:
        #     total_test_mse.append(np.sum(each_test_perf_mse))

        res_df = pd.DataFrame({'Task': Tasks,
                               'total_train_loss_mse': total_train_loss_mse, # 'total_train_loss_mae': total_train_loss_mae,
                               'total_val_loss_mse': total_val_mse,
                               'total_test_loss_mse': total_test_mse,
                               'total_train_perf_mse': total_train_perf_mse,
                               'total_val_perf_mse': total_val_perf_mse,
                               'total_test_perf_mse': total_test_perf_mse,
                               'total_test_perf_mae': total_test_perf_mae,
                               'total_val_perf_mae': total_val_perf_mae,
                                'Cut_off_epoch': Cut_off_epoch,
                                'Time_Required': Time_Required
                               })
            # res_df.to_csv(f'new_gain/{args.dataset}/results_ALL_Tasks_{Name}.csv', index=False)
        if not TAG:
            if args.frac_num == 0:
                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{run_idx}_STL_SGD.csv', index=False)
            elif args.frac_num == 7:
                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{run_idx}_Pairs_SGD.csv', index=False)
            elif args.frac_num == 28:
                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{run_idx}_G3_SGD.csv', index=False)
            elif args.frac_num == 126:
                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{run_idx}_ALL_SGD.csv', index=False)
            else:
                res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{run_idx}_groups_SGD.csv', index=False)
        else:
            res_df.to_csv(f'new_gain/{args.dataset}/results_ETTm1_{run_idx}_{Name}_ALL_SGD.csv', index=False)

