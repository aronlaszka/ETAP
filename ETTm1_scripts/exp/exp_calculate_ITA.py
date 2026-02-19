import math

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Autoformer
# , Transformer, Reformer, Informer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch import optim
import copy

import os
import time
import tqdm

import warnings
# import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from itertools import combinations


def combine(temp_list, n):
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            # 'Reformer': Reformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, mask_):
        total_loss = []
        task_loss_all = []
        self.model.eval()
        # print(f'len loader at vali = {len(vali_loader)}')

        with torch.no_grad():
            for vali_batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) * mask_
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask_
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                task_specific_loss = []
                for task_idx in range(batch_x.shape[2]):
                    task_specific_loss.append(criterion(outputs[:, :, task_idx], batch_y[:, :, task_idx]))
                task_loss_all.append(task_specific_loss)
                loss = criterion(pred, true)
                total_loss.append(loss)

        # print(f'task_loss_all = {len(task_loss_all)}, sum = {np.sum(task_loss_all)}')
        task_specific_loss_all = torch.FloatTensor(task_loss_all)
        # print(f'len total_loss ={len(total_loss)}')
        total_loss = np.average(total_loss)
        print(f'total loss avg = {total_loss}')
        self.model.train()
        return total_loss, task_specific_loss_all

    def vali_per_batch(self, batch_data, trained_model, criterion, mask_):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
        trained_model.eval()
        with torch.no_grad():

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.output_attention:
                outputs = trained_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = trained_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0

            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            task_specific_loss_all = []
            for idx in range(batch_x.shape[2]):
                task_specific_loss_all.append(criterion(outputs[:, :, idx], batch_y[:, :, idx]))
            tot_loss = criterion(pred, true)
        trained_model.train()
        return tot_loss, task_specific_loss_all

    def train_model(self, setting, mask_, run_idx):
        print(f'Training using Exp_tag')

        mask = torch.from_numpy(mask_).to(self.device)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        TRAIN_SIZE = len(train_loader.dataset)
        BATCH_SIZE = self.args.batch_size

        print(f'train data: {len(train_data)}, vali data:, {len(vali_data)}, test data:,{len(test_data)}')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path = path + '/' + 'TAG_Exact_' + str(run_idx)
        if not os.path.exists(path):
            os.makedirs(path)
        print(f'path = {path}')

        if self.args.is_training:
            check_point_file = path + '/' + 'checkpoint.pth'
            if os.path.exists(check_point_file):
                print(f'removing old checkpoint file {check_point_file}')
                os.remove(check_point_file)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_model_name = f'curr_best_model_TAG_{run_idx}.pth'
        check_point_path = f'curr_model_TAG_{run_idx}.pth'
        if os.path.exists(best_model_name):
            os.remove(best_model_name)






        # torch.save(self.model.state_dict(), best_model_per_epoch)

        tasks = list(range(len(mask)))
        gradient_metrics = {task: [] for task in tasks}

        timeStart = time.time()
        for epoch in range(self.args.train_epochs):

            train_loss = []

            self.model.train()
            print(f'current learning rate = {model_optim.param_groups[0]["lr"]}')


            iter_count = 0
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(train_loader)):
                # if i ==5:
                #     break
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask
                        loss = criterion(outputs[outputs != 0], batch_y[outputs != 0])
                        train_loss.append(loss.item())
                        print(f'loss: {loss.item()}')
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    # print(f'shape of outputs: {outputs.shape}, shape of batch_y: {batch_y.shape}')
                    # print(f'first sample of outputs: {outputs[0][-self.args.pred_len][f_dim:]}')
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                    # print(f'first sample of outputs: {outputs[0][-self.args.pred_len][f_dim:]}')

                    # print(f'first sample of batch_y: {batch_y[0][-self.args.pred_len][f_dim:]}')
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask
                    # print(f'first sample of batch_y: {batch_y[0][-self.args.pred_len][f_dim:]}')
                    # print(f'shape of outputs: {outputs.shape}, shape of batch_y: {batch_y.shape}')

                    col = np.where(mask_ == 1)[0]
                    loss = criterion(outputs[:, :, col], batch_y[:, :, col])

                    train_loss.append(loss.item())
                if (batch_idx + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_idx + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - batch_idx)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


            # _,pre_train_loss = self.vali(train_data, train_loader, criterion, mask)
            train_task_affinity_i = {task: {task: [] for task in tasks} for task in tasks}

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'loss': loss}, check_point_path)

            checkpoint = torch.load(check_point_path, weights_only=True)

            tmp_model = copy.deepcopy(self.model)
            tmp_model_param = copy.deepcopy(self.model.state_dict())
            tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=model_optim.param_groups[0]['lr'], momentum=0.9)


            tmp_model.load_state_dict(checkpoint['model_state_dict'])
            tmp_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            tmp_model.train()

            for task in range(len(mask)):
                print(f'Processing task: {task}')
                '''load the parameters before the update'''

                train_task_affinity_i_j = {task: [] for task in tasks}
                '''evaluate TAG'''
                for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(train_loader)):
                    tmp_optimizer.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    outputs = tmp_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask

                    pre_train_loss = [criterion(outputs[:, :, idx], batch_y[:, :, idx]) for idx in
                                      range(batch_x.shape[-1])]

                    '''update the gradient based on each task'''
                    task_loss = pre_train_loss[task]


                    task_loss.backward()
                    tmp_optimizer.step()

                    batch_Data = (batch_x, batch_y, batch_x_mark, batch_y_mark)
                    t_loss, t_pertask_loss = self.vali_per_batch(batch_Data, tmp_model, criterion, mask)

                    for idx in range(len(pre_train_loss)):
                        tag_val = ((1.0 - (t_pertask_loss[idx] / pre_train_loss[
                            idx])) / model_optim.param_groups[0]['lr']).cpu().detach().numpy()
                        train_task_affinity_i_j[idx].append(tag_val.item() / (math.ceil(TRAIN_SIZE / BATCH_SIZE)))

                '''get avg affinity for each task'''
                for task_2, affinity in train_task_affinity_i_j.items():
                    # print(f'task_2 = {task_2}, affinity = {len(affinity)}')
                    train_task_affinity_i_j[task_2] = np.average(affinity)

                    train_task_affinity_i[task][task_2].append(train_task_affinity_i_j[task_2])

                '''load the model parameters before the update'''
                tmp_model.load_state_dict(checkpoint['model_state_dict'])
                tmp_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f'*' * 200)

            tr_loss, tr_task_loss = self.vali(train_data, train_loader, criterion, mask)
            vali_loss, task_loss = self.vali(vali_data, vali_loader, criterion, mask)
            # test_loss,_ = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, tr_loss, vali_loss, ))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.counter == 0:
                '''save model'''
                torch.save(self.model.state_dict(), best_model_name)
                timeEnd = time.time()
                for combined_task, task_gain_map in train_task_affinity_i.items():
                    gradient_metrics[combined_task].append(task_gain_map)
                Final_timetaken = timeEnd - timeStart


            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch > 2:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        '''save TAG'''
        self.model.load_state_dict(torch.load(best_model_name))

        return self.model,epoch, gradient_metrics,Final_timetaken

    def test(self, setting, mask_, run_idx, test=0):

        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        vali_data, vali_loader = self._get_data(flag='val')

        criterion = self._select_criterion()
        mask = torch.from_numpy(mask_).to(self.device)
        # if test:
        print('loading model')
        # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        # best_model_name = f'curr_best_model_TAG_{run_idx}.pth'
        # self.model.load_state_dict(torch.load(best_model_name))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # train_set perf
        preds = []
        trues = []
        train_mse = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                tr_outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                tr_batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask
                col = np.where(mask_ == 1)[0]
                loss = criterion(tr_outputs[:, :, col], tr_batch_y[:, :, col])
                train_mse.append(loss.item())

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # if i % 50 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds) * mask_
        trues = np.array(trues) * mask_
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print(f'shape of preds: {preds.shape}, shape of trues: {trues.shape}')
        # print(f'preds: {preds[0][0]}, trues: {trues[0][0]}')

        print(f'avg train mse: {np.mean(train_mse)}')
        tr_mae, tr_mse, tr_rmse, tr_mape, tr_mspe, tr_per_mae, tr_per_mse = metric(preds, trues)
        print('train per mae:{}, train per mse{}'.format(tr_per_mae, tr_per_mse))
        print('tr mse:{}, tr mae:{}'.format(tr_mse, tr_mae))
        # print('valid mse:', np.mean(valid_mse))
        # print(f'different val_mse: {np.sum(val_per_mse)}')
        tr_mse = np.sum(tr_per_mse)
        tr_mae = np.sum(tr_per_mae)
        print(f'actual tr_mse = {tr_mse}, actual tr_mae = {tr_mae}')

        #       valid_perf
        preds = []
        trues = []
        valid_mse = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                val_outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                val_batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask
                col = np.where(mask_ == 1)[0]
                loss = criterion(val_outputs[:, :, col], val_batch_y[:, :, col])
                valid_mse.append(loss.item())

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # if i % 50 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds) * mask_
        trues = np.array(trues) * mask_
        # print('vali shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('vali shape:', preds.shape, trues.shape)

        # print(f'shape of preds: {preds.shape}, shape of trues: {trues.shape}')
        # print(f'preds: {preds[0][0]}, trues: {trues[0][0]}')

        val_mae, val_mse, val_rmse, val_mape, val_mspe, val_per_mae, val_per_mse = metric(preds, trues)
        val_mse = np.sum(val_per_mse)
        val_mae = np.sum(val_per_mae)
        print('val per mae:{}, val per mse{}'.format(val_per_mae, val_per_mse))
        # print('mse:{}, mae:{}'.format(val_mse, val_mae))
        # print('valid mse:', np.mean(valid_mse))
        # print(f'different val_mse: {np.sum(val_per_mse)}')

        np.save(folder_path + 'val_metrics.npy', np.array([val_mae, val_mse, val_rmse, val_mape, val_mspe]))
        np.save(folder_path + 'val_pred.npy', preds)
        np.save(folder_path + 'val_true.npy', trues)

        # test perf
        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 50 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds) * mask_
        trues = np.array(trues) * mask_
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, per_mae, per_mse = metric(preds, trues)
        mae = np.sum(per_mae)
        mse = np.sum(per_mse)

        print('per mae:{}, per mse{}'.format(per_mae, per_mse))
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_pairs.txt", 'a')
        f.write(setting + "  \n")
        f.write('\n\n' + 'task' + str(mask_) + '\n')
        f.write('train mse:{}, train mae:{}, '.format(tr_mse, tr_mae) + '\n')
        f.write('val mse:{}, val mae:{}, '.format(val_mse, val_mae))
        f.write('val per mae:{}, val per mse{}, '.format(val_per_mae, val_per_mse) + '\n')
        f.write('mse:{}, mae:{}, '.format(mse, mae))
        f.write('per mae:{}, per mse{}'.format(per_mae, per_mse))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'test_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'test_pred.npy', preds)
        np.save(folder_path + 'test_true.npy', trues)

        return tr_per_mae, tr_per_mse, val_per_mae, val_per_mse, per_mae, per_mse

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return