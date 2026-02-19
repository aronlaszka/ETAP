from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Autoformer
# , Transformer, Reformer, Informer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import math
import torch
import torch.nn as nn
from torch import optim
import tqdm
import os
import time
import warnings
import numpy as np
import copy
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

    def vali(self, vali_loader, criterion, mask_):
        total_loss = []
        task_loss_all = []
        self.model.eval()
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

    def train_model(self, setting, mask_, run_idx, w_momentum=False):
        print(f'Training using Exp_equivalent_new for ITA Approximation')
        mask = torch.from_numpy(mask_).to(self.device)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        TRAIN_SIZE = len(train_data)
        BATCH_SIZE = self.args.batch_size
        print(f'TRAIN_SIZE = {TRAIN_SIZE}, BATCH_SIZE = {BATCH_SIZE}')
        print(f'train data: {len(train_data)}, vali data:, {len(vali_data)}, test data:,{len(test_data)}')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        '''add a suffix to the path'''
        if w_momentum:
            path = path + '/' + 'TAG_Approx_w_momentum_' + str(run_idx)
        else:
            path = path + '/' + 'TAG_Approx_' + str(run_idx)
        if not os.path.exists(path):
            os.makedirs(path)

        print(f'path = {path}')

        if self.args.is_training:
            check_point_file = path + '/' + 'checkpoint.pth'
            if os.path.exists(check_point_file):
                print(f'removing old checkpoint file {check_point_file}')
                os.remove(check_point_file)


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if w_momentum:
            best_model_name = f'curr_best_model_tag_approx_w_momentum_{run_idx}.pth'
            check_point_path = f'curr_model_TAG_Approx_w_momentum_{run_idx}.pth'
        else:
            best_model_name = f'curr_best_model_tag_approx_{run_idx}.pth'
            check_point_path = f'curr_model_TAG_Approx_{run_idx}.pth'

        if os.path.exists(best_model_name):
            os.remove(best_model_name)

        # torch.save(self.model.state_dict(), best_model_name)


        '''print number of model parameters'''
        print(f'number of model parameters = {sum(p.numel() for p in self.model.parameters())}')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        tasks = list(range(len(mask)))
        gradient_metrics = {task: [] for task in tasks}

        timeStart = time.time()
        for epoch in range(self.args.train_epochs):
            train_loss = []
            # if epoch>2:
            #     adjust_learning_rate(model_optim, epoch + 1, self.args)

            self.model.train()

            task_gains_approximation = {base_task: {} for base_task in tasks}
            batch_grad_metrics = {combined_task: {task: 0. for task in tasks} for combined_task in
                                  gradient_metrics}

            print(f'epoch {epoch} current learning rate = {model_optim.param_groups[0]["lr"]}, {self.args.learning_rate}')

            # continue
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(train_loader)):
                single_task_specific_gradients = []
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask


                col = np.where(mask_ == 1)[0]
                # print(f'output')
                # print(mask_, type(mask_))
                # print(col, type(col))
                loss = criterion(outputs[:, :, col], batch_y[:, :, col])
                train_loss.append(loss.item())
                # print(f'train_loss = {train_loss}')

                tmp_model = copy.deepcopy(self.model)
                tmp_model.train()
                tmp_model_param = copy.deepcopy(self.model.state_dict())
                tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=model_optim.param_groups[0]['lr'], momentum=0.9)
                tmp_optimizer.load_state_dict(model_optim.state_dict())
                '''print how many param groups in model_optim'''
                if batch_idx%500 == 0:
                    print(f'len model_optim.param_groups = {len(model_optim.param_groups)}')  # just one param group
                    for each_idx, each_param in enumerate(model_optim.param_groups):
                        print(f'each_idx = {each_idx}, lr = {model_optim.param_groups[each_idx]["lr"]}')

                '''ITA Approximation'''
                per_task_loss = []
                for task_idx in range(batch_x.shape[-1]):
                    task_loss = criterion(outputs[:, :, task_idx], batch_y[:, :, task_idx])
                    per_task_loss.append(task_loss)
                    task_loss.backward(retain_graph=True)
                    # Store the gradients
                    task_gradients = [param.grad.clone() for param in self.model.parameters()]
                    single_task_specific_gradients.append((task_idx, task_gradients))
                    tmp_optimizer.zero_grad()


                '''reshape and flatten gradients for each task pair'''
                reshaped_gradients = [torch.cat([grad.view(-1) for grad in grads], dim=0)
                                      for _, grads in single_task_specific_gradients]

                '''calculate the updates'''
                if w_momentum:
                    task_gradient_updates = {}
                    if batch_idx == 0 and epoch == 0:
                        for single_task, task_gradient in single_task_specific_gradients:
                            task_gradient_updates[single_task] = \
                                [tmp_optimizer.param_groups[0]['lr'] * grad for grad in task_gradient]
                        print(f'zero momentum at batch_idx = {batch_idx}, epoch = {epoch}')
                    else:
                        # print(f'Found momentum {model_optim.param_groups[0]["momentum"]}')
                        for single_task, task_gradient in single_task_specific_gradients:
                            '''check if adam optimizer or sgd'''
                            base_update = []
                            if isinstance(model_optim, torch.optim.SGD):
                                for param, grad in zip(self.model.parameters(), task_gradient):
                                    state = model_optim.state[param]
                                    if 'momentum_buffer' in state:
                                        momentum_buffer = state['momentum_buffer']
                                        # print(f'momentum_buffer = {momentum_buffer}')
                                    else:
                                        momentum_buffer = torch.zeros_like(param)
                                        print(f'zero momentum_buffer = {momentum_buffer}')
                                        exit(0)
                                    momentum = model_optim.param_groups[0]['momentum']
                                    # Compute the update value based on SGD with momentum
                                    update_value = model_optim.param_groups[0]['lr'] * (grad + momentum * momentum_buffer)
                                    base_update.append(update_value)

                            else:
                                print('Found Adam Optimizer')
                                exit(0)
                                for param, grad in zip(self.model.parameters(), task_gradient):
                                    state = model_optim.state[param]
                                    if 'momentum_buffer' in state:
                                        momentum_buffer = state['momentum_buffer']
                                    else:
                                        momentum_buffer = torch.zeros_like(param)
                                    beta_1 = model_optim.param_groups[0]['betas'][0]
                                    update_value = learning_rate * grad - beta_1 * momentum_buffer
                                    base_update.append(update_value)

                            task_gradient_updates[single_task] = base_update
                else:
                    task_gradient_updates = {}
                    for single_task, task_gradient in single_task_specific_gradients:
                        base_update = [tmp_optimizer.param_groups[0]['lr'] * grad for grad in task_gradient]
                        task_gradient_updates[single_task] = base_update



                # print(f'task_gradient_updates = {task_gradient_updates.keys()}')
                '''flatten and concatenate updates for all tasks to get update matrix'''
                reshaped_updates = [torch.cat([update.view(-1) for update in updates], dim=0)
                                    for _, updates in task_gradient_updates.items()]


                '''convert gradients and updates to tensors'''
                G = torch.stack(reshaped_gradients)
                U = torch.stack(reshaped_updates)
                # print(f'shape of per_task_loss = {len(per_task_loss)}')
                L = torch.tensor(per_task_loss).view(-1, 1).cuda()
                # print(f'shape of G = {G.shape}, U = {U.shape}, L = {L.shape}') # G = [N,param], U.T = [param,N], L = [N,1]

                '''check'''
                ita_approximation_G_U = torch.matmul(G, U.T)
                '''wo-loss'''
                # ita_approximation = torch.divide(ita_approximation_G_U, L)
                ita_approximation = torch.divide(ita_approximation_G_U, model_optim.param_groups[0]['lr'])


                for idx, base_task in enumerate(tasks):
                    # Extract the ith column from ita_approximation of base task onto other tasks
                    ita_per_task = ita_approximation[:, idx] #checked this

                    task_gains_approximation[base_task] = {
                        task: ita_per_task[tasks.index(task)] for task in
                        tasks}

                '''Normal model Update'''
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                if (batch_idx + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_idx + 1, epoch + 1, loss.item()))


                # Record batch-level training and gradient metrics.
                for combined_task, task_gain_map in task_gains_approximation.items():
                    for task, gain in task_gain_map.items():
                        batch_grad_metrics[combined_task][task] += gain.cpu().numpy() / (
                            math.ceil(TRAIN_SIZE / BATCH_SIZE))



            avg_train_loss = np.average(train_loss)
            # tr_loss, tr_task_loss = self.vali(train_loader, criterion, mask)
            vali_loss, task_loss = self.vali(vali_loader, criterion, mask)

            print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss}, Vali Loss: {vali_loss}")

            for combined_task, task_gain_map in batch_grad_metrics.items():
                gradient_metrics[combined_task].append(task_gain_map)

            '''check validation loss'''
            early_stopping(vali_loss, self.model, path)

            if early_stopping.counter == 0:
                '''save model'''
                torch.save(self.model.state_dict(), best_model_name)



            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch > 2:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
                print(f'learning rate = {model_optim.param_groups[0]["lr"]}')


        print('save tag approximation')
        print(f'len gradient_metrics = {len(gradient_metrics)}')
        self.model.load_state_dict(torch.load(best_model_name))

        for task in gradient_metrics:
            gradient_metrics[task] = gradient_metrics[task][:-1 * (self.args.patience - 1)]

        timeEnd = time.time()
        Final_timetaken = timeEnd - timeStart

        return self.model,epoch,gradient_metrics,Final_timetaken

    def test(self, setting, mask_, run_idx, w_momentum=False,test=0):

        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        vali_data, vali_loader = self._get_data(flag='val')

        criterion = self._select_criterion()
        mask = torch.from_numpy(mask_).to(self.device)
        # if test:
        # print('loading model')
        if w_momentum:
            best_model_name = f'curr_best_model_tag_approx_w_momentum_{run_idx}.pth'
        else:
            best_model_name = f'curr_best_model_tag_approx_{run_idx}.pth'

        self.model.load_state_dict(torch.load(best_model_name))

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