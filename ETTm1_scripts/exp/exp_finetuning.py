from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Autoformer
#, Transformer, Reformer, Informer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import copy

import os
import time
import tqdm
from sklearn.linear_model import LogisticRegression

import warnings
# import matplotlib.pyplot as plt
import numpy as np
fix_seed = 2021
np.random.seed(fix_seed)

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
        model_optim = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
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
                for idx in range(batch_x.shape[2]):
                    task_specific_loss.append(criterion(outputs[:, :, idx], batch_y[:, :, idx]))
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

    def train_model(self, setting, mask_, run_idx):
        train_task_affinity = []
        vali_task_affinity = []
        mask = torch.from_numpy(mask_).to(self.device)
        print(f'mask = {mask}')
        print(f'np.shape(mask) = {np.shape(mask)}')
        tasks = np.where(mask_ == 1)[0]
        print(f'tasks = {tasks}')
        train_data, train_loader = self._get_data(flag='train')
        # print(f'len train_data = {len(train_data)},{np.shape(train_data)}')

        # print(f'len test_data = {len(test_data)}, {np.shape(test_data)}')
        vali_data, vali_loader = self._get_data(flag='val')
        # print(f'len vali_data = {len(vali_data)}, {np.shape(vali_data)}')
        test_data, test_loader = self._get_data(flag='test')
        # exit(0)

        gradients_dir = "Gradients_run_" + str(run_idx)

        print(f'train data: {len(train_data)}, vali data:, {len(vali_data)}, test data:,{len(test_data)}')
        # print(f'setting: {setting}')
        # print(f'self.args.checkpoints: {self.args.checkpoints}')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        '''load best model'''
        self.model.load_state_dict(torch.load(f'curr_best_model_{self.args.frac_num}_run_{run_idx}.pth'))

        grad_params = []
        remove_keys = ["pred_head", "bn"]
        for name, param in self.model.named_parameters():
            if any(key in name for key in remove_keys):
                continue
            # print(name)
            grad_params.append(param)

        gradient_dim = 0
        for param in grad_params:
            gradient_dim += param.numel()
        print("Gradient Dim: {}".format(gradient_dim))

        tmp_time = time.time()

        project_dim = 200
        project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
        project_matrix *= 1 / np.sqrt(project_dim)

        project_matrix = torch.from_numpy(project_matrix).float().to(self.device)
        project_matrix = project_matrix.to(self.device)
        print(f"Time taken to generate project_matrix: {time.time() - tmp_time} seconds")


        gradients = []
        for domain in tasks:
            gradient_file = f"{gradients_dir}/{domain}_train_gradients.npy"
            tmp_gradients = np.load(gradient_file)
            gradients.append(tmp_gradients)
        gradients = np.concatenate(gradients, axis=0)
        print("Gradients Shape: {}".format(gradients.shape))

        # randomly assign labels as 0 or 1
        labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])

        # reverse the gradients for the 0 labels
        mask_grad = np.copy(labels)
        mask_grad[labels == 0] = -1
        mask_grad = mask_grad.reshape(-1, 1)
        gradients = gradients * mask_grad
        train_num = int(len(gradients) * 0.8)
        train_gradients, train_labels = gradients[:train_num], labels[:train_num]
        test_gradients, test_labels = gradients[train_num:], labels[train_num:]

        # train a logistic regression model
        clf = LogisticRegression(random_state=0, penalty='l2', C=1e-4)  #
        clf.fit(gradients, labels)
        print(clf.score(gradients, labels))

        ## %%
        # projection_matrix = np.load(f"./gradients/{args.dataset_key}_{args.model_key}_{args.preset_key}_{args.project_dim}/projection_matrix_{args.run}.npy")
        proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
        '''proj_coef to device'''
        proj_coef = torch.FloatTensor(proj_coef).to(self.device)
        coef = project_matrix @ proj_coef.flatten()
        print(f'coef = {coef.shape}')
        '''convert coef to numpy'''
        coef = coef.cpu().numpy().flatten()

        print("L2 norm", np.linalg.norm(coef))
        coef = coef * 2 / np.linalg.norm(coef)
        print("L2 norm", np.linalg.norm(coef))

        def generate_state_dict(state_dict, coef, removing_keys=["pred_head", "bn"]):
            # reshape coef
            new_state_dict = {}
            cur_len = 0
            for key, param in self.model.named_parameters():
                if not param.requires_grad: continue
                param_len = param.numel()
                if any([rkey in key for rkey in removing_keys]):
                    new_state_dict[key] = state_dict[key].clone()
                else:
                    new_state_dict[key] = state_dict[key].clone() + \
                                          torch.FloatTensor(coef[cur_len:cur_len + param_len].reshape(param.shape)).to(
                                              self.device)
                    cur_len += param_len
            return new_state_dict

        state_dict = copy.deepcopy(self.model.state_dict())
        new_state_dict = generate_state_dict(state_dict, coef)
        pretrain_state_dict = state_dict
        finetuned_state_dict = new_state_dict

        self.model.load_state_dict(pretrain_state_dict)
        self.model.load_state_dict(finetuned_state_dict, strict=False)

        self.args.train_epochs = 0
        epoch = 0

        results_loss = {task: [] for task in tasks}
        for estimate_run in range(1,2):
            model_optim = self._select_optimizer()
            criterion = self._select_criterion()


            iter_count = 0
            train_loss = []

            '''
            self.model.train()
            epoch_time = time.time()

            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(train_loader)):
                # print(f'i = {i}, batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}')
                # print(f'batch_x_mark shape: {batch_x_mark.shape}, batch_y_mark shape: {batch_y_mark.shape}')
                # print (f' first sample of batch_y: {batch_y[0][0]}')
                # print (f' first sample of batch_y_mark: {batch_y_mark[0][0]}')

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # print(f'shape of batch_x: {batch_x.shape}, shape of batch_y: {batch_y.shape}, shape of batch_x_mark: {batch_x_mark.shape}, shape of batch_y_mark: {batch_y_mark.shape}')
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # print(f'first sample of dec_inp: {dec_inp[0][0]}')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask
                        loss = criterion(outputs[outputs!=0], batch_y[outputs!=0])
                        train_loss.append(loss.item())
                        print(f'loss: {loss.item()}')
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask

                    col = np.where(mask_==1)[0]
                    loss = criterion(outputs[:,:,col], batch_y[:,:,col])

                    train_loss.append(loss.item())

                if (batch_idx + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(batch_idx + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    print(f'time_taken: {time.time() - epoch_time}')
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                model_optim.zero_grad()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            '''
            # tr_loss = np.average(train_loss)
            # tr_loss, tr_task_loss = self.vali(train_data, train_loader, criterion, mask)
            # vali_loss, task_loss = self.vali(vali_data, vali_loader, criterion, mask)
            # test_loss, per_task_loss = self.vali(test_data, test_loader, criterion, mask)
            # print(f'tr_loss: {tr_loss}, vali_loss: {vali_loss}, test_loss: {test_loss}')
            # print(f'tr_task_loss shape: {tr_task_loss.shape}, task_loss shape: {task_loss.shape}, '
            #       f'\nper_task_loss shape: {per_task_loss.shape}')
            mask_test = np.zeros((7), )
            mask_test[np.array(tasks)] = 1
            print(f'mask_test = {mask_test}')

            per_mae, per_mse = self.test(setting, mask_test)
            print(f'per_mse: {per_mse}')

            test_metrics_loss = {task: per_mse[task] for task in tasks}
            print(f'test_metrics_loss = {test_metrics_loss}')

            for key, value in test_metrics_loss.items():
                results_loss[key].append(value)

            print(f'estimate_run = {estimate_run}, results_loss = {results_loss}')


        '''get average loss for each task'''
        for key, value in results_loss.items():
            results_loss[key] = np.average(value)
        # print(f'results_loss = {results_loss}')
        # exit(0)

        return results_loss

    def save_gradients(self, mask_, run_idx, train_loader):

        mask = torch.from_numpy(mask_).to(self.device)
        gradients_dir = "Gradients_run_" + str(run_idx)
        if not os.path.exists(gradients_dir):
            os.makedirs(gradients_dir)

        print(f'gradients_dir = {gradients_dir}')
        tasks = list(range(len(mask)))
        task_gradients = {task: [] for task in tasks}

        grad_params = []
        remove_keys = ["pred_head", "bn"]
        for name, param in self.model.named_parameters():
            if any(key in name for key in remove_keys):
                continue
            # print(name)
            grad_params.append(param)

        gradient_dim = 0
        for param in grad_params:
            gradient_dim += param.numel()
        print("Gradient Dim: {}".format(gradient_dim))

        tmp_time = time.time()

        project_dim = 200
        project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
        project_matrix *= 1 / np.sqrt(project_dim)

        project_matrix = torch.from_numpy(project_matrix).float().to(self.device)
        project_matrix = project_matrix.to(self.device)
        print(f"Time taken to generate project_matrix: {time.time() - tmp_time} seconds")

        # tmp_time = time.time()
        # # Generate a random matrix using a faster method (use random.randn instead of randint)
        # project_matrix = np.random.randn(gradient_dim, project_dim).astype(np.float32)
        # # Normalize the matrix in one step (instead of scaling each element)
        # project_matrix /= np.sqrt(project_dim)
        # # Convert to a PyTorch tensor and move to the correct device
        # project_matrix = torch.from_numpy(project_matrix).to(self.device)
        # print(f"Projection matrix generated in {time.time() - tmp_time:.4f} seconds")

        # Save gradients

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        start_time = time.time()


        print(f'Saving Gradients for run_{run_idx}')

        self.model.eval()
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm.tqdm(enumerate(train_loader)):
            self.model.load_state_dict(torch.load(f'Best_Model/checkpoint_{self.args.frac_num}_run_{run_idx}.pth'))

            single_task_specific_gradients = []
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # print(f'shape of batch_x: {batch_x.shape}, shape of batch_y: {batch_y.shape}, shape of batch_x_mark: {batch_x_mark.shape}, shape of batch_y_mark: {batch_y_mark.shape}')
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # print(f'first sample of dec_inp: {dec_inp[0][0]}')

            # encoder - decoder
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

                    for idx in range(batch_x.shape[-1]):
                        model_optim.zero_grad()
                        task_loss = criterion(outputs[:, :, idx], batch_y[:, :, idx])
                        # Calculate task-specific gradients before update
                        task_loss.backward(retain_graph=True)
                        # Store the gradients
                        task_gradients = [param.grad.clone() for param in self.model.parameters()]
                        single_task_specific_gradients.append((idx, task_gradients))

            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:] * mask
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) * mask

                col = np.where(mask_ == 1)[0]
                loss = criterion(outputs[:, :, col], batch_y[:, :, col])

                for idx in range(batch_x.shape[-1]):
                    task_loss = criterion(outputs[:, :, idx], batch_y[:, :, idx])
                    # Calculate task-specific gradients before update
                    task_loss.backward(retain_graph=True)
                    # Store the gradients
                    # per_task_gradients = [param.grad.clone() for param in self.model.parameters()]
                    per_task_gradients = [param.grad for param in self.model.parameters()]
                    tmp_gradients = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None],
                                              dim=0).cpu().numpy()
                    # Convert tmp_gradients from numpy array to PyTorch tensor
                    tmp_gradients = torch.from_numpy(tmp_gradients).float().to(self.device)
                    tmp_gradients = torch.matmul(tmp_gradients.reshape(1, -1), project_matrix).flatten()

                    # # single_task_specific_gradients.append((idx, per_task_gradients))
                    #
                    # '''flatten and concatenate gradients'''
                    # # tmp_gradients = torch.cat([gradient.view(-1) for gradient in per_task_gradients]).cpu().numpy()
                    #
                    # if tmp_gradients.size != project_matrix.shape[0]:
                    #     raise ValueError(
                    #         f"Gradient size {tmp_gradients.size} does not match expected size {project_matrix.shape[0]}")
                    #
                    # tmp_gradients = (tmp_gradients.reshape(1, -1) @ project_matrix).flatten()
                    task_gradients[tasks[idx]].append(tmp_gradients)
                    model_optim.zero_grad()

                # if batch_idx == 100:
                #     break


        for task_name, gradients in task_gradients.items():
            gradients = [grad.cpu().numpy() for grad in gradients]
            gradients = np.array(gradients)
            # print(f'gradients shape = {gradients.shape}')
            np.save(f"{gradients_dir}/{task_name}_train_gradients.npy", gradients)

        end_time = time.time()
        print(f"Time taken for save gradients: {end_time - start_time}")

    def test(self, setting, mask_, test=0):

        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        vali_data, vali_loader = self._get_data(flag='val')

        criterion = self._select_criterion()
        mask = torch.from_numpy(mask_).to(self.device)
        # if test:

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # test perf
        preds = []
        trues = []

        with torch.no_grad():
            for test_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                # if test_idx % 50 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(test_idx) + '.pdf'))

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

        # print('per mae:{}, per mse{}'.format(per_mae, per_mse))
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open("result_pairs.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('\n\n' + 'task' + str(mask_) + '\n')
        # f.write('train mse:{}, train mae:{}, '.format(tr_mse, tr_mae) + '\n')
        # f.write('val mse:{}, val mae:{}, '.format(val_mse, val_mae))
        # f.write('val per mae:{}, val per mse{}, '.format(val_per_mae, val_per_mse) + '\n')
        # f.write('mse:{}, mae:{}, '.format(mse, mae))
        # f.write('per mae:{}, per mse{}'.format(per_mae, per_mse))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'test_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'test_pred.npy', preds)
        # np.save(folder_path + 'test_true.npy', trues)

        return per_mae, per_mse
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