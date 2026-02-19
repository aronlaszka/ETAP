import pickle
import numpy as np
import pandas as pd
import torch
import copy
import time
import ast
import os

def select_groups_optimized(index, cur_group, cur_task_set, best_group, best_val, splits):
    # Check if this group covers all tasks.
    if len(cur_task_set) == num_tasks:
        best_tasks = {task: -1e8 for task in cur_task_set}

        # Compute the per-task best scores for each task and average them together.
        for group in cur_group:
            for task in cur_group[group]:
                best_tasks[task] = max(best_tasks[task], cur_group[group][task])
        group_avg = np.mean(list(best_tasks.values()))

        # Compare with the best grouping seen thus far.
        if group_avg > best_val[0]:
            best_val[0] = group_avg
            best_group.clear()
            best_group.update(cur_group)

    # Base case.
    if len(cur_group) == splits:
        return

    # Back to combinatorics
    for i in range(index, len(rtn_tup)):
        selected_group, selected_dict = rtn_tup[i]

        new_group = cur_group.copy()
        new_group[selected_group] = selected_dict

        new_task_set = cur_task_set.union(selected_group.split('|'))

        if len(new_group) <= splits:
            select_groups_optimized(i + 1, new_group, new_task_set, best_group, best_val, splits)

if __name__ == '__main__':
    # boosting_method = 'XGB'

    boosting_method = 'Ridge'
    SEEDS = [83, 22, 14, 29, 55, 10]

    ALL_Base_Models = ['BSPlines_Ridge','GaussianProcess','KNN','RandomForest']
    for base_model in ALL_Base_Models:
        group_selection_folder = f'{base_model}_{boosting_method}'
        if not os.path.exists(f'RESULTS/GROUP_SELECTION/{group_selection_folder}'):
            os.makedirs(f'RESULTS/GROUP_SELECTION/{group_selection_folder}')
        Groups_Path = f'RESULTS/GROUP_SELECTION/{group_selection_folder}'

        print(f'Processing for {base_model}')

        Datasets = ['celebA','chemical','ETTm1','Occupancy']
        for dataset in Datasets:
            if dataset == 'celebA':
                TASKS = ['5_o_Clock_Shadow', 'Black_Hair','Blond_Hair', 'Brown_Hair','Goatee', 'Mustache', 'No_Beard','Rosy_Cheeks', 'Wearing_Hat']
                Training_Samples = [5, 10, 20, 30, 40, 50, 75, 100]
            if dataset =='chemical':
                TASKS = [83, 78, 84, 85, 76, 86, 81, 80, 87, 55]
                TASKS = [str(t) for t in TASKS]
                Training_Samples = [5, 10, 20, 30, 40, 50, 75, 100]

            if dataset =='ETTm1':
                TASKS = [i for i in range(7)]
                TASKS = sorted(TASKS)
                TASKS = [str(t) for t in TASKS]
                Training_Samples = [5, 7, 10, 20, 30, 40, 45, 50]

            if dataset == 'Occupancy':
                TASKS = ['10C_INBOUND', '10C_OUTBOUND', '13_INBOUND', '13_OUTBOUND', '16_INBOUND', '16_OUTBOUND',
                         '1_INBOUND', '1_OUTBOUND', '7_INBOUND', '7_OUTBOUND']
                print(TASKS)
                TASKS = sorted(TASKS)
                Training_Samples = [5, 10, 15, 20, 25, 30, 40, 50]


            Training_Samples = Training_Samples[:5]
            All_selected_groups = []
            for training_sample in Training_Samples:
                mtg_data_path = 'PredData/'
                if dataset == 'Occupancy':
                    GAIN_SEED = 2024
                    testx = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Test_SEED_{GAIN_SEED}.pt')
                    testy = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_gains_Test_SEED_{GAIN_SEED}.pt')

                else:
                    testx = torch.load(f'{mtg_data_path}/FINAL_RANDOMIZED_{dataset}_tasks_map_test.pt')
                    testy = torch.load(f'{mtg_data_path}FINAL_RANDOMIZED_{dataset}_gain_test.pt')


                testx,testy = torch.FloatTensor(testx), torch.FloatTensor(testy)


                if dataset == 'chemical':
                    ALLx_selected = torch.load(f'{mtg_data_path}{dataset}_Combined_X_selected.pt')
                    ALLy_selected = torch.load(f'{mtg_data_path}{dataset}_Combined_Y_selected.pt')
                    ALLx_selected,ALLy_selected = torch.FloatTensor(ALLx_selected), torch.FloatTensor(ALLy_selected)
                    ALLx, ALLy = ALLx_selected, ALLy_selected
                elif dataset == 'Occupancy':
                    ALLx_selected = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_tasks_map_SEED_{GAIN_SEED}.pt')
                    ALLy_selected = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_gains_SEED_{GAIN_SEED}.pt')
                    ALLx_selected,ALLy_selected = torch.FloatTensor(ALLx_selected), torch.FloatTensor(ALLy_selected)
                    ALLx, ALLy = ALLx_selected, ALLy_selected
                else:
                    ALLx = torch.load(f'{mtg_data_path}{dataset}_Combined_X.pt')
                    ALLy = torch.load(f'{mtg_data_path}{dataset}_Combined_Y.pt')
                    ALLx, ALLy = torch.FloatTensor(ALLx), torch.FloatTensor(ALLy)

                if dataset == 'chemical':
                    Only_groups = []
                    for idx in range(0,len(testx)):
                        active_tasks = testx[idx]  # Which tasks are active (1.0)?
                        # label_values = testy[idx]  # Ground-truth values
                        # Find the active tasks
                        active_task_indices = torch.where(active_tasks == 1.0)[0]  # Indices of active tasks
                        active_task_names = [TASKS[i] for i in active_task_indices]  # Get their names

                        if len(active_task_names) == 1:  # If a single task is active
                            task = active_task_names[0]
                            combination_name = task
                        else:  # For combinations of tasks
                            combination_name = "|".join(active_task_names)  # Combine task names
                        Only_groups.append(combination_name)

                    print(f'len(Only_groups) = {len(Only_groups)}')

                # for num_hidden in Task_Embedding_Dimensions:
                for seed in SEEDS:
                    print(f'Processing for base_model {base_model} -- Boosting {boosting_method} -- seed {seed} -- training samples {training_sample}')

                    tasks_map_file_name = f'RESULTS/Initial_prediction/Task_Affinity_Score_{base_model}_Maps_{dataset}_{training_sample}_seed_{seed}.pkl'
                    pred_file_name = f'RESULTS/final_predictions/{dataset}/ALL_Task_Affinity_Score{boosting_method}_{dataset}_{base_model}_final_predictions_{training_sample}_seed_{seed}.pkl'

                    with open(tasks_map_file_name,'rb') as f:
                        tasks_map = pickle.load(f)
                        tasks_map_all_GT = torch.tensor(tasks_map)
                    with open(pred_file_name,'rb') as f:
                        pred_data = pickle.load(f)
                        pred_data_all_GT = torch.tensor(pred_data)

                    # tasks_map_all_test = tasks_map_all_GT[training_sample:]
                    # pred_data_all_test = pred_data_all_GT[training_sample:]
                    tasks_map_all_test = ALLx
                    pred_data_all_test = pred_data_all_GT




                    '''only take test-dataset'''
                    Predicted_Gains = {}
                    count = 0
                    for idx in range(0,len(ALLx)):
                        active_tasks = ALLx[idx]  # Which tasks are active (1.0)?
                        pred_values = pred_data_all_test[idx]  # Predicted values
                        # Find the active tasks
                        active_task_indices = torch.where(active_tasks == 1.0)[0]  # Indices of active tasks
                        active_task_names = [TASKS[i] for i in active_task_indices]  # Get their names

                        if len(active_task_names) == 1:  # If a single task is active
                            task = active_task_names[0]
                            if dataset == 'chemical':
                                if task in Only_groups:
                                    Predicted_Gains[task] = {task: -1e8}
                            else:
                                Predicted_Gains[task] = {task: -1e8}
                        else:  # For combinations of tasks
                            count += 1
                            combination_name = "|".join(active_task_names)  # Combine task names
                            task_contributions = {
                                TASKS[i]: float(pred_values[i])  # predicted_gains
                                for i in active_task_indices
                            }
                            if dataset == 'chemical' or dataset == 'Occupancy':
                                if combination_name in Only_groups:
                                    Predicted_Gains[combination_name] = task_contributions
                            else:
                                Predicted_Gains[combination_name] = task_contributions
                    print(f'length of Predicted_Gains: {len(Predicted_Gains)}')

                    Splits = [2, 3, 4]
                    Groups = []
                    Scores = []
                    Indiv_Scores = []
                    SPLITS = []
                    for splits in Splits:
                        tasks = TASKS
                        num_tasks = len(tasks)

                        rtn = copy.deepcopy(Predicted_Gains)
                        print(f'\n\nTask-Groups {len(rtn)}')

                        print(f'len rtn.keys() = {len(rtn.keys())}')

                        # assert (len(rtn.keys()) == 2 ** len(TASKS) - 1)
                        rtn_tup = [(key, val) for key, val in rtn.items()]
                        # print(rtn_tup)
                        selected_group = {}
                        selected_val = [-100000000]
                        time_start = time.time()
                        select_groups_optimized(index=0, cur_group={}, cur_task_set=set(), best_group=selected_group,
                                                best_val=selected_val, splits=splits)
                        time_end = time.time()
                        print(f'required time = {time_end - time_start}')
                        print(f' selected_group = {selected_group}')
                        print(f' selected_val = {selected_val}')

                        print(f'*************')

                        for group_str in selected_group.keys():
                            SPLITS.append(splits)
                            group = group_str.split('|')
                            Groups.append(group)
                            Scores.append(selected_val[0])
                            print(
                                f'group = {group}, score = {selected_val[0]}, \nselected_group = {selected_group[group_str]}')
                            Indiv_Scores.append(selected_group[group_str])

                            All_selected_groups.append(group)
                    res = pd.DataFrame({'SPLITS': SPLITS, 'Groups': Groups, 'Indiv_Scores': Indiv_Scores})
                    res.to_csv(f'{Groups_Path}/Final_{base_model}_{boosting_method}_{dataset.upper()}_Predicted_Groups_OnlyTest_with_TrainingSample_{training_sample}_seed_{seed}.csv', index=False)