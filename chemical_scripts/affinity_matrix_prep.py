import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def prepare_fifty_ita_matrix(TASKS,ITA_file):
    '''read line by line'''
    gain_matrix = np.array([[0.0 for i in range(len(TASKS))] for j in range(len(TASKS))])

    gain_dict = {task:[] for task in TASKS}

    with open(ITA_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',[')
            task_name = int(line[0])
            if task_name not in TASKS:
                continue
            task_idx = TASKS.index(task_name)
            all_epochs = line[1].split('}, {')
            tot_line = 0
            for each_epoch in all_epochs:
                each_epoch = each_epoch.split(',')

                task_no = 0
                for each_task in each_epoch:
                    each_task = each_task.replace('}]', '')
                    each_task = each_task.split(':')
                    gain_dict[TASKS[task_idx]].append(float(each_task[1]))

                    gain_matrix[task_idx][task_no] = gain_matrix[task_idx][task_no] + float(each_task[1])
                    task_no += 1

                tot_line = tot_line + 1

    gain_matrix_dict = pd.DataFrame.from_dict(gain_matrix)
    gain_matrix_dict.columns = TASKS
    return gain_matrix_dict


if __name__ == '__main__':

    TASKS = [83, 78, 84, 85, 76, 86, 81, 80, 87, 55]
    Arch = '_Arch_Arch_1'
    '''get original ITA scores'''
    w_momentum = False
    Method = 'ITA_Approx'
    Tot_run = 6
    Folds = 10
    MethodName = 'ETAP_Affinity'
    Runs = []
    for run in Runs:
        Avg_ITA_dict = {task: {task: 0.0 for task in TASKS} for task in TASKS}

        if Method == 'ITA':
            df_filename = f'chem_results_new/ITA/gradient_metrics_ITA_run_{run}{Arch}_FIXED_{MethodName}ALL.csv'
        elif Method == 'ETAP_Affinity' and w_momentum:
            df_filename = f'chem_results_new/ITA/gradient_metrics_{Method}_run_{run}_w_momentum{Arch}_FIXED_{MethodName}ALL.csv'
        elif Method == 'ETAP_Affinity' and not w_momentum:
            df_filename = f'chem_results_new/ITA/gradient_metrics_{Method}_run_{run}_wo_momentum{Arch}_FIXED_{MethodName}ALL.csv'

        method_dict = prepare_fifty_ita_matrix(TASKS, df_filename)
        method_dict = np.array(method_dict)

        '''copy revised_integrals to Avg_ITA_dict'''
        for idx in range(0,len(TASKS)):
            for jdx in range(0,len(TASKS)):
                Avg_ITA_dict[TASKS[idx]][TASKS[jdx]]=method_dict[idx][jdx]

        print(f'Avg_ITA_dict = {Avg_ITA_dict}')


        method_dict = pd.DataFrame.from_dict(Avg_ITA_dict)
        method_dict.columns = TASKS
        if Method == 'ETAP_Affinity' and w_momentum:
            method_dict.to_csv(f'chem_results_new/ITA/{Method}_w_momentum_matrix_run_{run}_FIXED_ALL.csv', index=False)
        elif Method == 'ETAP_Affinity' and not w_momentum:
            method_dict.to_csv(f'chem_results_new/ITA/{Method}_wo_momentum_matrix_run_{run}_FIXED_ALL.csv', index=False)

    name_idx = 0
    for Method in ['ETAP_Affinity_w_momentum', 'ETAP_Affinity_wo_momentum']:
        Run_times = []
        start_idx = 1
        ITA_Dict = {}
        Tot_run = len(Runs)
        start_idx = Runs[0]
        AVG_CORR = []
        print(f'\n*********{Method.upper()} RUNS')
        for run in Runs:
            if Method == 'ITA_Approx_w_momentum':
                method_file = pd.read_csv(f'chem_results_new/ITA/{Method}_matrix_run_{run}_FIXED_ALL.csv')
            else:
                MethodName = 'ITA_Approx_'
                method_file = pd.read_csv(f'chem_results_new/ITA/{Method}_matrix_run_{run}_FIXED_ALL.csv')


            ITA_Dict[run] = method_file
            '''avg over all runs'''
            if run == start_idx:
                avg_matrix = method_file
            else:
                avg_matrix = avg_matrix + method_file

            pair_run = 'Avg'
            pairwise_affinities = pd.read_csv(f'chem_results_new/Chemical_Pairwise_Affinity_run_{pair_run}.csv')
            pairwise_affinities = pairwise_affinities.to_numpy()
            '''remove the diagonal elements'''
            task_num = len(TASKS)
            diagonal_indices = np.arange(task_num) * task_num + np.arange(task_num)
            filtered_pairwise_affinity = np.delete(pairwise_affinities.flatten(), diagonal_indices)
            filtered_ITA = np.delete(method_file.values.flatten(), diagonal_indices)
            print(f'run = {run}, correlation between pairwise affinity and ({Method}) = {np.corrcoef(filtered_pairwise_affinity, filtered_ITA)[0][1]}')
            corr = np.corrcoef(filtered_pairwise_affinity, filtered_ITA)[0][1]
            AVG_CORR.append(corr)

        avg_matrix = avg_matrix/Tot_run
        avg_matrix_df = pd.DataFrame(avg_matrix)
        avg_matrix_df.columns = TASKS
        avg_matrix_df.to_csv(f'chem_results_new/ITA/{Method}_matrix_avg_FIXED_ALL.csv', index=False)

        print(f'Final Average Correlation: {np.mean(AVG_CORR)} $\pm$ {np.std(AVG_CORR)}')


