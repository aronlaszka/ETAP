import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import ast
import math
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', None)

SEEDS = [83, 35, 9, 22,21, 14,  29, 55, 10, 8]

# comparison_method = 'residual'
comparison_method = 'final'
ALL_MODELS = ['BSplines_Ridge',
              'RandomForest', 'KNN']
Dataset = ['celebA','chemical', 'ETTm1','Occupancy']
Boosting_methods = ['Ridge','XGB']
ita_method_name = 'Task_Affinity_Score'
for dataset in Dataset:
    MODELS = ALL_MODELS

    if dataset == 'celebA':
        TASKS = ['5_o_Clock_Shadow', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Goatee', 'Mustache', 'No_Beard',
                 'Rosy_Cheeks', 'Wearing_Hat']
        Training_Samples = [5, 10, 20, 30, 40, 50, 75, 100]

    if dataset == 'chemical':
        TASKS = [83, 78, 84, 85, 76, 86, 81, 80, 87, 55]
        TASKS = [str(t) for t in TASKS]
        Training_Samples = [5, 10, 20, 30, 40, 50, 75, 100]

    if dataset == 'ETTm1':
        TASKS = [i for i in range(7)]
        TASKS = [str(t) for t in TASKS]
        Training_Samples = [5, 7, 10, 20, 30, 40, 45, 50]

    if dataset == 'Occupancy':
        TASKS = ['10C_INBOUND', '10C_OUTBOUND', '13_INBOUND', '13_OUTBOUND', '16_INBOUND', '16_OUTBOUND',
                 '1_INBOUND', '1_OUTBOUND', '7_INBOUND', '7_OUTBOUND']
        print(TASKS)
        TASKS = sorted(TASKS)
        Training_Samples = [5, 10, 15, 20, 25, 30, 40, 50]

    TASKS = sorted((TASKS))
    mtg_data_path = 'PredData/'
    if dataset == 'Occupancy':
        GAIN_SEED = 2024
        train_x = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Train_SEED_{GAIN_SEED}.pt')
        train_y = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_gains_Train_SEED_{GAIN_SEED}.pt')
        testx = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Test_SEED_{GAIN_SEED}.pt')
        testy = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_gains_Test_SEED_{GAIN_SEED}.pt')
        ALL_X = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_tasks_map_SEED_{GAIN_SEED}.pt')
        ALL_Y = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_gains_SEED_{GAIN_SEED}.pt')
    else:
        testx = torch.load(f'PredData/FINAL_RANDOMIZED_{dataset}_tasks_map_test.pt')
        testy = torch.load(f'PredData/FINAL_RANDOMIZED_{dataset}_gain_test.pt')
        train_x = torch.load(f'PredData/RANDOMIZED_{dataset}_tasks_map_from_ground_truth_train.pt')
        train_y = torch.load(f'PredData/RANDOMIZED_{dataset}_gain_collection_from_ground_truth_train.pt')

        ALL_X = torch.load(f'{mtg_data_path}{dataset}_Combined_X.pt')
        ALL_Y = torch.load(f'{mtg_data_path}{dataset}_Combined_Y.pt')



    testx,testy = torch.FloatTensor(testx), torch.FloatTensor(testy)
    train_x,train_y = torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    ALL_X,ALL_Y = torch.FloatTensor(ALL_X), torch.FloatTensor(ALL_Y)

    '''remove test sample where map sum =1'''
    all_indices_test = list(set(range(len(testx))))
    for idx in range(len(testx)):
        if torch.sum(testx[idx]) == 1:
            all_indices_test.remove(idx)
    testx = testx[all_indices_test]
    testy = testy[all_indices_test]

    for boosting_method in Boosting_methods:
        Model_Dataframe = []
        for model_name in MODELS:
            print(f'Method {boosting_method}, base model {model_name}')

            seed_df = []

            Model_Rsq_seed = {}
            Model_Corr_seed = {}
            for seed in SEEDS:
                randomized_Seed = []
                df = []
                test_rsq = []
                test_corr = []
                train_rsq = []
                train_corr = []
                for num_training_sample in Training_Samples:

                    predicted_residuals_file = f'RESULTS/residual_prediction/{dataset}/{ita_method_name}_{boosting_method}_{dataset}_{model_name}_residuals_{num_training_sample}_seed_{seed}.pt'
                    boosting_predictions = torch.load(predicted_residuals_file)
                    ALL_predicted_residuals_file = f'RESULTS/residual_prediction/{dataset}/ALL_{ita_method_name}_{boosting_method}_{dataset}_{model_name}_residuals_{num_training_sample}_seed_{seed}.pt'
                    boosting_predictions_ALL = torch.load(ALL_predicted_residuals_file)

                    tasks_map_file_name = f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}_Maps_{dataset}_{num_training_sample}_seed_{seed}.pkl'
                    with open(tasks_map_file_name, "rb") as f:
                        task_map_train_test = pickle.load(f)
                    '''convert to pytorch tensor'''
                    base_prediction_file = f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}_Predictions_{dataset}_{num_training_sample}_seed_{seed}.pkl'
                    with open(base_prediction_file, 'rb') as f:
                        base_prediction_from_model = pickle.load(f)
                    base_prediction_file_ALL = f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}_ALL_Predictions_{dataset}_{num_training_sample}_seed_{seed}.pkl'
                    with open(base_prediction_file_ALL, 'rb') as f:
                        base_prediction_from_model_ALL = pickle.load(f)
                    trainx = task_map_train_test[:num_training_sample]
                    trainy =  []
                    for each_task_map in trainx:
                        '''get index from all train_dataset'''
                        for idx in range(0,len(train_x)):
                            if torch.equal(train_x[idx], each_task_map):
                                # print(train_x[idx], each_task_map)
                                trainy.append(list(train_y[idx]))
                                break
                    trainy = torch.Tensor(trainy)
                    # print(f'dataset: {dataset}, num_training_sample: {num_training_sample}, shapes: {trainx.shape, trainy.shape}')



                    base_prediction_from_model_train = base_prediction_from_model[:num_training_sample]
                    base_prediction_from_model_test = base_prediction_from_model[num_training_sample:]


                    train_error_count = 0
                    for idx in range(0,len(trainx)):
                        non_zero_idxs_grp = np.nonzero(trainx[idx])[0]
                        non_zero_idxs_grp = [each for each in non_zero_idxs_grp]
                        if torch.sum(trainx[idx]) == 1:
                            print(f'{dataset}, here at single task in training {idx}: train = {trainx[idx]}, label = {trainy[idx]}, prediction = {base_prediction_from_model_train[idx]}')
                            continue
                        non_zero_idxs_preds = np.nonzero(base_prediction_from_model_train[idx])[0]
                        non_zero_idxs_preds = [each for each in non_zero_idxs_preds]
                        non_zero_idx_labels = np.nonzero(trainy[idx])[0]
                        non_zero_idx_labels = [each for each in non_zero_idx_labels]
                        if np.equal(non_zero_idxs_grp, non_zero_idx_labels).all() and np.equal(non_zero_idxs_preds, non_zero_idx_labels).all():
                            pass
                        else:
                            print(f'TRAIN - Raise Error at idx:{idx}')
                            print(f'trainx = {trainx[idx]}, trainy = {trainy[idx]}, base_pred_train = {base_prediction_from_model_train[idx]}')
                            print(non_zero_idxs_grp, non_zero_idxs_preds, non_zero_idx_labels)
                            train_error_count+=1
                            # exit()

                    test_error_count = 0
                    for idx in range(0,len(testx)):
                        non_zero_idxs_grp = np.nonzero(testx[idx])[0]
                        non_zero_idxs_grp = [each for each in non_zero_idxs_grp]

                        non_zero_idx_labels = np.nonzero(testy[idx])[0]
                        non_zero_idx_labels = [each for each in non_zero_idx_labels]

                        if len(np.nonzero(base_prediction_from_model_test[idx]))==0:
                            print(f'testx = {testx[idx]}, testy = {testy[idx]}')
                            print(f'base_prediction_from_model_test[idx]: {base_prediction_from_model_test[idx]}')
                            test_error_count+=1
                            continue
                        non_zero_idxs_preds = np.nonzero(base_prediction_from_model_test[idx])[0]
                        non_zero_idxs_preds = [each for each in non_zero_idxs_preds]

                        if not np.equal(non_zero_idxs_grp, non_zero_idx_labels).all() or not np.equal(non_zero_idxs_preds, non_zero_idx_labels).all():
                            print(f'TEST - Raise Error at idx:{idx}')
                            print(
                                f'testx = {testx[idx]}, testy = {testy[idx]}, base_pred_test = {base_prediction_from_model_test[idx]}')
                            print(non_zero_idxs_grp, non_zero_idxs_preds, non_zero_idx_labels)
                            test_error_count+=1
                            # print(non_zero_idxs_grp, non_zero_idxs_preds, non_zero_idx_labels)
                            # exit()
                    all_error_count = 0
                    for idx in range(0, len(ALL_X)):
                        non_zero_idxs_grp = np.nonzero(ALL_X[idx])[0]
                        non_zero_idxs_grp = [each for each in non_zero_idxs_grp]
                        if torch.sum(ALL_X[idx]) == 1:
                            continue
                        non_zero_idxs_preds = np.nonzero(base_prediction_from_model_ALL[idx])[0]
                        non_zero_idxs_preds = [each for each in non_zero_idxs_preds]
                        non_zero_idx_labels = np.nonzero(ALL_Y[idx])[0]
                        non_zero_idx_labels = [each for each in non_zero_idx_labels]
                        if np.equal(non_zero_idxs_grp, non_zero_idx_labels).all() and np.equal(non_zero_idxs_preds,
                                                                                               non_zero_idx_labels).all():
                            pass
                        else:
                            print(f'ALL - Raise Error at idx:{idx}')
                            print(
                                f'ALL_X = {ALL_X[idx]}, ALL_Y = {ALL_Y[idx]}, base_pred_train = {base_prediction_from_model_ALL[idx]}')
                            print(non_zero_idxs_grp, non_zero_idxs_preds, non_zero_idx_labels)
                            all_error_count += 1

                    if all_error_count > 0:
                        print(f'ALL - finished with train_error_count = {all_error_count}')
                    if test_error_count > 0:
                        print(f'Test - finished with test_error_count = {test_error_count}')
                    if train_error_count == 0 and test_error_count == 0:
                        print('Passed train-test check')

                    Boosting_train_predictions = boosting_predictions[:num_training_sample]
                    Boosting_test_predictions = boosting_predictions[num_training_sample:]
                    Boosting_all_predictions = boosting_predictions_ALL
                    # '''multiply with mask'''
                    # if boosting_method == 'XGB':
                    #     Boosting_train_predictions = Boosting_train_predictions.mul(trainx)
                    #     Boosting_test_predictions = Boosting_test_predictions.mul(testx)




                    '''add residual prediction with model prediction'''
                    if comparison_method == 'residual':
                        final_predictions_test = Boosting_test_predictions
                        final_predictions_train = Boosting_train_predictions
                        testy = torch.load(
                            f'RESULTS/Initial_prediction/{ita_method_name}_{dataset}_{model_name}_residuals_test_{num_training_sample}_seed_{seed}.pt')
                        trainy = torch.load(
                            f'RESULTS/Initial_prediction/{ita_method_name}_{dataset}_{model_name}_residuals_train_{num_training_sample}_seed_{seed}.pt')
                        all_y = torch.load(
                            f'RESULTS/Initial_prediction/{ita_method_name}_{dataset}_{model_name}_residuals_all_{num_training_sample}_seed_{seed}.pt')

                    if comparison_method == 'final':
                        final_predictions_test = base_prediction_from_model_test + Boosting_test_predictions
                        final_predictions_train = base_prediction_from_model_train + Boosting_train_predictions
                        final_predictions_all = base_prediction_from_model_ALL + Boosting_all_predictions
                    # final_predictions_test = Boosting_test_predictions
                    # for each_idx in range(0,len(testx)):
                    #     print(f'task map = {testx[each_idx]}')
                    #     print(f'base prediction = {base_prediction_from_model_test[each_idx]}')
                    #     print(f'Boosting prediction = {Boosting_test_predictions[each_idx]}')
                    #     print(f'final prediction = {final_predictions_test[each_idx]}')
                    #     exit()

                    print(f'shape of testy: {testy.shape}, testx: {testx.shape}')
                    non_zero_test_labels = testy[testx >=1.0]
                    non_zero_test_predictions = final_predictions_test[testx >=1.0]
                    print(f'shape of trainx: {trainx.shape}, trainy: {trainy.shape}')
                    non_zero_train_labels = trainy[trainx >0.]
                    non_zero_train_predictions = final_predictions_train[trainx >0.]

                    if comparison_method == 'final':
                        '''save the final predictions'''
                        if not os.path.exists(f'RESULTS/final_predictions/{dataset}'):
                            os.makedirs(f'RESULTS/final_predictions/{dataset}')

                        entire_predictions_train_test = torch.cat((final_predictions_train, final_predictions_test), dim=0)
                        entire_prediction_file = f'RESULTS/final_predictions/{dataset}/{boosting_method}_{dataset}_{model_name}_final_predictions_{num_training_sample}_seed_{seed}.pkl'
                        with open(entire_prediction_file, 'wb') as f:
                            pickle.dump(entire_predictions_train_test, f)
                        entire_prediction_file = f'RESULTS/final_predictions/{dataset}/ALL_{ita_method_name}{boosting_method}_{dataset}_{model_name}_final_predictions_{num_training_sample}_seed_{seed}.pkl'
                        with open(entire_prediction_file, 'wb') as f:
                            pickle.dump(final_predictions_all, f)





                    # print(non_zero_test_labels.shape, len(non_zero_test_predictions))
                    # exit(0)
                    # print(f'training_samples: {num_training_sample}, test r-squared: {r2_score(non_zero_test_labels, non_zero_test_predictions):0.4f}')
                    test_rsq.append(r2_score(non_zero_test_labels, non_zero_test_predictions))
                    test_corr.append(np.corrcoef(non_zero_test_predictions, non_zero_test_labels)[0,1])

                    train_rsq.append(r2_score(non_zero_train_labels, non_zero_train_predictions))
                    train_corr.append(np.corrcoef(non_zero_train_labels, non_zero_train_predictions)[0, 1])
                    # Task_Embedding_Dimensions.append(num_hidden)
                    # randomized_Seed.append(seed)
                    # exit()
                # test_rsq = [round(rsq, 4) for rsq in test_rsq]
                # test_corr = [round(corr, 4) for corr in test_corr]
                # train_rsq = [round(rsq, 4) for rsq in train_rsq]
                # train_corr = [round(corr, 4) for corr in train_corr]
                # Model_Names = [model_name]*len(train_corr)
                Model_Rsq_seed[seed] = test_rsq
                Model_Corr_seed[seed] = test_corr
                if seed== SEEDS[0]:
                    new_df = pd.DataFrame({'Training_Samples': Training_Samples,
                                             f'Test_Rsq_seed_{seed}': test_rsq,
                                           f'Test_Correlation_seed_{seed}': test_corr,
                                           f'Train_Rsq_seed_{seed}': train_rsq,
                                           f'Train_Correlation_seed_{seed}': train_corr,
                                           f'Base_Models' : [model_name]*len(train_corr),
                                           })

                else:
                    new_df[f'Test_Rsq_seed_{seed}'] = test_rsq
                    new_df[f'Test_Correlation_seed_{seed}'] = test_corr
                    new_df[f'Train_Rsq_seed_{seed}'] = train_rsq
                    new_df[f'Train_Correlation_seed_{seed}'] = train_corr
                    # new_df[f'Base_Models'] = [model_name]*len(train_corr)
            print(f'len(new_df) = {len(new_df)}')
            # Base_Models = MODELS * len(new_df)
            # new_df['Base_Models'] = Base_Models

            Columns = []
            for seed in SEEDS:
                Columns.append(f'Test_Rsq_seed_{seed}')
            for seed in SEEDS:
                Columns.append(f'Train_Rsq_seed_{seed}')
            for seed in SEEDS:
                Columns.append(f'Test_Correlation_seed_{seed}')
            for seed in SEEDS:
                Columns.append(f'Train_Correlation_seed_{seed}')

            Columns = ['Base_Models','Training_Samples'] + Columns

            new_df = new_df[Columns]
            Model_Dataframe.append(new_df)

        Model_Dataframe = pd.concat(Model_Dataframe)
        Model_Dataframe=Model_Dataframe.sort_values(by=['Training_Samples'], ascending=True)
        print(len(Model_Dataframe))
        Model_Dataframe = Model_Dataframe.reset_index(drop=True)
        Avg_rsq = []
        for i in range(len(Model_Dataframe)):
            tot_rsq = 0
            for seed in SEEDS:
                tot_rsq+=Model_Dataframe[f'Test_Rsq_seed_{seed}'][i]
            Avg_rsq.append(tot_rsq/len(SEEDS))
        print(f'Avg_rsq = {Avg_rsq}')
        Model_Dataframe['Test_RSquare'] = Avg_rsq

        if comparison_method == 'residual':
            Final_df = Model_Dataframe[['Base_Models','Training_Samples','Test_RSquare']]

            plt.figure(figsize=(12, 6))
            best_avg_rsq = -math.inf
            from collections import deque

            Winner_List = {}

            for model in Final_df['Base_Models'].unique():
                model_data = Final_df[Final_df['Base_Models'] == model]
                plt.plot(model_data['Training_Samples'], model_data['Test_RSquare'], label=model, marker='o')
                if np.mean(model_data['Test_RSquare'])>best_avg_rsq:
                    best_avg_rsq = np.mean(model_data['Test_RSquare'])
                    print(f'update best model: {model}')
                    Winner_List.update({model:best_avg_rsq})


            plt.title(f'Boosting Model: {boosting_method} {dataset.upper()} - RESIDUAL-PREDICTIONS', fontsize=14)
            plt.xlabel('Training Samples', fontsize=10)
            plt.ylabel('Test R-Square', fontsize=10)
            plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()

            sort_values = sorted(list(Winner_List.values()), reverse=True)
            print(f'{dataset.upper()} - ')
            for idx in range(len(sort_values)):
                for k,v in Winner_List.items():
                    if v == sort_values[idx]:
                        print(f'Base Model Rank {idx+1} : {k}: {v:.5f}')
                        break

            print(f'{dataset.upper()} - \tWinner_List:{Winner_List} \tbest_avg_rsq: {best_avg_rsq:.6f}')
        if comparison_method == 'final':
            Model_Dataframe.to_csv(
                    f'RESULTS/final_predictions/{ita_method_name}_{boosting_method}_{dataset}_Boosting_Final_Results.csv', index=False)
        if comparison_method == 'residual':
            Model_Dataframe.to_csv(f'RESULTS/residual_prediction/{ita_method_name}_{boosting_method}_{dataset}_Boosting_Residual_Results.csv',
                               index=False)

if comparison_method == 'residual':
    plt.show()
