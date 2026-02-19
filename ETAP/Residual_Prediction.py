import copy
import tqdm
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import random
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to train XGBoost with hyperparameter tuning
def train_xgboost(X_train, y_train, X_test, y_test,ALL_X, ALL_Y):
    # Define XGBoost model
    print(f'MODEL NAME: XGBoost')
    model = xgb.XGBRegressor(objective='reg:squarederror')
    # model = xgb.XGBRegressor(objective=masked_loss)

    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [i for i in range(10, 100, 10)],
        'max_depth': [i for i in range(2,5)],
        'learning_rate': [0.001, 0.01, 0.05],
        'subsample': [0.2,0.4, 0.6, 0.7,],
        'colsample_bytree': [0.2, 0.4, 0.6, 0.8]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=3, scoring='neg_mean_squared_error',
                               verbose=1, n_jobs=-1)
    # Fit model with training data
    grid_search.fit(X_train, y_train)
    # Get best model from grid search
    best_model = grid_search.best_estimator_

    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    print(best_model.score(X_test, y_test))
    print(f'shape of y_pred: {y_pred.shape}')

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rsquare = r2_score(y_test, y_pred)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Test Mean Squared Error:", mse)
    print("Test R-Squared:", rsquare)
    y_test_non_zero = y_test[X_test != 0]
    y_pred_non_zero = y_pred[X_test != 0]
    print(f'shapes: {y_pred_non_zero.shape}, {y_test_non_zero.shape}')
    print(f'R-Square: {r2_score(y_test_non_zero, y_pred_non_zero)}')
    print(f'correlation: {np.corrcoef(y_pred_non_zero, y_test_non_zero)[0,1]}')
    # print(y_test_non_zero[:5])
    # print(y_pred_non_zero[:5])
    print(f'mean squared error: {mean_squared_error(y_test_non_zero, y_pred_non_zero)}')
    mean_predictions = [np.mean(y_pred_non_zero) for each in y_pred_non_zero]

    test_mse = mean_squared_error(y_test_non_zero, y_pred_non_zero)
    mean_mse  = mean_squared_error(y_test_non_zero, mean_predictions)
    test_rsq = r2_score(y_test_non_zero, y_pred_non_zero)
    mean_rsq = r2_score(y_test_non_zero, mean_predictions)
    test_corr = np.corrcoef(y_test_non_zero, y_pred_non_zero)[0,1]
    print(f'meAN mse: {mean_mse}')


    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    all_pred = best_model.predict(ALL_X)
    # y_train_pred = torch.Tensor(y_train_pred)
    # y_test_pred = torch.Tensor(y_test_pred)

    # y_train_pred = y_train_pred*X_train
    # y_test_pred = y_test_pred*X_test
    hp_dict = grid_search.best_params_
    y_train_pred[X_train == 0] = 0
    y_test_pred[X_test == 0] = 0
    all_pred[X_train == 0] = 0

    return test_mse, test_rsq, test_corr, mean_mse, mean_rsq,y_train_pred, y_test_pred,  hp_dict, all_pred

def train_multioutput_ridge_regression(X_train, y_train, X_test, y_test, ALL_X, ALL_Y):
    # print(f'MODEL NAME: Multi-output Ridge Regression')
    model = MultiOutputRegressor(Ridge())
    param_grid = {'estimator__alpha': np.logspace(-6, 2, 20)} # Search alpha from 10^-6 to 10^6

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f'best params: {grid_search.best_params_}')
    print(f'Best cross-validation score: {-grid_search.best_score_}')

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # print(f'only considering loss for non-zero labels')
    y_pred_train = best_model.predict(X_train)
    y_pred_non_zero = y_pred_train[X_train != 0]
    y_train_non_zero = y_train[X_train != 0]
    # print(f'shapes: {y_pred_non_zero.shape}, {y_train_non_zero.shape}')

    train_rsq = r2_score(y_train_non_zero, y_pred_non_zero)
    train_correlation = np.corrcoef(y_train_non_zero, y_pred_non_zero)[0, 1]
    train_mse = mean_squared_error(y_train_non_zero, y_pred_non_zero)
    print(f'Train R-Square: {train_rsq}, Train Correlation: {train_correlation}, Train MSE: {train_mse}')

    y_pred_test = best_model.predict(X_test)
    y_pred_non_zero = y_pred_test[X_test != 0]
    y_test_non_zero = y_test[X_test != 0]
    # print(f'shapes: {y_pred_non_zero.shape}, {y_test_non_zero.shape}')
    test_rsq = r2_score(y_test_non_zero, y_pred_non_zero)
    test_correlation = np.corrcoef(y_test_non_zero, y_pred_non_zero)[0, 1]
    test_mse = mean_squared_error(y_test_non_zero, y_pred_non_zero)
    print(f'Test R-Square: {test_rsq}, Test Correlation: {test_correlation}, Test MSE: {test_mse}')

    # print("Model coefficients for each task:")
    # for i, estimator in enumerate(best_model.estimators_):
    #     print(f"Task {i + 1} coefficients:")
    #     # Coefficients for each task
    #     coefficients = estimator.coef_
    #     # Check for large coefficients (magnitude)
    #     large_coefficients = coefficients[abs(coefficients) > 0.1]  # Example threshold
    #     print(f"Large coefficients (> 0.1) for task {i + 1}: {large_coefficients}")
    #     # Check for small coefficients (near-zero)
    #     small_coefficients = coefficients[abs(coefficients) < 0.01]  # Example threshold
    #     print(f"Small coefficients (< 0.01) for task {i + 1}: {small_coefficients}")


    #applying mask
    y_pred_train[X_train == 0] = 0
    y_pred_test[X_test == 0] = 0
    All_pred = best_model.predict(ALL_X)
    All_pred_non_zero = All_pred[ALL_X != 0]
    All_pred[ALL_X == 0] = 0

    return test_mse, test_rsq, test_correlation, y_pred_train, y_pred_test, grid_search.best_params_,All_pred




Datasets = ['celebA','chemical', 'ETTm1','Occupancy']
SEEDS = [83, 35, 9, 22,21, 14,  29, 55, 10, 8]
ALL_MODELS = ['BSplines_Ridge','RandomForest', 'KNN']

# boosting_model = 'XGB'
boosting_model = 'Ridge'
ita_method = 'Task_Affinity_Score_'

for dataset in Datasets:
    if dataset == 'celebA':
        MODELS = ALL_MODELS
        print(MODELS)
    else:
        MODELS = ALL_MODELS

    DATASET_LIST = []
    Test_rsq_list = []
    Test_mse_list = []
    Test_corr_list = []
    Mean_mse_list = []
    Mean_rsq_list = []
    ALL_HP_List = []
    Training_Samples_List = []
    SEEDS_LIST = []
    Base_Model_List = []
    for seed in tqdm.tqdm(SEEDS):
        seed_everything(seed)
        if dataset == 'ETTm1':
            Training_Samples = [5, 7, 10, 20, 30, 40, 45, 50]

        elif dataset == 'Occupancy':
            Training_Samples = [5, 10, 15, 20, 25, 30, 40, 50]
        else:
            Training_Samples = [5, 10, 20, 30, 40, 50, 75, 100]

        Training_Samples = Training_Samples
        for model_name in MODELS:
            for num_training_sample in Training_Samples:
                mtg_data_path = f'PredData/'
                if dataset == 'Occupancy':
                    GAIN_SEED = 2024
                    X_train = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Train_SEED_{GAIN_SEED}.pt')
                    X_test = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Test_SEED_{GAIN_SEED}.pt')
                    all_groups = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_tasks_map_SEED_{GAIN_SEED}.pt')
                    all_labels = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_gains_SEED_{GAIN_SEED}.pt')
                else:
                    X_train = torch.load(f'PredData/RANDOMIZED_{dataset}_tasks_map_from_ground_truth_train.pt')
                    X_test = torch.load(f'PredData/FINAL_RANDOMIZED_{dataset}_tasks_map_test.pt')

                    all_groups = torch.load(f'{mtg_data_path}{dataset}_Combined_X.pt')
                    all_labels = torch.load(f'{mtg_data_path}{dataset}_Combined_Y.pt')

                y_test = torch.load(f'RESULTS/Initial_prediction/{ita_method}{dataset}_{model_name}_residuals_test_{num_training_sample}_seed_{seed}.pt')
                print(f'reading RESULTS/Initial_prediction/{ita_method}{dataset}_{model_name}_residuals_test_{num_training_sample}_seed_{seed}.pt')
                X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)


                '''remove test sample where map sum =1'''
                all_indices_test = list(set(range(len(X_test))))
                for idx in range(len(X_test)):
                    if torch.sum(X_test[idx]) == 1:
                        all_indices_test.remove(idx)
                X_test = X_test[all_indices_test]
                # y_test = y_test[all_indices_test]

                Num_Tasks = X_train.shape[1]
                # print(f'dataset {dataset} has {Num_Tasks} tasks')

                if num_training_sample <= len(X_train):
                    tasks_map_file_name = f'RESULTS/Initial_prediction/Task_Affinity_Score_{model_name}_Maps_{dataset}_{num_training_sample}_seed_{seed}.pkl'
                    with open(tasks_map_file_name, "rb") as f:
                        tasks_map = pickle.load(f)

                    # print(tasks_map.shape)

                    '''convert to pytorch tensor'''
                    tasks_map = torch.tensor(tasks_map)
                    new_X_train = tasks_map[:num_training_sample]

                    '''check if baseline model exists'''
                    found_flag = False
                    # new_X_train = new_X_train.numpy()
                    for each in new_X_train:
                        # print(each)
                        if torch.sum(each) >= Num_Tasks:
                            # if np.sum(each) >= Num_Tasks:
                            found_flag = True
                    if found_flag:
                        print(f'baseline model already exists')
                    else:
                        print(f'{dataset}\tadding the baseline model to the training data')
                        for i in range(len(X_train)):
                            if ((len(X_train[i][X_train[i] == 1]) == len(X_train[0]))):
                                all_task_idx = i
                                group = X_train[i]
                        '''remove one at random from training data'''
                        get_random_index = random.sample(range(len(new_X_train)), 1)[0]
                        print(f'get_random_index  = {get_random_index}, all_task_idx = {all_task_idx}')
                        new_X_train = np.delete(new_X_train, get_random_index, 0)
                        '''check if baseline model exists'''
                        for each in new_X_train:
                            if torch.sum(each) >= Num_Tasks:
                                found_flag = True
                    X_train = new_X_train.numpy()

                y_train = torch.load(f'RESULTS/Initial_prediction/{ita_method}{dataset}_{model_name}_residuals_train_{num_training_sample}_seed_{seed}.pt')


                '''convert y_train to numpy ndarray'''
                y_train = y_train.numpy()
                for idx in range(0, len(X_train)):
                    # print(f'X : {X_train[idx]}, y : {y_train[idx]}')
                    ## check non-zero index for both X and y
                    nonzero_index_X = np.nonzero(X_train[idx])[0]
                    nonzero_index_X = [each for each in nonzero_index_X]
                    nonzero_index_y = np.nonzero(y_train[idx])[0]
                    nonzero_index_y = [each for each in nonzero_index_y]
                    # print(f'{nonzero_index_X} {nonzero_index_y}')
                    if len(nonzero_index_X) == 1:
                        continue
                    ## check if all elem matches
                    if len(nonzero_index_X) != len(nonzero_index_y):
                        print(f'TRAIN: Raise Error at idx = {idx}')
                        print(f'X: {nonzero_index_X}')
                        print(f'y: {nonzero_index_y}')
                        if dataset == 'ETTm1' and model_name == 'KNN':
                            continue
                        # exit()
                    else:
                        # print(f'X: {nonzero_index_X}')
                        # print(f'y: {nonzero_index_y}')
                        ## check if all value equal
                        if not np.equal(nonzero_index_X, nonzero_index_y).all():
                            #     if nonzero_index_X[each_i] != nonzero_index_y[each_i]:
                            print(f'TRAIN: Raise Error: elem does not match')
                            print(f'X: {nonzero_index_X}')
                            print(f'y: {nonzero_index_y}')
                            exit()

                y_test = y_test.numpy()
                for idx in range(0, len(X_test)):
                    # print(f'X : {X_train[idx]}, y : {y_train[idx]}')
                    ## check non-zero index for both X and y
                    nonzero_index_X = np.nonzero(X_test[idx])[0]
                    nonzero_index_y = np.nonzero(y_test[idx])[0]
                    # print(f'{len(nonzero_index_X)} {len(nonzero_index_y)}')
                    ## check if all elem matches
                    if len(nonzero_index_X) == 1:
                        continue
                    if len(nonzero_index_X) != len(nonzero_index_y):
                        print(f'TEST: Raise Error at idx = {idx}')
                        print(f'X: {X_test[idx]}')
                        print(f'y: {y_test[idx]}')
                        print(f'nonzero_index_X: {nonzero_index_X}')
                        print(f'nonzero_index_y: {nonzero_index_y}')
                        exit()
                    else:
                        # print(f'X: {nonzero_index_X}')
                        # print(f'y: {nonzero_index_y}')
                        if not np.equal(nonzero_index_X, nonzero_index_y).all():
                            #     if nonzero_index_X[each_i] != nonzero_index_y[each_i]:
                            print(f'TEST: Raise Error: elem does not match')
                            print(f'X: {nonzero_index_X}')
                            print(f'y: {nonzero_index_y}')
                            exit()

                print(f'{dataset}, No Mismatch for model = {model_name}, SEED = {seed}')
                # continue

                print(
                    f'shape of X_train = {X_train.shape}, shape of y_train = {y_train.shape}, shape of X_test = {X_test.shape}, shape of y_test = {y_test.shape}')
                # print(f'first sample = {X_train[0]}, first label = {y_train[0]}')
                if boosting_model == 'XGB':
                    test_mse, test_rsq, test_corr, mean_mse, mean_rsq, y_train_pred, y_test_pred, hp_dict, All_pred = train_xgboost(
                        X_train, y_train, X_test, y_test, all_groups, all_labels)
                if boosting_model == 'Ridge':
                    test_mse, test_rsq, test_corr, y_train_pred, y_test_pred, hp_dict, All_pred = train_multioutput_ridge_regression(
                        X_train, y_train, X_test, y_test, all_groups, all_labels)
                print(
                    f'Num-Sample: {num_training_sample}, test rsq = {test_rsq}, test mse = {test_mse}, test corr = {test_corr}')

                # exit()
                #
                Training_Samples_List.append(num_training_sample)
                DATASET_LIST.append(dataset)
                Test_rsq_list.append(test_rsq)
                Test_mse_list.append(test_mse)
                Test_corr_list.append(test_corr)

                if boosting_model == 'XGB':
                    Mean_mse_list.append(mean_mse)
                    Mean_rsq_list.append(mean_rsq)
                ALL_HP_List.append(copy.deepcopy(hp_dict))
                print(f'ALL_HP_List = {ALL_HP_List}')
                SEEDS_LIST.append(seed)
                Base_Model_List.append(model_name)

                '''convert to tensor'''
                y_train_pred = torch.from_numpy(y_train_pred)
                y_test_pred = torch.from_numpy(y_test_pred)
                All_pred = torch.from_numpy(All_pred)
                print(f'shapes: {y_train_pred.shape}, {y_test_pred.shape}, {All_pred.shape}')

                '''save the predictions'''
                if not os.path.exists(f'RESULTS/residual_prediction/{dataset}'):
                    os.makedirs(f'RESULTS/residual_prediction/{dataset}')
                final_output = torch.cat([y_train_pred, y_test_pred], 0)
                pred_file_name = f'RESULTS/residual_prediction/{dataset}/{ita_method}{boosting_model}_{dataset}_{model_name}_residuals_{num_training_sample}_seed_{seed}.pt'
                torch.save(final_output, pred_file_name)

                pred_file_name_final = f'RESULTS/residual_prediction/{dataset}/ALL_{ita_method}{boosting_model}_{dataset}_{model_name}_residuals_{num_training_sample}_seed_{seed}.pt'
                torch.save(All_pred, pred_file_name_final)
    print(len(Training_Samples_List),len(ALL_HP_List))
    res_df = pd.DataFrame({'Dataset':DATASET_LIST,
                           'SEED':SEEDS_LIST,
                           'Training_Samples':Training_Samples_List,
                           'Test_RSQ':Test_rsq_list,
                           'Test_MSE':Test_mse_list,
                           'Test_CORR':Test_corr_list,
                           'Base_Model':Base_Model_List,
                           f'Best_Hyperparameters_{boosting_model}':ALL_HP_List,
                           })

    res_df.to_csv(f'RESULTS/residual_prediction/{boosting_model}_{dataset}_Different_BaseModels_residuals_Prediction.csv')
