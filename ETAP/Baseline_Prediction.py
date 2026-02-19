import math
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import pickle
import ast
import copy
import itertools
import torch
import torch.nn as nn
import random
from sklearn.model_selection import KFold,cross_val_score,train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
import tqdm
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct,Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os


def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_ITA(X_Values,revised_integrals,TASKS):

    Only_groups = []
    for idx in range(0, len(X_Values)):
        active_tasks = X_Values[idx]  # Which tasks are active (1.0)?
        # Find the active tasks
        active_task_indices = torch.where(active_tasks == 1.0)[0]  # Indices of active tasks
        active_task_names = [TASKS[i] for i in active_task_indices]  # Get their names

        if len(active_task_names) == 1:  # If a single task is active
            task = active_task_names[0]
            combination_name = task
        else:  # For combinations of tasks
            combination_name = "|".join(active_task_names)  # Combine task names
        Only_groups.append(combination_name)

    # print(f'X_Values = {X_Values[10]}')
    # print(f'Only Groups = {Only_groups[:3]}')

    combinations = []
    # Generate all combinations
    for r in range(0,Num_Tasks+ 1):
        combinations.extend(itertools.combinations(TASKS, r))

    # Print the combinations
    # print(len(combinations))
    # print(combinations)

    rtn = {}
    true_label = {}
    for combi in combinations:
        if len(combi) == 0:
            continue
        task_grp = '|'.join(combi)
        if task_grp not in Only_groups:
            continue
        rtn[task_grp] = {task: 0. for task in combi}
        # print(f'combi = {combi}')
        for each_task in combi:
            for other_task in combi:
                if each_task != other_task:
                    rtn[task_grp][each_task] += revised_integrals[other_task][each_task]  # B->A

    for group in rtn:
        if '|' in group:
            for task in rtn[group]:
                rtn[group][task] /= (len(group.split('|')) - 1)
    # print(f'rtn = {len(rtn)}')
    # print(f'X_Values = {len(X_Values)}')
    X_final = []
    for each_x in X_Values:
        ITA_Val = copy.deepcopy(each_x)
        non_zero = [j for j, e in enumerate(each_x) if e > 0]
        task_names = [TASKS[idx] for idx in non_zero]
        # print(task_names)
        task_grp = '|'.join(task_names)
        ita_val_rtn = rtn[task_grp]
        for idx in non_zero:
            ITA_Val[idx] = ita_val_rtn[TASKS[idx]]
        # each_x = np.concatenate([each_x, ITA_Val], axis=0)
        ITA_Val = np.array(ITA_Val)
        # print(f'ITA_Val = {ITA_Val}')
        '''append to the X_final array'''
        X_final.append(ITA_Val)
    X_final = np.array(X_final)
    # print(f'X_final = {X_final[10]}')
    return rtn, X_final
    # return rtn



# Function to calculate R^2
def r_square(prediction_vals, targets):
    ss_residual = torch.sum((targets - prediction_vals) ** 2)
    ss_total = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()

def get_predictions_for_groups(X_values,tasks_map_best_GT,pred_data_best_GT):
    Only_groups = []
    for idx in range(0, len(X_values)):
        active_tasks = X_values[idx]  # Which tasks are active (1.0)?
        # Find the active tasks
        active_task_indices = torch.where(active_tasks == 1.0)[0]  # Indices of active tasks
        active_task_names = [TASKS[i] for i in active_task_indices]  # Get their names

        if len(active_task_names) == 1:  # If a single task is active
            task = active_task_names[0]
            combination_name = task
        else:  # For combinations of tasks
            combination_name = "|".join(active_task_names)  # Combine task names
        Only_groups.append(combination_name)

    # print(f'len(Only_groups) = {len(Only_groups)}')
    # print(f'shapes: {tasks_map_best_GT.shape,pred_data_best_GT.shape}')

    Predicted_Gains = {}
    for idx in range(0,len(tasks_map_best_GT)):
        active_tasks = tasks_map_best_GT[idx]  # Which tasks are active (1.0)?
        pred_values = pred_data_best_GT[idx]  # Predicted values
        #Separate the test set groups only for group-selection

        # Find the active tasks
        active_task_indices = torch.where(active_tasks == 1.0)[0]  # Indices of active tasks
        active_task_names = [TASKS[i] for i in active_task_indices]  # Get their names

        if len(active_task_names) == 1:  # If a single task is active
            task = active_task_names[0]
            if task in Only_groups:
                Predicted_Gains[task] = {task: 0.0}  # Single task, set gain to 0.0
        else:  # For combinations of tasks
            combination_name = "|".join(active_task_names)  # Combine task names
            task_contributions = {
                TASKS[i]: float(pred_values[i])  # predicted_gains
                for i in active_task_indices
            }
            if combination_name in Predicted_Gains.keys():
                print(f'already exists')
            if combination_name in Only_groups:
                Predicted_Gains[combination_name] = task_contributions

    X_predicted_MTGNet = []
    for each_x in X_values:
        predicted_val = copy.deepcopy(each_x)
        non_zero = [j for j, e in enumerate(each_x) if e > 0]
        task_names = [TASKS[idx] for idx in non_zero]
        task_grp = '|'.join(task_names)
        mtgnet_val_rtn = Predicted_Gains[task_grp]
        for idx in non_zero:
            predicted_val[idx] = mtgnet_val_rtn[TASKS[idx]]
        predicted_val = np.array(predicted_val)
        '''append to the X_final array'''
        X_predicted_MTGNet.append(predicted_val)
    X_predicted_MTGNet = np.array(X_predicted_MTGNet)

    return Predicted_Gains,X_predicted_MTGNet


def affine_transformation(flattened_y,flattened_y_ita):
    # Compute means
    mean_y = np.mean(flattened_y)
    mean_ITA = np.mean(flattened_y_ita)

    # Compute standard deviations
    std_y = np.std(flattened_y)
    std_ITA = np.std(flattened_y_ita)


    correlation = np.corrcoef(flattened_y_ita, flattened_y)[0, 1] # Compute correlation
    a = correlation * (std_y / std_ITA) # Compute slope
    b = mean_y - a * mean_ITA # Compute the intercept (b)
    transformed_y_ita = a * np.array(flattened_y_ita) + b
    return transformed_y_ita,a,b


def perform_cross_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=experiment)
    all_coefficients = []
    all_metrics = []

    for train_idx, val_idx in kf.split(X):
        # Split the data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Step 1: Add bias term and perform linear regression
        X_train_augmented = np.column_stack((X_train, np.ones(X_train.shape[0])))  # Add bias term
        coefficients = np.linalg.lstsq(X_train_augmented, y_train, rcond=None)[0]
        all_coefficients.append(coefficients)

        # Step 2: Validate on the validation set
        X_val_augmented = np.column_stack((X_val, np.ones(X_val.shape[0])))  # Add bias term
        predictions = np.dot(X_val_augmented, coefficients)

        # Step 3: Compute metrics
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r_squared = r2_score(y_val, predictions)
        correlation = np.corrcoef(y_val, predictions)[0, 1]

        all_metrics.append((rmse, r_squared, correlation))

    # Average coefficients and metrics
    avg_coefficients = np.mean(all_coefficients, axis=0)
    avg_rmse = np.mean([m[0] for m in all_metrics])
    avg_r_squared = np.mean([m[1] for m in all_metrics])
    avg_correlation = np.mean([m[2] for m in all_metrics])
    # print(f'all_coefficients = {all_coefficients}')
    return avg_coefficients, avg_rmse, avg_r_squared, avg_correlation
def evaluate_model_performance(train_data,test_data):
    X_combined_train, train_Labels = train_data
    X_combined_test, test_Labels = test_data

    coefficients, avg_rmse, avg_r_squared, avg_correlation = perform_cross_validation(X_combined_train, train_Labels,
                                                                                      n_splits=5)



    print(f"Average Weights: a1={coefficients[0]:.6f}, a2={coefficients[1]:.6f}, Bias={coefficients[2]:.6f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average R-squared: {avg_r_squared:.4f}")
    print(f"Average Correlation: {avg_correlation:.4f}")



    # Step 1: Perform linear regression to compute weights
    # X_train_augmented = np.column_stack((X_combined_train, np.ones(X_combined_train.shape[0])))  # Add bias term
    # coefficients = np.linalg.lstsq(X_train_augmented, train_Labels, rcond=None)[0]  # Solve for weights

    # Extract weights and bias
    a1, a2, b = coefficients
    print(f"{dataset}: Weights: a1={a1:.6f}, a2={a2:.6f}, Bias: b={b:.6f}")
    # exit()


    # Step 2: Apply transformation on test data
    X_test_augmented = np.column_stack((X_combined_test, np.ones(X_combined_test.shape[0])))  # Add bias term
    predictions = np.dot(X_test_augmented, coefficients)  # Predicted labels
    print(f'shape of predictions = {predictions.shape}')

    # Step 3: Compute metrics
    rmse = np.sqrt(mean_squared_error(test_Labels, predictions))
    correlation = np.corrcoef(test_Labels, predictions)[0, 1]
    r_squared = r2_score(test_Labels, predictions)

    print(f"RMSE: {rmse:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"R-squared: {r_squared:.4f}")



    total_variance = np.var(predictions)
    print(f'total_variance = {total_variance:.6f}', end = ' ')
    transformed_feature_1 = X_combined_test[:, 0] * a1
    transformed_feature_2 = X_combined_test[:, 1] * a2
    # print(transformed_feature_1)
    var_feature1 = torch.var(transformed_feature_1) / total_variance
    var_feature2 = torch.var(transformed_feature_2) / total_variance
    print(f"Variance Contribution - Feature 1: {var_feature1:.6f}, Feature 2: {var_feature2:.6f}")

    predicted_labels = []
    idx_count = 0
    predictions = torch.tensor(predictions)
    # print(f'predictions = {predictions[:10]}')
    for i in range(0,len(testx)):
        predicted_vals = copy.deepcopy(testx[i])
        # print(f'testx = {testx[i]}')
        # print(f'predicted_vals = {predicted_vals} {predicted_vals.shape}')
        for j in range(0,len(testx[i])):
            if testx[i][j]>0:
                predicted_vals[j] = predictions[idx_count]
                idx_count+=1
        predicted_labels.append(predicted_vals)
    # Check if all tensors in predicted_labels have the same shape
    shapes = [label.shape for label in predicted_labels]
    if all(shape == shapes[0] for shape in shapes):  # Ensure consistent shapes
        predicted_labels = torch.stack(predicted_labels)
    else:
        raise ValueError("Tensors in `predicted_labels` have inconsistent shapes.")

    print(len(predicted_labels))
    print(f'shape of testy = {testy.shape}, shape of predicted_labels = {predicted_labels.shape}')

    print(f'RSq = {r2_score(testy[testx != 0],predicted_labels[testx != 0])}')
    print(f'Corr = {np.corrcoef(testy[testx != 0], predicted_labels[testx != 0])[0][1]}')
    # exit()

    # residuals = test_Labels - predictions
    # residual_ax[Datasets.index(dataset)].hist(residuals, bins=30)
    # residual_ax[Datasets.index(dataset)].set_title(f"{dataset.upper()}")
    # residual_ax[Datasets.index(dataset)].grid()
    # residual_ax[Datasets.index(dataset)].set_ylabel('Count')

    return r_squared, correlation, rmse

def find_optimal_depth_for_DT(orig_X_train,orig_y_train, transformed_X_train):
    RMSE = {'Train': [], 'Test': []}
    RSQ = {'Train': [], 'Test': []}
    CORR = {'Train': [], 'Test': []}
    best_depth = None
    best_rsq_val = -math.inf
    All_Val_Score = []
    DEPTHS = [i for i in range(1, 10)]
    n_iterations = 100
    learning_rate = 0.01
    for max_depth in DEPTHS:
        rmse_train = []
        rmse_val = []
        r2_train = []
        r2_val = []
        corr_train = []
        corr_val = []
        before_val_rsq = []
        kf = KFold(n_splits=5, shuffle=True, random_state=experiment)  # 5-fold cross-validation
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(orig_X_train)):
            X_train, X_val = transformed_X_train[train_idx], transformed_X_train[val_idx]
            y_train, y_val = orig_y_train[train_idx], orig_y_train[val_idx]
            init_X_train,init_X_val = orig_X_train[train_idx], orig_X_train[val_idx]
            # print(f'shapes of train and validations = {X_train.shape, y_train.shape}, {X_val.shape, y_val.shape}')

            before_val_rsq.append(r2_score(y_val, init_X_val))

            # Initialize predictions as the base predictions (e.g., ITA scores)
            current_predictions_train = init_X_train
            current_predictions_test = init_X_val

            # Iterative boosting
            # print(f'shapes initially: {current_predictions_train.shape}')

            for i in range(n_iterations):
                # Compute residuals
                residuals_train = y_train - current_predictions_train
                # print(f'shape of residual: {residuals_train.shape}')

                # Fit a weak model to the residuals
                boosted_model = DecisionTreeRegressor(max_depth=max_depth)
                boosted_model.fit(X_train, residuals_train)

                # Predict residuals and update predictions
                residual_preds_train = boosted_model.predict(X_train)
                residual_preds_test = boosted_model.predict(X_val)

                current_predictions_train += learning_rate * residual_preds_train
                current_predictions_test += learning_rate * residual_preds_test

            # Evaluate the final predictions
            rmse_tr = np.sqrt(mean_squared_error(y_train, current_predictions_train))
            rmse_vl = np.sqrt(mean_squared_error(y_val, current_predictions_test))
            r2_vl = r2_score(y_val, current_predictions_test)
            r2_tr = r2_score(y_train, current_predictions_train)
            correlation_tr = np.corrcoef(y_train, current_predictions_train)[0, 1]
            correlation_vl = np.corrcoef(y_val, current_predictions_test)[0, 1]

            rmse_train.append(rmse_tr)
            rmse_val.append(rmse_vl)
            r2_train.append(r2_tr)
            r2_val.append(r2_vl)
            corr_train.append(correlation_tr)
            corr_val.append(correlation_vl)

        # print(f"RMSE Train: {np.mean(rmse_train):.6f}, RMSE Test: {np.mean(rmse_val):.6f}")
        # print(f"R-squared Train: {np.mean(r2_train):.6f}, R-squared Test: {np.mean(r2_val):.6f}")
        # print(f'correlation_train = {np.mean(corr_train):.6f}, correlation Test = {np.mean(corr_val):.6f}')

        RMSE['Train'].append(np.mean(rmse_train))
        RMSE['Test'].append(np.mean(rmse_val))
        RSQ['Train'].append(np.mean(r2_train))
        RSQ['Test'].append(np.mean(r2_val))
        CORR['Train'].append(np.mean(corr_train))
        CORR['Test'].append(np.mean(corr_val))
        r2_val_score = np.mean(r2_val)
        if r2_val_score > best_rsq_val:
            best_rsq_val = r2_val_score
            best_depth = max_depth
        # print(f'before val rsq = {np.mean(before_val_rsq)}')
    print(f'Best Depth = {best_depth}, best_rsq = {best_rsq_val}, before Val Score = {np.mean(before_val_rsq)}')
    return best_depth


def evaluate_Random_forest_performance(train_data,test_data, all_data):

    # ITA predictions (initial weak predictions) and true labels
    X_train,y_train = train_data
    X_test,y_test = test_data
    all_x, all_y = all_data

    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    all_x = all_x.reshape(-1,1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    all_x_scaled = scaler.transform(all_x)

    # Define a baseline RandomForestRegressor
    rf = RandomForestRegressor(random_state=experiment)

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [c for c in range(10, 210, 20)],
        'max_depth': [c for c in range(1, 11)],
        'min_samples_split': [c for c in range(2, 10, 2)],
        'min_samples_leaf': [c for c in range(1, 10, 2)],
    }
    print(f'param_grid = {param_grid}')

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,  # Parallelize across all available cores
        verbose=1  # Increase to see detailed progress
    )

    # Fit the grid search on the training set
    grid_search.fit(X_train_scaled, y_train)

    # Best estimator from the grid search
    best_rf = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # Predict on the test set
    y_test_pred = best_rf.predict(X_test_scaled)
    y_train_pred = best_rf.predict(X_train_scaled)
    all_x_pred = best_rf.predict(all_x_scaled)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mse)
    test_r2 = r2_score(y_test, y_test_pred)
    test_corr = np.corrcoef(y_test, y_test_pred)[0, 1]

    print(f"Test MSE: {mse:.5f}")
    print(f"Test RMSE: {test_rmse:.5f}")
    print(f"Test R^2: {test_r2:.5f}")
    print(f"Test Correlation: {test_corr:.5f}")

    # plot_type = 'X_test'
    # if plot_type == 'y_test':
    #     scatter_ax[Datasets.index(dataset)].scatter(y_test, y_test_pred, alpha=0.7, color="purple",
    #                                                 label='True values vs KNN prediction')
    #     # scatter_ax[Datasets.index(dataset)].scatter(y_test,X_test, label='True values vs ITA Scores')
    #     scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',
    #                                              label="Ideal (min-max)")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'True Label')
    #     scatter_ax[Datasets.index(dataset)].legend()
    # else:
    #     scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test, color="purple", alpha=0.6, label="True Test Data")
    #     scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test_pred, color="blue", alpha=0.7,
    #                                                 label='Predicted values')
    #     # scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',label="Ideal")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'ITA Scores')
    #
    # scatter_ax[Datasets.index(dataset)].set_ylabel(f'True Label and Prediction from RandomForest')
    # scatter_ax[Datasets.index(dataset)].set_title(f'{dataset.upper()} - RandomForest Regression Results')
    # scatter_ax[Datasets.index(dataset)].legend()

    predicted_labels_train, predicted_labels,all_predictions  = get_predicted_gains(y_train_pred, y_test_pred, all_x_pred)

    return test_r2, test_corr, test_rmse, predicted_labels_train, predicted_labels,all_predictions


#KNN
def evaluate_knn_performance(train_data, test_data, all_data):
    # Unpack train and test data
    X_train, y_train = train_data
    X_test, y_test = test_data
    ALL_X, ALL_y = all_data

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    ALL_X = ALL_X.reshape(-1, 1)

    # Scale the feature (often helpful for GPR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ALL_X_scaled = scaler.transform(ALL_X)

    X_train, X_test = X_train_scaled, X_test_scaled
    ALL_X = ALL_X_scaled



    #get optimal neighbours
    orig_X_train, orig_y_train = X_train, y_train
    print(f'orig_X_train = {orig_X_train.shape} for dataset: {dataset}')

    RMSE = {'Train': [], 'Test': []}
    RSQ = {'Train': [], 'Test': []}
    CORR = {'Train': [], 'Test': []}
    best_neighbor = None
    best_rsq_val = -math.inf
    max_k = int(np.sqrt(orig_y_train.shape[0]))
    print(f'max_k by rule of thumb = {max_k} for dataset : {dataset}')
    Neighbors = [i for i in range(5, max_k+20)]

    # Example: Cross-validation to find optimal k
    def find_optimal_k(X_train, y_train, max_k=max_k):
        cv_errors = []
        max_neighbour = min(max_k + 20, X_train.shape[0])
        for k in range(1, max_neighbour):
            knn = KNeighborsRegressor(n_neighbors=k)
            mse = -np.mean(cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
            cv_errors.append(mse)

        optimal_k = np.argmin(cv_errors) + 1
        return optimal_k, cv_errors

    # Usage
    optimal_k, cv_errors = find_optimal_k(X_train, y_train)
    print(f"Optimal k: {optimal_k} for dataset : {dataset}, best mse: {np.min(cv_errors):.5f}")
    best_neighbor = optimal_k
    # for n_neighbors in Neighbors:
    #     rmse_train = []
    #     rmse_val = []
    #     r2_train = []
    #     r2_val = []
    #     corr_train = []
    #     corr_val = []
    #     before_val_rsq = []
    #     mse_val = []
    #     kf = KFold(n_splits=5, shuffle=True, random_state=experiment)  # 5-fold cross-validation
    #     # Cross-validation loop
    #     for fold, (train_idx, val_idx) in enumerate(kf.split(orig_X_train)):
    #         X_train, X_val = orig_X_train[train_idx], orig_X_train[val_idx]
    #         y_train, y_val = orig_y_train[train_idx], orig_y_train[val_idx]
    #
    #         before_val_rsq.append(r2_score(y_val,X_val))
    #         # Initialize KNN Regressor
    #         knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    #
    #         # Train the KNN regressor
    #         knn.fit(X_train.reshape(-1, 1), y_train)
    #         # Predict on training and testing data
    #         y_train_pred = knn.predict(X_train.reshape(-1, 1))
    #         y_val_pred = knn.predict(X_val.reshape(-1, 1))
    #
    #         # Compute metrics
    #         rmse_tr = np.sqrt(mean_squared_error(y_train, y_train_pred))
    #         rmse_vl = np.sqrt(mean_squared_error(y_val, y_val_pred))
    #         r2_tr = r2_score(y_train, y_train_pred)
    #         r2_vl = r2_score(y_val, y_val_pred)
    #         correlation_tr = np.corrcoef(y_train, y_train_pred)[0][1]
    #         correlation_vl = np.corrcoef(y_val, y_val_pred)[0][1]
    #         mse_val.append(mean_squared_error(y_val, y_val_pred))
    #
    #         rmse_train.append(rmse_tr)
    #         rmse_val.append(rmse_vl)
    #         r2_train.append(r2_tr)
    #         r2_val.append(r2_vl)
    #         corr_train.append(correlation_tr)
    #         corr_val.append(correlation_vl)
    #
    #         # print(f"RMSE Train: {np.mean(rmse_train):.6f}, RMSE Test: {np.mean(rmse_val):.6f}")
    #         # print(f"R-squared Train: {np.mean(r2_train):.6f}, R-squared Test: {np.mean(r2_val):.6f}")
    #         # print(f'correlation_train = {np.mean(corr_train):.6f}, correlation Test = {np.mean(corr_val):.6f}')
    #
    #     RMSE['Train'].append(np.mean(rmse_train))
    #     RMSE['Test'].append(np.mean(rmse_val))
    #     RSQ['Train'].append(np.mean(r2_train))
    #     RSQ['Test'].append(np.mean(r2_val))
    #     CORR['Train'].append(np.mean(corr_train))
    #     CORR['Test'].append(np.mean(corr_val))
    #
    #     r2_val_score = np.mean(r2_val)
    #     if r2_val_score > best_rsq_val:
    #         best_rsq_val = r2_val_score
    #         best_neighbor = n_neighbors
    #         best_mse_val = np.mean(mse_val)
    # print(f'Best Neighbor = {best_neighbor}, best_rsq = {best_rsq_val}, before Val Score = {np.mean(before_val_rsq)}, mse = {best_mse_val:.5f}')
    #
    # row = Datasets.index(dataset)
    # for row, subfig in enumerate(subfigs):
    #     if row !=Datasets.index(dataset):
    #         continue
    #     subfig.suptitle(f'{dataset.upper()} --, Validation Rsq (before): {np.mean(before_val_rsq):0.4f} ---- Validation Rsq (after) = {best_rsq_val:.4f}, '
    #                     f'neighbour = {best_neighbor}')
    #     ax2 = subfig.subplots(nrows=1, ncols=3)
    #     for col, ax in enumerate(ax2):
    #         if col ==0:
    #             ax.plot(Neighbors,RSQ['Train'], label='Train')
    #             ax.plot(Neighbors,RSQ['Test'], label='val')
    #             ax.set_ylabel('RSQ')
    #         if col == 1:
    #             ax.plot(Neighbors,CORR['Train'], label='Train')
    #             ax.plot(Neighbors,CORR['Test'], label='val')
    #             ax.set_ylabel('Correlation')
    #         if col == 2:
    #             ax.plot(Neighbors, RMSE['Train'], label='Train')
    #             ax.plot(Neighbors, RMSE['Test'], label='val')
    #             ax.set_ylabel('RMSE')
    #         ax.grid()
    #         ax.legend()
    #         # ax.set_xticks(Neighbors)
    #         # ax.set_xlim(0,Neighbors[-1])
    #         if row == 2:
    #             ax.set_xlabel(f'Best n_neighbour')

    X_train, y_train = orig_X_train, orig_y_train

    # Initialize KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=best_neighbor)
    # Train the KNN regressor
    knn.fit(X_train, y_train)
    # Predict on training and testing data
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    all_x_pred = knn.predict(ALL_X)

    # Compute metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    corr_train = np.corrcoef(y_train, y_train_pred)[0][1]
    corr_test = np.corrcoef(y_test, y_test_pred)[0][1]

    print(f"KNN Performance with k={best_neighbor}")
    print(f"RMSE Train: {rmse_train:.6f}, RMSE Test: {rmse_test:.6f}")
    print(f"R-squared Train: {r2_train:.6f}, R-squared Test: {r2_test:.6f}")

    # Visualization
    # # Original data
    # fig_idx = Datasets.index(dataset)
    # visualization_ax[fig_idx].scatter(X_train, y_train, label="Train Data", alpha=0.7, color="blue")
    # visualization_ax[fig_idx].scatter(X_test, y_test, label="Test Data", alpha=0.7, color="green")
    #
    # # # KNN predictions (smooth curve)
    # X_full = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)
    # y_full_pred = knn.predict(X_full)
    # visualization_ax[fig_idx].plot(X_full, y_full_pred, label="KNN Regression", color="red", linewidth=2)
    #
    # visualization_ax[fig_idx].set_xlabel("Input Feature")
    # visualization_ax[fig_idx].set_ylabel("Target")
    # visualization_ax[fig_idx].set_title(f"{dataset}")
    # visualization_ax[fig_idx].legend()
    # visualization_ax[fig_idx].grid()


    # # Residual analysis
    # residuals = []
    # for k in [best_neighbor]:
    #     knn = KNeighborsRegressor(n_neighbors=k)
    #     knn.fit(X_train.reshape(-1, 1), y_train)
    #     y_test_pred = knn.predict(X_test.reshape(-1, 1))
    #     residuals.append(y_test - y_test_pred)
    #
    # for i, res in enumerate(residuals):
    #     residual_ax[fig_idx].scatter(X_test, res, label=f"Residuals (k={best_neighbor})", alpha=0.7)
    #
    # residual_ax[fig_idx].axhline(0, color="black", linestyle="--", linewidth=1)
    # residual_ax[fig_idx].set_title(f"Residual Analysis: {dataset}")
    # residual_ax[fig_idx].set_xlabel("Input Feature")
    # residual_ax[fig_idx].set_ylabel("Residuals")
    # residual_ax[fig_idx].legend()
    # if dataset == 'ETTm1':
    #     fig_visualization.suptitle(f"KNN Regression Visualization")
    #     plt.show()
        # fig_residual.suptitle(f"KNN Residuals")
        # plt.show()

    #
    # # Scatter plot: True vs Predicted
    # plt.figure(figsize=(6, 6))
    # plt.scatter(y_test, y_test_pred, alpha=0.7, color="purple", label="Test Predictions")
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal")
    # plt.xlabel("True Values")
    # plt.ylabel("Predicted Values")
    # plt.title("True vs Predicted")
    # plt.legend()
    # plt.show()
    #
    # plot_type = 'X_test'
    # if plot_type == 'y_test':
    #     scatter_ax[Datasets.index(dataset)].scatter(y_test,y_test_pred, alpha=0.7, color="purple", label = 'True values vs KNN prediction')
    #     # scatter_ax[Datasets.index(dataset)].scatter(y_test,X_test, label='True values vs ITA Scores')
    #     scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal (min-max)")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'True Label')
    #     scatter_ax[Datasets.index(dataset)].legend()
    # else:
    #     # scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test_pred,label='ITA scores vs Predicted values')
    #     # scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',label="Ideal")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'ITA Scores')
    #     sort_idx = np.argsort(X_test.ravel())  # Flatten if X_test is shape (N,1)
    #     X_test_sorted = X_test[sort_idx]
    #     y_test_sorted = y_test[sort_idx]
    #     y_test_pred_sorted = y_test_pred[sort_idx]
    #     # 2. Plot
    #     scatter_ax[Datasets.index(dataset)].scatter(X_test_sorted, y_test_sorted, color="purple", alpha=0.6, label="True Test Data")
    #     scatter_ax[Datasets.index(dataset)].plot(X_test_sorted, y_test_pred_sorted, color="red", linewidth=2,
    #                                              label='Predicted values')
    #
    # scatter_ax[Datasets.index(dataset)].set_ylabel(f'Prediction from KNN')
    # scatter_ax[Datasets.index(dataset)].set_title(f'{dataset.upper()} - KNN Regression Results')
    #

    predicted_labels_train, predicted_labels,all_predictions = get_predicted_gains(y_train_pred, y_test_pred, all_x_pred)

    return r2_test, corr_test, rmse_test, predicted_labels_train, predicted_labels,all_predictions


def get_predicted_gains(y_train_pred, y_test_pred, all_pred,):
    """
    Reconstructs the full predicted matrices for train and test sets by inserting
    the predicted values into the appropriate positions (based on mask > 0).

    Args:
        y_train_pred (list or array): Flattened predicted values for training set.
        y_test_pred (list or array): Flattened predicted values for test set.
        all_pred (ignored): Kept for compatibility but not used.
        testx (list of tensors): Binary mask indicating where predictions are needed (test).
        all_train_mask (list of tensors): Binary mask indicating where predictions are needed (train).

    Returns:
        predicted_labels_train (Tensor): Reconstructed training predictions.
        predicted_labels_test (Tensor): Reconstructed test predictions.
    """
    predictions_test = torch.tensor(y_test_pred)
    predictions_train = torch.tensor(y_train_pred)
    predictions_all = torch.tensor(all_pred)

    predicted_labels_test = []
    predicted_labels_train = []
    predicted_labels_all = []

    idx_test = 0
    for mask_row in testx:
        row = mask_row.clone()
        for j in range(len(row)):
            if row[j] > 0:
                row[j] = predictions_test[idx_test]
                idx_test += 1
        predicted_labels_test.append(row)

    idx_train = 0
    for mask_row in all_train_mask:
        row = mask_row.clone()
        for j in range(len(row)):
            if row[j] > 0:
                row[j] = predictions_train[idx_train]
                idx_train += 1
        predicted_labels_train.append(row)

    idx_all = 0
    for mask_row in all_groups:
        row = mask_row.clone()
        for j in range(len(row)):
            if row[j] > 0:
                row[j] = predictions_all[idx_all]
                idx_all += 1
        predicted_labels_all.append(row)

    # Check and stack
    try:
        predicted_labels_test = torch.stack(predicted_labels_test)
        predicted_labels_train = torch.stack(predicted_labels_train)
        predicted_labels_all = torch.stack(predicted_labels_all)
    except RuntimeError as e:
        raise ValueError("Inconsistent shapes in predicted labels.") from e

    return predicted_labels_train, predicted_labels_test, predicted_labels_all

# def get_predicted_gains(y_train_pred, y_test_pred, all_pred):
#     final_pred_train = y_train_pred
#     final_pred_test = y_test_pred
#     final_pred = all_pred
#     predicted_labels = []
#     idx_count = 0
#     predictions = torch.tensor(final_pred_test)
#     for i in range(0, len(testx)):
#         predicted_vals = copy.deepcopy(testx[i])
#         # print(f'testx = {testx[i]}')
#         # print(f'predicted_vals = {predicted_vals} {predicted_vals.shape}')
#         for j in range(0, len(testx[i])):
#             if testx[i][j] > 0:
#                 predicted_vals[j] = predictions[idx_count]
#                 idx_count += 1
#         predicted_labels.append(predicted_vals)
#     predicted_labels_train = []
#     idx_count = 0
#     predictions_train = torch.tensor(final_pred_train)
#     # print(f'predictions = {predictions[:10]}')
#     for i in range(0, len(all_train_mask)):
#         predicted_vals = copy.deepcopy(all_train_mask[i])
#         # print(f'testx = {testx[i]}')
#         # print(f'predicted_vals = {predicted_vals} {predicted_vals.shape}')
#         for j in range(0, len(all_train_mask[i])):
#             if all_train_mask[i][j] > 0:
#                 predicted_vals[j] = predictions_train[idx_count]
#                 idx_count += 1
#         predicted_labels_train.append(predicted_vals)
#
#     # Check if all tensors in predicted_labels have the same shape
#     shapes = [label.shape for label in predicted_labels]
#     if all(shape == shapes[0] for shape in shapes):  # Ensure consistent shapes
#         predicted_labels = torch.stack(predicted_labels)
#     else:
#         raise ValueError("Tensors in `predicted_labels` have inconsistent shapes.")
#
#     shapes = [label.shape for label in predicted_labels_train]
#     if all(shape == shapes[0] for shape in shapes):  # Ensure consistent shapes
#         predicted_labels_train = torch.stack(predicted_labels_train)
#     else:
#         raise ValueError("Tensors in `predicted_labels_train` have inconsistent shapes.")
#
#     return predicted_labels_train, predicted_labels

def evaluate_BSpline_LR(train_data, test_data, all_data, bias_val):

    # Unpack train and test data
    X_train, y_train = train_data
    X_test, y_test = test_data
    ALL_X, ALL_y = all_data
    print(f'train and test shapes: {X_train.shape} {y_train.shape} {X_test.shape} {y_test.shape}')
    # pipeline for Spline transformation + Linear Regression
    pipeline = Pipeline([
        ('spline', SplineTransformer(include_bias=bias_val)),
        ('linear', LinearRegression(fit_intercept=True))])
    # print(pipeline)
    # if bias_val:
    #     pipeline = Pipeline([
    #         ('spline', SplineTransformer(include_bias=bias_val)),
    #         ('linear', LinearRegression(fit_intercept=False))])
    # else:
    #     pipeline = Pipeline([
    #         ('spline', SplineTransformer(include_bias=bias_val)),
    #         ('linear', LinearRegression(fit_intercept=True))])
    # hyperparameter grid
    max_knot = int(math.sqrt(X_train.shape[0]))
    print(f'max_knot = {max_knot}')
    param_grid = {
        'spline__degree': [i for i in range(2, 7)],
        'spline__n_knots': [c for c in range(2,max_knot+5,1)]
        # 'spline__n_knots': [2, 3, 4, 5, 6] + [c for c in range(10, max_knot + 10, 5)]
    }
    print(f'param_grid = {param_grid}')

    if dataset!='ETTm1':
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error')
    else:
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train[:, np.newaxis], y_train)

    print(f"{dataset}, Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", -grid_search.best_score_, )
    print(f'train variance: {torch.var(y_train)}, test variance: {torch.var(y_test)}')

    best_pipeline = grid_search.best_estimator_

    best_pipeline.fit(X_train[:, np.newaxis], y_train)
    coefs = best_pipeline.named_steps['linear'].coef_
    print(f'coefs = {coefs}')
    print("intercept_:", best_pipeline.named_steps['linear'].intercept_)

    # E.g. to see the shape after the best_pipeline transforms the data
    X_transformed = best_pipeline[:-1].transform(X_test[:,np.newaxis])  # all steps except the final 'linear'
    # print("Shape after best_pipeline transforms:", X_transformed.shape)

    spline_transformer = SplineTransformer(n_knots=grid_search.best_params_['spline__n_knots'],
                                           degree=grid_search.best_params_['spline__degree'],
                                           include_bias=bias_val)
    # X_spline = spline_transformer.fit_transform(X_train[:, np.newaxis],y_train)
    # X_spline_test = spline_transformer.transform(X_test[:, np.newaxis])
    # print(f"X_spline_test = {X_spline_test.shape}")
    #
    # exit()
    # print(f'bias = {pipeline.named_steps["linear"].bias_}')

    # Combinations = [f'{param["spline__degree"]}_{param["spline__n_knots"]}' for param in grid_search.cv_results_['params']]
    # for each_combination, each_score in zip(Combinations, grid_search.cv_results_['mean_test_score']):
    #     n_knots = int(each_combination.split('_')[1])
    #     degree = int(each_combination.split('_')[0])
    #     spline_transformer = SplineTransformer(n_knots=n_knots, degree=degree,
    #                                            include_bias=False)
    #     X_spline = spline_transformer.fit_transform(X_train[:, np.newaxis])
    #     basis_sums = X_spline.sum(axis=1)  # Sum across columns (basis functions)
    #     print(f'Combination (degree,n_knots) = {(degree,n_knots)}, cross-val score = {-each_score:0.6f},\t basis sum = {np.mean(basis_sums):.6f}')

    # # plot combination vs cross-validation score
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.plot(grid_search.cv_results_['mean_test_score'], label="CV MSE")
    # ax.set_xlabel("Hyperparameter Combination")
    # ax.set_ylabel("Mean Squared Error")
    #
    # ax.legend()
    # plt.show()



    # for n_knots in param_grid['spline__n_knots']:

    # if dataset=='chemical':
    #     # # Plot spline basis for different n_knots
    #     if len(param_grid['spline__n_knots'])%2==0:
    #         fig_basis_chemical, basis_ax_chemical = plt.subplots(2, len(param_grid['spline__n_knots'])//2, figsize=(10, 6))
    #     else:
    #         fig_basis_chemical, basis_ax_chemical = plt.subplots(2, len(param_grid['spline__n_knots'])//2+1, figsize=(10, 6))
    #
    #
    #     for n_knots in param_grid['spline__n_knots']:
    #         test = SplineTransformer(n_knots=n_knots, degree=grid_search.best_params_['spline__degree'], include_bias=False)
    #         X_spline = test.fit_transform(X_train[:, np.newaxis])
    #         basis_sums = X_spline.sum(axis=1)  # Sum across columns (basis functions)
    #
    #         col_idx = param_grid['spline__n_knots'].index(n_knots)
    #         if col_idx < len(param_grid['spline__n_knots']) // 2:
    #             row = 0
    #         else:
    #             row = 1
    #             col_idx = len(param_grid['spline__n_knots']) - (col_idx + 1)
    #
    #         basis_ax_chemical[row, col_idx].plot(X_train, X_spline)
    #         basis_ax_chemical[row, col_idx].plot(X_train, basis_sums, color='black', linewidth=2, label = 'Sum')
    #         basis_ax_chemical[row, col_idx].set_title(#f"degree: {grid_search.best_params_['spline__degree']}, n_knots={n_knots}, \n"
    #                                                   f"basis sum = {np.mean(basis_sums):.6f}")
    #         # basis_ax_chemical[row, col_idx].set_xlabel(f"{dataset} -- Input Feature")
    #         basis_ax_chemical[row, col_idx].set_ylabel(f"Basis -(degree,n_knots) = {(grid_search.best_params_['spline__degree'],n_knots)}")
    #         basis_ax_chemical[row, col_idx].legend()
    #     fig_basis_chemical.suptitle(f'{dataset.upper()} -- B-Spline Basis Functions')
    #     fig_basis_chemical.show()

    # n_knots = grid_search.best_params_['spline__n_knots']
    spline_transformer = SplineTransformer(n_knots=grid_search.best_params_['spline__n_knots'],
                                           degree=grid_search.best_params_['spline__degree'],
                                           include_bias=bias_val)
    X_spline = spline_transformer.fit_transform(X_train[:, np.newaxis])
    X_spline_test = spline_transformer.transform(X_test[:, np.newaxis])

    # X_plot = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)
    # X_spline_plot = spline_transformer.transform(X_plot)
    # basis_sums_plot = X_spline_plot.sum(axis=1)
    # fig_idx = Datasets.index(dataset)
    # basis_ax[fig_idx].plot(X_plot, X_spline_plot, alpha=0.7)
    # basis_ax[fig_idx].plot(X_plot, basis_sums_plot, color='black', linewidth=2, label="Sum of Basis")
    # basis_ax[fig_idx].set_title(
    #     f"degree: {grid_search.best_params_['spline__degree']}, n_knots={grid_search.best_params_['spline__n_knots']}, basis sum = {np.mean(basis_sums_plot):.6f}")
    # basis_ax[fig_idx].set_xlabel(f"{dataset} -- Input Feature")
    # if bias_val:
    #     basis_ax[fig_idx].set_ylabel(f"Basis Function Value with Bias")
    # else:
    #     basis_ax[fig_idx].set_ylabel(f"Basis Function Value without Bias")
    # basis_ax[fig_idx].legend()


    # print(f'row = {row}, fig_idx = {fig_idx}')
    # basis_ax[fig_idx].plot(X_train, X_spline)
    # basis_ax[fig_idx].plot(X_train, basis_sums, color='black', linewidth=2, label = "Sum of Basis")



    # Evaluate on test data
    y_train_pred = grid_search.best_estimator_.predict(X_train[:, np.newaxis])
    y_test_pred = grid_search.best_estimator_.predict(X_test[:, np.newaxis])
    all_pred = grid_search.best_estimator_.predict(ALL_X[:, np.newaxis])
    test_mse = mean_squared_error(y_test, y_test_pred)
    print("Test MSE:", test_mse)

    # Calculate Residual Sum of Squares (SS_res) and Total Sum of Squares (SS_tot)
    ss_res = ((y_test - y_test_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    # Compute R-squared
    test_r_squared = 1 - (ss_res / ss_tot)
    print("Test R-squared different:", test_r_squared.item())

    # Evaluate the model
    test_corr = np.corrcoef(y_test, y_test_pred)[0][1]
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_rsq = r2_score(y_test, y_test_pred)

    print(f'shape of X_spline: {X_spline_test.shape}')
    import matplotlib.pyplot as plt
    for jdx in range(0, len(X_test)):
        if abs(y_test_pred[jdx] - y_test[jdx])>1:
            print(f'jdx {jdx}: ITA: {X_test[jdx]}, y_test: {y_test[jdx]}, y_test_pred: {y_test_pred[jdx]}')
            print(f'X_spline_test: {X_spline_test[jdx]}')

            # plt.plot(X_spline_test[jdx], color='black', linewidth=2)
        # else:
        #     if jdx%30==0:
        #         print(f'Other points: jdx {jdx}: ITA: {X_test[jdx]}, y_test: {y_test[jdx]}, y_test_pred: {y_test_pred[jdx]}')
        #         print(f'X_spline_test: {X_spline_test[jdx]}')
    # plt.show()
    # # Optional: Visualize the basis functions and their sum
    X_plot = np.linspace(X_test.min(), X_test.max(), 500).reshape(-1, 1)
    X_spline_plot = spline_transformer.transform(X_plot)
    basis_sums_plot = X_spline_plot.sum(axis=1)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(X_plot, X_spline_plot, label="Basis Functions")
    # plt.plot(X_plot, basis_sums_plot, color='black', linewidth=2, label="Sum of Basis (Should be 1)")
    # plt.xlabel('X-test')
    # plt.ylabel('spline_transformer.transform(X_test)')
    # # plt.legend()
    # plt.title("B-Spline Basis Functions and Their Sum")
    # plt.show()



    '''index sort X_test'''

    # train_test_fig,axes = plt.subplots(1,2)
    #
    # sort_idx = np.argsort(X_train)
    # X_train = X_train[sort_idx]
    # y_train_pred = y_train_pred[sort_idx]
    # y_train = y_train[sort_idx]
    # axes[0].scatter(X_train, y_train, color="blue", label="True Test Data (sorted)")
    # axes[0].scatter(X_train, y_train_pred, color="red", label="Predictions (sorted)")
    # axes[0].grid()
    # axes[0].set_title(f'train data')
    # axes[0].set_xlabel(f'train features: Spline degree = {grid_search.best_params_["spline__degree"]}, n_knots = {grid_search.best_params_["spline__n_knots"]}')
    # axes[0].set_ylabel('train labels and predictions')
    # axes[0].legend()
    #
    # sort_idx = np.argsort(X_test)
    # X_test = X_test[sort_idx]
    # y_test_pred = y_test_pred[sort_idx]
    # y_test = y_test[sort_idx]
    # axes[1].scatter(X_test, y_test, color="blue", label="True Test Data (sorted)")
    # axes[1].scatter(X_test, y_test_pred, color="red", label="Predictions (sorted)")
    # axes[1].grid()
    # axes[1].set_title(f'RMSE: {test_rmse:.5f}, test rsq: {test_rsq:.5f}, correlation = {test_corr:.4f}')
    # axes[1].set_xlabel(f'test features: Spline degree = {grid_search.best_params_["spline__degree"]}, n_knots = {grid_search.best_params_["spline__n_knots"]}')
    # axes[1].set_ylabel('test labels and predictions')
    # axes[1].legend()
    # plt.show()

    # plot_type = 'X_test'
    # if plot_type == 'y_test':
    #     scatter_ax[Datasets.index(dataset)].scatter(y_test, y_test_pred, alpha=0.7, color="purple",
    #                                                 label='True values vs KNN prediction')
    #     # scatter_ax[Datasets.index(dataset)].scatter(y_test,X_test, label='True values vs ITA Scores')
    #     scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',
    #                                              label="Ideal (min-max)")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'True Label')
    #     scatter_ax[Datasets.index(dataset)].legend()
    # else:
    #     # scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test, color="purple", alpha = 0.6, label="True Test Data")
    #     # scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test_pred, color="blue", alpha = 0.7, label='Predicted values')
    #     #plot a line with the predictions
    #     # 1. Sort by the values of X_test so the line plot is continuous
    #
    #     sort_idx = np.argsort(X_test.ravel())  # Flatten if X_test is shape (N,1)
    #     X_test_sorted = X_test[sort_idx]
    #     y_test_sorted = y_test[sort_idx]
    #     y_test_pred_sorted = y_test_pred[sort_idx]
    #     # 2. Plot
    #     scatter_ax[Datasets.index(dataset)].scatter(X_test_sorted, y_test_sorted, color="purple", alpha=0.6, label="True Test Data")
    #     scatter_ax[Datasets.index(dataset)].plot(X_test_sorted, y_test_pred_sorted, color="red", linewidth=2,
    #                                              label='Predicted values')
    #
    #     # scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',label="Ideal")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'ITA Scores')

    # scatter_ax[Datasets.index(dataset)].set_ylabel(f'True Label and Prediction from BSpline+LR')
    # scatter_ax[Datasets.index(dataset)].set_title(f'{dataset.upper()} - BSpline+LR Regression')
    # scatter_ax[Datasets.index(dataset)].legend()

    predicted_labels_train, predicted_labels_test, all_predictions = get_predicted_gains(y_train_pred, y_test_pred, all_pred)
    print(f'Test RMSE: {test_rmse}, test_rsq = {test_rsq}')
    # exit()

    return test_rsq, test_corr, test_rmse, predicted_labels_train, predicted_labels_test, all_predictions


def evaluate_BSpline_Ridge(train_data, test_data, all_data, bias_val):

    # Unpack train and test data
    X_train, y_train = train_data
    X_test, y_test = test_data
    ALL_X, ALL_y = all_data

    print(f'train and test shapes: {X_train.shape} {y_train.shape} {X_test.shape} {y_test.shape}')
    # pipeline for Spline transformation + Ridge Regression
    pipeline = Pipeline([
        ('spline', SplineTransformer(include_bias=bias_val)),
        ('ridge', Ridge())])

    # hyperparameter grid
    max_knot = int(math.sqrt(X_train.shape[0]))
    print(f'max_knot = {max_knot}')
    param_grid = {
        'spline__degree': [i for i in range(2, 7)],
        'spline__n_knots': [c for c in range(2,max_knot+10,1)],
        'ridge__alpha': [0.001,0.002,0.003,0.004,0.005,]+[i/100 for i in range(1,11)]+[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        # 'spline__n_knots': [2, 3, 4, 5, 6] + [c for c in range(10, max_knot + 10, 5)]
    }
    print(f'param_grid = {param_grid}')

    if dataset != 'ETTm1':
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error')
    else:
        grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train[:, np.newaxis], y_train)

    print(f"{dataset}, Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", -grid_search.best_score_, )
    print(f'train variance: {torch.var(y_train)}, test variance: {torch.var(y_test)}')

    best_pipeline = grid_search.best_estimator_

    best_pipeline.fit(X_train[:, np.newaxis], y_train)
    coefs = best_pipeline.named_steps['ridge'].coef_
    print(f'coefs = {coefs}')
    print("intercept_:", best_pipeline.named_steps['ridge'].intercept_)

    # E.g. to see the shape after the best_pipeline transforms the data
    X_transformed = best_pipeline[:-1].transform(X_test[:, np.newaxis])  # all steps except the final 'linear'
    # print("Shape after best_pipeline transforms:", X_transformed.shape)

    spline_transformer = SplineTransformer(n_knots=grid_search.best_params_['spline__n_knots'],
                                           degree=grid_search.best_params_['spline__degree'],
                                           include_bias=bias_val)
    # X_spline = spline_transformer.fit_transform(X_train[:, np.newaxis],y_train)
    # X_spline_test = spline_transformer.transform(X_test[:, np.newaxis])
    # print(f"X_spline_test = {X_spline_test.shape}")
    #
    # exit()
    # print(f'bias = {pipeline.named_steps["linear"].bias_}')

    # Combinations = [f'{param["spline__degree"]}_{param["spline__n_knots"]}' for param in grid_search.cv_results_['params']]
    # for each_combination, each_score in zip(Combinations, grid_search.cv_results_['mean_test_score']):
    #     n_knots = int(each_combination.split('_')[1])
    #     degree = int(each_combination.split('_')[0])
    #     spline_transformer = SplineTransformer(n_knots=n_knots, degree=degree,
    #                                            include_bias=False)
    #     X_spline = spline_transformer.fit_transform(X_train[:, np.newaxis])
    #     basis_sums = X_spline.sum(axis=1)  # Sum across columns (basis functions)
    #     print(f'Combination (degree,n_knots) = {(degree,n_knots)}, cross-val score = {-each_score:0.6f},\t basis sum = {np.mean(basis_sums):.6f}')

    # # plot combination vs cross-validation score
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.plot(grid_search.cv_results_['mean_test_score'], label="CV MSE")
    # ax.set_xlabel("Hyperparameter Combination")
    # ax.set_ylabel("Mean Squared Error")
    #
    # ax.legend()
    # plt.show()

    # for n_knots in param_grid['spline__n_knots']:

    # if dataset=='chemical':
    #     # # Plot spline basis for different n_knots
    #     if len(param_grid['spline__n_knots'])%2==0:
    #         fig_basis_chemical, basis_ax_chemical = plt.subplots(2, len(param_grid['spline__n_knots'])//2, figsize=(10, 6))
    #     else:
    #         fig_basis_chemical, basis_ax_chemical = plt.subplots(2, len(param_grid['spline__n_knots'])//2+1, figsize=(10, 6))
    #
    #
    #     for n_knots in param_grid['spline__n_knots']:
    #         test = SplineTransformer(n_knots=n_knots, degree=grid_search.best_params_['spline__degree'], include_bias=False)
    #         X_spline = test.fit_transform(X_train[:, np.newaxis])
    #         basis_sums = X_spline.sum(axis=1)  # Sum across columns (basis functions)
    #
    #         col_idx = param_grid['spline__n_knots'].index(n_knots)
    #         if col_idx < len(param_grid['spline__n_knots']) // 2:
    #             row = 0
    #         else:
    #             row = 1
    #             col_idx = len(param_grid['spline__n_knots']) - (col_idx + 1)
    #
    #         basis_ax_chemical[row, col_idx].plot(X_train, X_spline)
    #         basis_ax_chemical[row, col_idx].plot(X_train, basis_sums, color='black', linewidth=2, label = 'Sum')
    #         basis_ax_chemical[row, col_idx].set_title(#f"degree: {grid_search.best_params_['spline__degree']}, n_knots={n_knots}, \n"
    #                                                   f"basis sum = {np.mean(basis_sums):.6f}")
    #         # basis_ax_chemical[row, col_idx].set_xlabel(f"{dataset} -- Input Feature")
    #         basis_ax_chemical[row, col_idx].set_ylabel(f"Basis -(degree,n_knots) = {(grid_search.best_params_['spline__degree'],n_knots)}")
    #         basis_ax_chemical[row, col_idx].legend()
    #     fig_basis_chemical.suptitle(f'{dataset.upper()} -- B-Spline Basis Functions')
    #     fig_basis_chemical.show()

    # n_knots = grid_search.best_params_['spline__n_knots']
    spline_transformer = SplineTransformer(n_knots=grid_search.best_params_['spline__n_knots'],
                                           degree=grid_search.best_params_['spline__degree'],
                                           include_bias=bias_val)
    X_spline = spline_transformer.fit_transform(X_train[:, np.newaxis])
    X_spline_test = spline_transformer.transform(X_test[:, np.newaxis])

    # X_plot = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)
    # X_spline_plot = spline_transformer.transform(X_plot)
    # basis_sums_plot = X_spline_plot.sum(axis=1)
    # fig_idx = Datasets.index(dataset)
    # basis_ax[fig_idx].plot(X_plot, X_spline_plot, alpha=0.7)
    # basis_ax[fig_idx].plot(X_plot, basis_sums_plot, color='black', linewidth=2, label="Sum of Basis")
    # basis_ax[fig_idx].set_title(
    #     f"degree: {grid_search.best_params_['spline__degree']}, n_knots={grid_search.best_params_['spline__n_knots']}, basis sum = {np.mean(basis_sums_plot):.6f}")
    # basis_ax[fig_idx].set_xlabel(f"{dataset} -- Input Feature")
    # if bias_val:
    #     basis_ax[fig_idx].set_ylabel(f"Basis Function Value with Bias")
    # else:
    #     basis_ax[fig_idx].set_ylabel(f"Basis Function Value without Bias")
    # basis_ax[fig_idx].legend()

    # print(f'row = {row}, fig_idx = {fig_idx}')
    # basis_ax[fig_idx].plot(X_train, X_spline)
    # basis_ax[fig_idx].plot(X_train, basis_sums, color='black', linewidth=2, label = "Sum of Basis")

    # Evaluate on test data
    y_train_pred = grid_search.best_estimator_.predict(X_train[:, np.newaxis])
    y_test_pred = grid_search.best_estimator_.predict(X_test[:, np.newaxis])
    all_pred = grid_search.best_estimator_.predict(ALL_X[:, np.newaxis])
    test_mse = mean_squared_error(y_test, y_test_pred)
    print("Test MSE:", test_mse)

    # Visualization

    # Evaluate the model
    test_corr = np.corrcoef(y_test, y_test_pred)[0][1]
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_rsq = r2_score(y_test, y_test_pred)

    print(f'shape of X_spline: {X_spline_test.shape}')
    import matplotlib.pyplot as plt
    for jdx in range(0, len(X_test)):
        if abs(y_test_pred[jdx] - y_test[jdx]) > 1:
            print(f'jdx {jdx}: ITA: {X_test[jdx]}, y_test: {y_test[jdx]}, y_test_pred: {y_test_pred[jdx]}')
            print(f'X_spline_test: {X_spline_test[jdx]}')

            # plt.plot(X_spline_test[jdx], color='black', linewidth=2)
        # else:
        #     if jdx%30==0:
        #         print(f'Other points: jdx {jdx}: ITA: {X_test[jdx]}, y_test: {y_test[jdx]}, y_test_pred: {y_test_pred[jdx]}')
        #         print(f'X_spline_test: {X_spline_test[jdx]}')
    # plt.show()
    # # Optional: Visualize the basis functions and their sum
    X_plot = np.linspace(X_test.min(), X_test.max(), 500).reshape(-1, 1)
    X_spline_plot = spline_transformer.transform(X_plot)
    basis_sums_plot = X_spline_plot.sum(axis=1)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(X_plot, X_spline_plot, label="Basis Functions")
    # plt.plot(X_plot, basis_sums_plot, color='black', linewidth=2, label="Sum of Basis (Should be 1)")
    # plt.xlabel('X-test')
    # plt.ylabel('spline_transformer.transform(X_test)')
    # # plt.legend()
    # plt.title("B-Spline Basis Functions and Their Sum")
    # plt.show()

    '''index sort X_test'''

    # train_test_fig, axes = plt.subplots(1, 2)
    #
    # sort_idx = np.argsort(X_train)
    # X_train = X_train[sort_idx]
    # y_train_pred = y_train_pred[sort_idx]
    # y_train = y_train[sort_idx]
    # axes[0].scatter(X_train, y_train, color="blue", label="True Test Data (sorted)")
    # axes[0].scatter(X_train, y_train_pred, color="red", label="Predictions (sorted)")
    # axes[0].grid()
    # axes[0].set_title(f'train data')
    # axes[0].set_xlabel(
    #     f'train features: Spline degree = {grid_search.best_params_["spline__degree"]}, n_knots = {grid_search.best_params_["spline__n_knots"]}')
    # axes[0].set_ylabel('train labels and predictions')
    # axes[0].legend()
    #
    # sort_idx = np.argsort(X_test)
    # X_test = X_test[sort_idx]
    # y_test_pred = y_test_pred[sort_idx]
    # y_test = y_test[sort_idx]
    # axes[1].scatter(X_test, y_test, color="blue", label="True Test Data (sorted)")
    # axes[1].scatter(X_test, y_test_pred, color="red", label="Predictions (sorted)")
    # axes[1].grid()
    # axes[1].set_title(f'RMSE: {test_rmse:.5f}, test rsq: {test_rsq:.5f}, correlation = {test_corr:.4f}')
    # axes[1].set_xlabel(
    #     f'test features: Spline degree = {grid_search.best_params_["spline__degree"]}, n_knots = {grid_search.best_params_["spline__n_knots"]}')
    # axes[1].set_ylabel('test labels and predictions')
    # axes[1].legend()
    # plt.show()

    # plot_type = 'X_test'
    # if plot_type == 'y_test':
    #     scatter_ax[Datasets.index(dataset)].scatter(y_test, y_test_pred, alpha=0.7, color="purple",
    #                                                 label='True values vs KNN prediction')
    #     # scatter_ax[Datasets.index(dataset)].scatter(y_test,X_test, label='True values vs ITA Scores')
    #     scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',
    #                                              label="Ideal (min-max)")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'True Label')
    #     scatter_ax[Datasets.index(dataset)].legend()
    # else:
    #     # scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test, color="purple", alpha = 0.6, label="True Test Data")
    #     # scatter_ax[Datasets.index(dataset)].scatter(X_test, y_test_pred, color="blue", alpha = 0.7, label='Predicted values')
    #     #plot a line with the predictions
    #     # 1. Sort by the values of X_test so the line plot is continuous
    #
    #     sort_idx = np.argsort(X_test.ravel())  # Flatten if X_test is shape (N,1)
    #     X_test_sorted = X_test[sort_idx]
    #     y_test_sorted = y_test[sort_idx]
    #     y_test_pred_sorted = y_test_pred[sort_idx]
    #     # 2. Plot
    #     scatter_ax[Datasets.index(dataset)].scatter(X_test_sorted, y_test_sorted, color="purple", alpha=0.6, label="True Test Data")
    #     scatter_ax[Datasets.index(dataset)].plot(X_test_sorted, y_test_pred_sorted, color="red", linewidth=2,
    #                                              label='Predicted values')
    #
    #     # scatter_ax[Datasets.index(dataset)].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',label="Ideal")
    #     scatter_ax[Datasets.index(dataset)].set_xlabel(f'ITA Scores')

    # scatter_ax[Datasets.index(dataset)].set_ylabel(f'True Label and Prediction from BSpline+LR')
    # scatter_ax[Datasets.index(dataset)].set_title(f'{dataset.upper()} - BSpline+LR Regression')
    # scatter_ax[Datasets.index(dataset)].legend()

    predicted_labels_train, predicted_labels,all_predictions = get_predicted_gains(y_train_pred, y_test_pred,all_pred)
    print(f'Test RMSE: {test_rmse}, test_rsq = {test_rsq}')
    # exit()
    return test_rsq, test_corr, test_rmse, predicted_labels_train, predicted_labels,all_predictions




def evaluate_GPR_Matern(train_data,test_data):
    X_train, y_train = train_data
    X_test, y_test = test_data

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Scale the feature (often helpful for GPR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    #Matern kernel
    kernel = C(1.0, (1e-10, 1e3)) * Matern(length_scale=1.0,
                                               nu=1.5,
                                               length_scale_bounds=(1e-10, 1e3))

    # Instantiate a base GPR model
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,  # initial alpha
        n_restarts_optimizer=20,
        random_state=experiment
    )

    # Define the parameter grid
    param_grid = {
        'kernel__k1__constant_value': [1.0, 10.0, 100.0],
        'kernel__k2__length_scale': [1.0, 0.1, 10.0, 100.0],
        'kernel__k2__length_scale_bounds': [(1e-5, 1e3), (1e-10, 1e3),(1e-20, 1e3), (1e-3, 1e3)],
        'kernel__k2__nu': [1.5, 2.5],
        'alpha': [1e-10, 1e-5, 1e-3, 1e-1],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=gpr,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,  # Use all available cores
        verbose=1  # Increase to see search progress
    )

    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)

    # Print best parameters
    print("Best parameters found:", grid_search.best_params_)

    # Retrieve the best GPR
    best_gpr = grid_search.best_estimator_

    # Make predictions on the test set
    y_test_pred, y_std = best_gpr.predict(X_test_scaled, return_std=True)
    y_train_pred,_ = best_gpr.predict(X_train_scaled, return_std=True)
    # Evaluate performance
    mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mse)
    test_rsq = r2_score(y_test, y_test_pred)
    test_corr = np.corrcoef(y_test, y_test_pred)[0, 1]

    print(f"Test MSE: {mse:.5f}")
    print(f"Test R²: {test_rsq:.5f}")
    print(f"Test RMSE: {test_rmse:.5f}")
    print(f"Test Correlation: {test_corr:.5f}")

    print(f'y_test_pred = {y_test_pred}')

    predicted_labels_train, predicted_labels = get_predicted_gains(y_train_pred, y_test_pred)

    return test_rsq, test_corr, test_rmse,predicted_labels_train, predicted_labels



# fig,ax = plt.subplots(3, 5)
# fig,residual_ax = plt.subplots(1,3)

# fig2,ax2 = plt.subplots(3,3)
# subfigs = fig2.subfigures(nrows=3, ncols=1)

# fig2.suptitle(f'Decision Tree Boosting on ITA Scores')
# # clear subplots
# for ax in ax2:
#     ax.remove()
# add subfigure per subplot
# gridspec = ax2[0].get_subplotspec().get_gridspec()
# subfigs = [fig2.add_subfigure(gs) for gs in gridspec]
# figtree,tree_ax = plt.subplots(1,3)



SEEDS = [83, 35, 9, 22,21, 14,  29, 55, 10, 8]
for experiment in SEEDS:
    # affinity_scoring_method = 'ITA' #from fifty et al 2021
    affinity_scoring_method = 'ETAP_Affinity' #Our task-affinity score based on linear approximation.
    model_name = 'KNN'
    # model_name ='BSPlines_LR'
    # model_name = 'BSPlines_Ridge'
    # model_name = 'RandomForest'
    # fig_visualization,visualization_ax = plt.subplots(1,3)
    # fig_residual, residual_ax = plt.subplots(1,3)
    Datasets = ['celebA', 'chemical', 'ETTm1', 'Occupancy']
    AFFINE_RSQ_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}
    AFFINE_CORR_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}
    AFFINE_RMSE_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}
    MODEL_RSQ_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}
    MODEL_CORR_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}
    MODEL_RMSE_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}
    Training_Samples_Dict = {'celebA':[],'chemical':[],'ETTm1':[], 'Occupancy':[]}


    for ts_idx in range(0,8):

        Dataset_Rsq = []
        Dataset_Corr = []
        Dataset_RMSE = []
        Training_Groups = []
        ITA_Corr = []
        ITA_Rsq = []
        ITA_RMSE = []
        for dataset in Datasets:
            seed_everything(experiment)
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


            num_training_sample = Training_Samples[ts_idx]


            Num_Tasks = len(TASKS)
            if dataset == 'celebA':
                if affinity_scoring_method == 'ETAP_Affinity':
                    ITA_file = f'AffinityScores/{dataset}/ITA_Approximation_wo_Momentum_matrix_New.csv'
                if affinity_scoring_method == 'ITA':
                    ITA_file = f'AffinityScores/{dataset}/ITA_matrix_New.csv'
            if dataset == 'chemical':
                if affinity_scoring_method == 'ETAP_Affinity':
                    ITA_file = f'AffinityScores/{dataset}/ITA_Approx_wo_momentum_matrix_avg_FIXED_ALL.csv'
                if affinity_scoring_method == 'ITA':
                    ITA_file = f'AffinityScores/{dataset}/ITA_matrix_avg_SGD.csv'
            if dataset == 'ETTm1':
                if affinity_scoring_method == 'ETAP_Affinity':
                    ITA_file = f'AffinityScores/{dataset}/ITA_Approximation_w_Momentum_matrix_SGD_newRes.csv'
                if affinity_scoring_method == 'ITA':
                    ITA_file = f'AffinityScores/{dataset}/ITA_matrix_SGD_newRes.csv'
            if dataset == 'Occupancy':
                if affinity_scoring_method == 'ETAP_Affinity':
                    ITA_file = f'AffinityScores/{dataset}/ITA_Approx_w_Momentum_matrix_SEED_2024.csv'
                if affinity_scoring_method == 'ITA':
                    ITA_file = f'AffinityScores/{dataset}/ITA_matrix_SEED_2024.csv'


            ITA_df = pd.read_csv(ITA_file)
            Num_Tasks = len(TASKS)
            revised_integrals_ITA_Approx = {task: {task: 0. for task in TASKS} for task in TASKS}
            for i in range(len(TASKS)):
                for j in range(len(TASKS)):
                    revised_integrals_ITA_Approx[TASKS[i]][TASKS[j]] = ITA_df.iloc[i][j]
            revised_integrals = revised_integrals_ITA_Approx
            TASKS = sorted(TASKS)

            ITA_df = pd.read_csv(ITA_file)
            Num_Tasks = len(TASKS)
            revised_integrals_ITA_Approx = {task: {task: 0. for task in TASKS} for task in TASKS}
            for i in range(len(TASKS)):
                for j in range(len(TASKS)):
                    revised_integrals_ITA_Approx[TASKS[i]][TASKS[j]] = ITA_df.iloc[i][j]
            revised_integrals = revised_integrals_ITA_Approx
            TASKS = sorted(TASKS)

            mtg_data_path = 'PredData/'
            if dataset == 'Occupancy':
                GAIN_SEED = 2024
                X_train = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Train_SEED_{GAIN_SEED}.pt')
                y_train = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_gains_Train_SEED_{GAIN_SEED}.pt')
                testx = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_tasks_map_Test_SEED_{GAIN_SEED}.pt')
                testy = torch.load(f'{mtg_data_path}RANDOMIZED_{dataset}_gains_Test_SEED_{GAIN_SEED}.pt')
                all_groups = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_tasks_map_SEED_{GAIN_SEED}.pt')
                all_labels = torch.load(f'PredData/ALL_RANDOMIZED_Occupancy_gains_SEED_{GAIN_SEED}.pt')
            else:
                X_train = torch.load(f'{mtg_data_path}/RANDOMIZED_{dataset}_tasks_map_from_ground_truth_train.pt')
                y_train = torch.load(f'{mtg_data_path}/RANDOMIZED_{dataset}_gain_collection_from_ground_truth_train.pt')
                testx = torch.load(f'{mtg_data_path}/FINAL_RANDOMIZED_{dataset}_tasks_map_test.pt')
                testy = torch.load(f'{mtg_data_path}FINAL_RANDOMIZED_{dataset}_gain_test.pt')
                all_groups = torch.load(f'{mtg_data_path}{dataset}_Combined_X.pt')
                all_labels = torch.load(f'{mtg_data_path}{dataset}_Combined_Y.pt')

            print(f'shapes of X_train: {X_train.shape}, y_train: {y_train.shape}')

            if num_training_sample < len(X_train):
                random_indices = random.sample(range(len(X_train)), num_training_sample)
                remaining_indices = list(set(range(len(X_train))) - set(random_indices))

                # Create the new training set
                new_X_train = X_train[random_indices]
                new_y_train = y_train[random_indices]

                found_flag = False
                for each in new_X_train:
                    if np.sum(each) >= Num_Tasks:
                        found_flag = True

                if not found_flag:
                    print(f'{dataset}\tadding the baseline model to the training data')
                    for i in range(len(X_train)):
                        if ((len(X_train[i][X_train[i] == 1]) == len(X_train[0]))):
                            group = X_train[i]
                            label = y_train[i]

                    '''remove one at random from training data'''
                    get_random_index = random.sample(range(len(new_X_train)), 1)[0]
                    new_X_train = np.delete(new_X_train, get_random_index, 0)
                    new_y_train = np.delete(new_y_train, get_random_index, 0)
                    print(f'get_random_index = {get_random_index}')

                    '''add to new training set'''
                    new_X_train = np.append(new_X_train, [group], axis=0)
                    new_y_train = np.append(new_y_train, [label], axis=0)

                    '''check if baseline model exists'''
                    for each in new_X_train:
                        if np.sum(each) >= Num_Tasks:
                            found_flag = True

                if found_flag:
                    print(f'baseline model already exists')
                    # for idx in range(len(new_X_train)):
                    #     print(f'{new_X_train[idx]} -> {new_y_train[idx]}')

                X_train = new_X_train
                y_train = new_y_train


            testx, testy = torch.FloatTensor(testx), torch.FloatTensor(testy)
            X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            print(f'Training Data Shapes: {X_train.shape}, {y_train.shape}')
            print(f'Testing Data Shapes: {testx.shape}, {testy.shape}')
            '''remove test sample where map sum =1'''
            all_indices_test = list(set(range(len(testx))))
            for idx in range(len(testx)):
                if torch.sum(testx[idx]) == 1:
                    all_indices_test.remove(idx)
            testx = testx[all_indices_test]
            testy = testy[all_indices_test]


            all_groups, all_labels = torch.FloatTensor(all_groups), torch.FloatTensor(all_labels)

            train_labels = y_train
            all_train_mask = X_train
            Training_Groups.append(len(X_train))

            print(f'shape of x_train: {X_train.shape}, testx : {testx.shape}, all_groups: {all_groups.shape}')
            ita_dict_train, X_train_predicted_ITA = get_ITA(X_train, revised_integrals, TASKS)
            ita_dict_test, X_test_predicted_ITA = get_ITA(testx, revised_integrals, TASKS)
            ita_dict_combined, allGroups_predicted_ITA = get_ITA(all_groups, revised_integrals, TASKS)
            # exit(0)

            flattened_features_train = []
            flattened_features_test = []
            flattened_labels_train = []
            flattened_labels_test = []

            flattened_features_all = []
            flattened_labels_all = []

            only_ita_train = []
            only_ita_test = []
            only_ita_all = []

            for i in range(len(X_train)):
                for j in range(len(X_train[i])):
                    if X_train[i][j] > 0:
                        feat_2 = X_train_predicted_ITA[i][j]
                        only_ita_train.append(feat_2)
                        flattened_features_train.append(feat_2)
                        flattened_labels_train.append(y_train[i][j])
            for i in range(len(testx)):
                for j in range(len(testx[i])):
                    if testx[i][j] > 0:
                        feat_2 = X_test_predicted_ITA[i][j]
                        only_ita_test.append(feat_2)
                        flattened_features_test.append(feat_2)
                        flattened_labels_test.append(testy[i][j])

            for i in range(len(all_groups)):
                for j in range(len(all_groups[i])):
                    if all_groups[i][j] > 0:
                        feat_2 = allGroups_predicted_ITA[i][j]
                        only_ita_all.append(feat_2)
                        flattened_features_all.append(feat_2)
                        flattened_labels_all.append(all_labels[i][j])

            print(f'total labels: {len(flattened_labels_train), len(flattened_labels_test)}')

            transformed_y_ita_train, a, b = affine_transformation(flattened_labels_train, only_ita_train)
            transformed_y_ita_train = torch.tensor(transformed_y_ita_train)
            print(f'dataset = {dataset.upper()}, num_training_samples:{num_training_sample} a = {a}, b = {b}')


            flattened_features_train = torch.tensor(flattened_features_train, dtype=torch.float32)
            flattened_features_test = torch.tensor(flattened_features_test, dtype=torch.float32)
            flattened_labels_train = torch.tensor(flattened_labels_train, dtype=torch.float32)
            flattened_labels_test = torch.tensor(flattened_labels_test, dtype=torch.float32)
            flattened_features_all = torch.tensor(flattened_features_all, dtype=torch.float32)
            flattened_labels_all = torch.tensor(flattened_labels_all, dtype=torch.float32)

            correlation_feat2 = np.corrcoef(flattened_features_train, flattened_labels_train)[0, 1]

            # print(f"Correlation of Feature 1 (MTGNet) with Labels: {correlation_feat1:.5f}",end=' ')
            # print(f"Correlation of Feature (ITA) with Labels: {correlation_feat2:.5f}")
            # correlation_feat1 = np.corrcoef(flattened_features_test[:, 0], flattened_labels_test)[0, 1]
            correlation_feat2 = np.corrcoef(flattened_features_test, flattened_labels_test)[0, 1]
            rsq_feat2 = r2_score(flattened_labels_train, flattened_features_train)

            rsq_feat1 = r2_score(flattened_labels_test, flattened_features_test)
            print(f"R-Square of Feature 2 (ITA wo Affine Transformation) with Labels (test): {rsq_feat1:.5f}")
            
            transformed_y_test_predicted = a * np.array(only_ita_test) + b
            rsq_feat2 = r2_score(flattened_labels_test, transformed_y_test_predicted)
            print(f"R-Square of Feature 2 (ITA w affine transformation) with Labels (test): {rsq_feat2:.5f}")
            # MTGNet_Rsq.append(rsq_feat1)
            ITA_Rsq.append(rsq_feat2)

            rmse_feat2 = np.sqrt(mean_squared_error(flattened_labels_test, transformed_y_test_predicted))
            ITA_RMSE.append(rmse_feat2)

            transformed_y_test_predicted = torch.tensor(transformed_y_test_predicted)
            flattened_features_train = transformed_y_ita_train
            flattened_features_test = transformed_y_test_predicted
            correlation_feat2 = np.corrcoef(transformed_y_test_predicted, flattened_labels_test)[0, 1]
            print(f"Correlation of Feature 2 (ITA) with Labels (test): {correlation_feat2:.5f}")
            ITA_Corr.append(correlation_feat2)

            # print(f'train shapes: {flattened_labels_train.shape, flattened_features_train.shape}')
            # print(f'test shapes: {flattened_labels_test.shape, flattened_features_test.shape}')
            print(f'After Boosting: ')


            if model_name == 'KNN':
                test_rsq, test_corr, test_rmse, train_predictions, test_predictions, all_predictions = evaluate_knn_performance(
                    (flattened_features_train, flattened_labels_train),
                    (flattened_features_test, flattened_labels_test), (flattened_features_all, flattened_labels_all))

            if model_name == 'BSPlines_LR':
                bias_val = True
                test_rsq, test_corr, test_rmse, train_predictions, test_predictions, all_predictions = evaluate_BSpline_LR(
                    (flattened_features_train, flattened_labels_train),
                    (flattened_features_test, flattened_labels_test),
                    (flattened_features_all, flattened_labels_all), bias_val)
            if model_name == 'BSPlines_Ridge':
                bias_val = True
                test_rsq, test_corr, test_rmse, train_predictions, test_predictions, all_predictions = evaluate_BSpline_Ridge(
                    (flattened_features_train, flattened_labels_train),
                    (flattened_features_test, flattened_labels_test), (flattened_features_all, flattened_labels_all),
                    bias_val)

                print(f'Test R^2: {test_rsq}, Test RMSE: {test_rmse}, test_corr: {test_corr}')
                # exit()

            if model_name == 'RandomForest':
                test_rsq, test_corr, test_rmse, train_predictions, test_predictions, all_predictions = evaluate_Random_forest_performance(
                    (flattened_features_train, flattened_labels_train),
                    (flattened_features_test, flattened_labels_test), (flattened_features_all, flattened_labels_all))

            Dataset_Corr.append(test_corr)
            Dataset_Rsq.append(test_rsq)
            Dataset_RMSE.append(test_rmse)

            print("train_labels shape =", train_labels.shape)
            print("train_predictions shape =", train_predictions.shape)
            print(f'type of train_labels: {type(train_labels)}, type of testy = {type(testy)}')
            print(f'type of train_predictions: {type(train_predictions)}, type of test_predictions = {type(test_predictions)}')
            residuals_train = train_labels - train_predictions
            residuals_test = testy - test_predictions
            residuals_train = torch.tensor(residuals_train)
            residuals_test = torch.tensor(residuals_test)
            residuals_all = all_groups - all_predictions

                # exit()
            if model_name == 'B-SPlines':
                bias_term = f'_bias_{bias_val}'
            else:
                bias_term = ''

            if affinity_scoring_method == 'ITA':
                ita_method_name = 'TAG_ITAScore'
            if 'ETAP_Affinity' in affinity_scoring_method:
                ita_method_name = 'Task_Affinity_Score'

            '''save the residuals to a pytorch tensor'''
            torch.save(residuals_train,
                           f'RESULTS/Initial_prediction/{ita_method_name}_{dataset}_{model_name}{bias_term}_residuals_train_{num_training_sample}_seed_{experiment}.pt')
            torch.save(residuals_test,
                       f'RESULTS/Initial_prediction/{ita_method_name}_{dataset}_{model_name}{bias_term}_residuals_test_{num_training_sample}_seed_{experiment}.pt')
            torch.save(residuals_all,f'RESULTS/Initial_prediction/{ita_method_name}_{dataset}_{model_name}{bias_term}_residuals_all_{num_training_sample}_seed_{experiment}.pt')
            final_task_map = torch.cat([all_train_mask, testx], 0)
            print(f'shape of final_task_map = {final_task_map.shape}')

            final_predictions = torch.cat([train_predictions, test_predictions], 0)
            print(f'shape of final_predictions = {final_predictions.shape}')

            tasks_map_file_name = f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}{bias_term}_Maps_{dataset}_{num_training_sample}_seed_{experiment}.pkl'
            with open(tasks_map_file_name, "wb") as fp:
                pickle.dump(final_task_map, fp)

            pred_file_name = f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}{bias_term}_Predictions_{dataset}_{num_training_sample}_seed_{experiment}.pkl'
            with open(pred_file_name, "wb") as fb:
                pickle.dump(final_predictions, fb)

            pred_file_name_ALL = f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}{bias_term}_ALL_Predictions_{dataset}_{num_training_sample}_seed_{experiment}.pkl'
            with open(pred_file_name_ALL, "wb") as fb:
                pickle.dump(all_predictions, fb)



        print(f'Dataset = {Datasets}')
        print(f'Number of Training groups = {num_training_sample}')
        print(f'Before\t\tAfter')
        print('Dataset |    Groups   |            R-Square        |          Correlation     |             RMSE         |')
        print(f'       -------------------------------------------------------------------------------------')
        print(f'        |            | (transformation) |  ({model_name})  | (transformation) | ({model_name}) | (transformation) | ({model_name}) |')
        print(f'---------------------------------------------------------------------------------------------')
        # print(f'MTGNet Rsquare = {MTGNet_Rsq}')
        for i in range(len(ITA_Rsq)):
            print(f'{Datasets[i][:5]}\t\t {Training_Groups[i]}\t\t {ITA_Rsq[i]:.6f}\t{Dataset_Rsq[i]:.6f}\t'
                  f'{ITA_Corr[i]:.10f}\t{Dataset_Corr[i]:.6f}\t {ITA_RMSE[i]:.6f}\t{Dataset_RMSE[i]:.6f}')
            AFFINE_RSQ_Dict[Datasets[i]].append(ITA_Rsq[i])
            AFFINE_CORR_Dict[Datasets[i]].append(ITA_Corr[i])
            AFFINE_RMSE_Dict[Datasets[i]].append(ITA_RMSE[i])
            MODEL_RSQ_Dict[Datasets[i]].append(Dataset_Rsq[i])
            MODEL_CORR_Dict[Datasets[i]].append(Dataset_Corr[i])
            MODEL_RMSE_Dict[Datasets[i]].append(Dataset_RMSE[i])
            Training_Samples_Dict[Datasets[i]].append(Training_Groups[i])

        print(f'---------------------------------------------------------------------------------------------')

    '''dictionaries to dataframe'''
    df_rsq = pd.DataFrame.from_dict(AFFINE_RSQ_Dict)
    df_corr = pd.DataFrame.from_dict(AFFINE_CORR_Dict)
    df_rmse = pd.DataFrame.from_dict(AFFINE_RMSE_Dict)
    df_model_rsq = pd.DataFrame.from_dict(MODEL_RSQ_Dict)
    df_model_corr = pd.DataFrame.from_dict(MODEL_CORR_Dict)
    df_model_rmse = pd.DataFrame.from_dict(MODEL_RMSE_Dict)
    df_training_samples = pd.DataFrame.from_dict(Training_Samples_Dict)
    '''join the dataframes'''
    model_name = model_name.replace('-','')
    df_rsq = df_rsq.join(df_model_rsq, lsuffix='_Affine', rsuffix=f'_{model_name}')
    df_corr = df_corr.join(df_model_corr, lsuffix='_Affine', rsuffix=f'_{model_name}')
    df_rmse = df_rmse.join(df_model_rmse, lsuffix='_Affine', rsuffix=f'_{model_name}')
    # df_training_samples = df_training_samples.join(df_model_rmse, lsuffix='_ITA', rsuffix='_Model')
    print(df_training_samples.columns)
    print(df_rsq.columns)
    print(df_corr.columns)
    print(df_rmse.columns)
    '''join the dataframes'''
    df_rsq = df_rsq.join(df_corr, lsuffix='_Rsq', rsuffix='_Correlation')

    df_rsq['celebA_Affine_RMSE'] = AFFINE_RMSE_Dict['celebA']
    df_rsq[f'celebA_{model_name}_RMSE'] = MODEL_RMSE_Dict['celebA']
    df_rsq[f'chemical_Affine_RMSE'] = AFFINE_RMSE_Dict['chemical']
    df_rsq[f'chemical_{model_name}_RMSE'] = MODEL_RMSE_Dict['chemical']
    df_rsq[f'ETTm1_Affine_RMSE'] = AFFINE_RMSE_Dict['ETTm1']
    df_rsq[f'ETTm1_{model_name}_RMSE'] = MODEL_RMSE_Dict['ETTm1']
    df_rsq[f'celebA_Training_Samples'] = Training_Samples_Dict['celebA']
    df_rsq[f'chemical_Training_Samples'] = Training_Samples_Dict['chemical']
    df_rsq[f'ETTm1_Training_Samples'] = Training_Samples_Dict['ETTm1']
    df_rsq[f'Occupancy_Training_Samples'] = Training_Samples_Dict[f'Occupancy']
    df_rsq[f'Occupancy_Affine_RMSE'] = AFFINE_RMSE_Dict[f'Occupancy']
    df_rsq[f'Occupancy_{model_name}_RMSE'] = MODEL_RMSE_Dict[f'Occupancy']



    print(df_rsq.columns)


    celebA_results = df_rsq[['celebA_Training_Samples','celebA_Affine_Rsq', f'celebA_{model_name}_Rsq',
                             'celebA_Affine_Correlation',  f'celebA_{model_name}_Correlation','celebA_Affine_RMSE', f'celebA_{model_name}_RMSE']]
    '''remove dataset name from columns'''
    celebA_results.columns = ['Training_Samples', 'Affine_Rsq', f'{model_name}_Rsq','Affine_Correlation',  f'{model_name}_Correlation','Affine_RMSE', f'{model_name}_RMSE']
    celebA_results['Dataset'] = ['celebA' for i in range(len(Training_Samples))]
    chemical_results = df_rsq[['chemical_Training_Samples','chemical_Affine_Rsq', f'chemical_{model_name}_Rsq',
                             'chemical_Affine_Correlation',  f'chemical_{model_name}_Correlation','chemical_Affine_RMSE', f'chemical_{model_name}_RMSE']]
    chemical_results.columns = ['Training_Samples','Affine_Rsq', f'{model_name}_Rsq','Affine_Correlation',  f'{model_name}_Correlation','Affine_RMSE', f'{model_name}_RMSE']
    chemical_results['Dataset'] = ['chemical' for i in range(len(Training_Samples))]

    ETTm1_results = df_rsq[['ETTm1_Training_Samples','ETTm1_Affine_Rsq', f'ETTm1_{model_name}_Rsq',
                             'ETTm1_Affine_Correlation',  f'ETTm1_{model_name}_Correlation','ETTm1_Affine_RMSE', f'ETTm1_{model_name}_RMSE']]
    ETTm1_results.columns = ['Training_Samples','Affine_Rsq', f'{model_name}_Rsq','Affine_Correlation',  f'{model_name}_Correlation','Affine_RMSE', f'{model_name}_RMSE']
    ETTm1_results['Dataset'] = ['ETTm1' for i in range(len(Training_Samples))]

    Occupancy_results = df_rsq[[f'Occupancy_Training_Samples', f'Occupancy_Affine_Rsq', f'Occupancy_{model_name}_Rsq',
                              f'Occupancy_Affine_Correlation', f'Occupancy_{model_name}_Correlation',
                              f'Occupancy_Affine_RMSE', f'Occupancy_{model_name}_RMSE']]
    '''remove dataset name from columns'''
    Occupancy_results.columns = ['Training_Samples', 'Affine_Rsq', f'{model_name}_Rsq', 'Affine_Correlation',
                               f'{model_name}_Correlation', 'Affine_RMSE', f'{model_name}_RMSE']
    Occupancy_results['Dataset'] = ['Occupancy' for i in range(len(Training_Samples))]

    df_rsq = pd.concat([celebA_results,chemical_results,ETTm1_results,Occupancy_results])
    df_rsq = df_rsq[['Dataset','Training_Samples','Affine_Rsq', f'{model_name}_Rsq','Affine_Correlation',  f'{model_name}_Correlation','Affine_RMSE', f'{model_name}_RMSE']]

    print(len(df_rsq))
    df_rsq.to_csv(f'RESULTS/Initial_prediction/{ita_method_name}_{model_name}_Results_random_seed_{experiment}.csv',
                      index=False)