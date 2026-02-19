import pandas as pd
import copy
import numpy as np
import math
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Intel Math Kernel Library
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
import random
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import ast
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
import itertools
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# print(f'version = {tf.__version__}')

# USE_GPU = False
# if USE_GPU:
#     device_idx = 0
#     gpus = tf.config.list_physical_devices('GPU')
#     gpu_device = gpus[device_idx]
#     core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
#     tf.config.experimental.set_memory_growth(gpu_device, True)
#     tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Adapted from https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py
GATE_OP = 1

class PCGrad(tf.compat.v1.train.Optimizer):
  """PCGrad. https://arxiv.org/pdf/2001.06782.pdf."""

  def __init__(self, opt, use_locking=False, name="PCGrad"):
    """optimizer: the optimizer being wrapped."""
    super(PCGrad, self).__init__(use_locking, name)
    self.optimizer = opt

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    assert isinstance(loss, list)
    num_tasks = len(loss)
    loss = tf.stack(loss)
    tf.random.shuffle(loss)

    # Compute per-task gradients.
    grads_task = tf.vectorized_map(lambda x: tf.concat(
        [tf.reshape(grad, [-1, ]) for grad in tf.gradients(
            x, var_list) if grad is not None], axis=0), loss)
    # Debugging gradients
    for task_grad in grads_task:
        if tf.reduce_any(tf.math.is_nan(task_grad)):
            print(f"NaN in gradient for task. Gradient: {task_grad}")

    # Compute gradient projections.
    def proj_grad(grad_task):
      # print(f'we are going to project gradient')
      for k in range(num_tasks):
        inner_product = tf.reduce_sum(grad_task*grads_task[k])
        proj_direction = inner_product / tf.reduce_sum(
            grads_task[k]*grads_task[k])
        grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
      return grad_task

    proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

    # Unpack flattened projected gradients back to their original shapes.
    proj_grads = []
    for j in range(num_tasks):
      start_idx = 0
      empty_proj_grad_count = 0
      for idx, var in enumerate(var_list):
        grad_shape = var.get_shape()
        flatten_dim = np.prod(
            [grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
        proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]


        if tf.size(proj_grad) > 0:
            proj_grad = tf.reshape(proj_grad, grad_shape)
        else:
            # print(f"Warning: proj_grad is empty for shape {grad_shape}, skipping reshape.")
            empty_proj_grad_count+=1
            proj_grad = tf.zeros(grad_shape, dtype=tf.float32)  # Ensure dtype is correct

        if len(proj_grads) < len(var_list):
          proj_grads.append(proj_grad)
        else:
            proj_grads[idx] += proj_grad
        start_idx += flatten_dim
    # print(f'total empty_proj_grad_count: {empty_proj_grad_count}')
    grads_and_vars = list(zip(proj_grads, var_list))
    return grads_and_vars

  # def compute_gradients(self, loss_list, var_list=None,
  #                       gate_gradients=GATE_OP,
  #                       aggregation_method=None,
  #                       colocate_gradients_with_ops=False,
  #                       grad_loss=None):
  #     assert isinstance(loss_list, list)
  #     num_tasks = len(loss_list)
  #
  #     # Pre-compute all gradients safely
  #     grads_per_task = []
  #     for i, loss in enumerate(loss_list):
  #         grads = tf.gradients(loss, var_list)
  #         grads_flat = []
  #         for grad in grads:
  #             if grad is not None:
  #                 grads_flat.append(tf.reshape(grad, [-1]))
  #             else:
  #                 grads_flat.append(tf.zeros([tf.reduce_prod(tf.shape(var_list[grads.index(grad)]))]))
  #         grads_concat = tf.concat(grads_flat, axis=0)
  #         grads_per_task.append(grads_concat)
  #
  #     grads_task = tf.stack(grads_per_task)  # shape [num_tasks, total_param_dim]
  #
  #     # NaN Debug
  #     for i, g in enumerate(grads_per_task):
  #         tf.debugging.assert_all_finite(g, f"NaN in gradient for task {i}")
  #         # Compute gradient projections.
  #
  #     def proj_grad(grad_task):
  #         # print(f'we are going to project gradient')
  #         for k in range(num_tasks):
  #             inner_product = tf.reduce_sum(grad_task * grads_task[k])
  #             proj_direction = inner_product / tf.reduce_sum(
  #                 grads_task[k] * grads_task[k])
  #             grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
  #         return grad_task
  #
  #     proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)
  #
  #     # Unpack flattened projected gradients back to their original shapes.
  #     proj_grads = []
  #     for j in range(num_tasks):
  #         start_idx = 0
  #         empty_proj_grad_count = 0
  #         for idx, var in enumerate(var_list):
  #             grad_shape = var.get_shape()
  #             flatten_dim = np.prod(
  #                 [grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
  #             proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
  #
  #             if tf.size(proj_grad) > 0:
  #                 proj_grad = tf.reshape(proj_grad, grad_shape)
  #             else:
  #                 # print(f"Warning: proj_grad is empty for shape {grad_shape}, skipping reshape.")
  #                 empty_proj_grad_count += 1
  #                 proj_grad = tf.zeros(grad_shape, dtype=tf.float32)  # Ensure dtype is correct
  #
  #             if len(proj_grads) < len(var_list):
  #                 proj_grads.append(proj_grad)
  #             else:
  #                 proj_grads[idx] += proj_grad
  #             start_idx += flatten_dim
  #     # print(f'total empty_proj_grad_count: {empty_proj_grad_count}')
  #     grads_and_vars = list(zip(proj_grads, var_list))
  #     return grads_and_vars

  def _create_slots(self, var_list):
    self.optimizer._create_slots(var_list)

  def _prepare(self):
    self.optimizer._prepare()

  def _apply_dense(self, grad, var):
    return self.optimizer._apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self.optimizer._resource_apply_dense(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

  def _apply_sparse(self, grad, var):
    return self.optimizer._apply_sparse(grad, var)

  def _resource_scatter_add(self, x, i, v):
    return self.optimizer._resource_scatter_add(x, i, v)

  def _resource_apply_sparse(self, grad, var, indices):
    return self.optimizer._resource_apply_sparse(grad, var, indices)

  def _finish(self, update_ops, name_scope):
    return self.optimizer._finish(update_ops, name_scope)

  def _call_if_callable(self, param):
    """Call the function if param is callable."""
    return param() if callable(param) else param



def readData(molecule_list):
    data_param_dictionary = {}
    lengths = []
    for molecule in molecule_list:
        # csv = (f"{DataPath}DATA/{molecule}_Chemical_Data_for_MTL.csv")
        csv = (f"{DataPath}{molecule}_Molecule_Data.csv")
        chem_molecule_data = pd.read_csv(csv, low_memory=False)
        chem_molecule_data.loc[chem_molecule_data['181'] < 0, '181'] = 0

        DataSet = np.array(chem_molecule_data, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        Number_of_Features = np.shape(DataSet)[1]

        Input_Features = chem_molecule_data.columns[:Number_of_Features - 1]
        Target_Features = chem_molecule_data.columns[Number_of_Features - 1:]

        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]
        # print(Sample_Inputs[0])
        Sample_Label = np.zeros((Number_of_Records, len(Target_Features)))
        for t in range(Number_of_Records):
            Sample_Label[t] = DataSet[t, Number_of_Features - len(Target_Features):]

        Number_of_Features = len(Input_Features)
        data_param_dictionary.update({f'Molecule_{molecule}_FF_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'Molecule_{molecule}_Labels': Sample_Label})
        lengths.append(len(Sample_Inputs))

        '''*********************************'''

    return data_param_dictionary, Number_of_Features, lengths
def SplitLabels(Target):
    label_data = np.zeros((len(Target), 1))
    for t in range(len(Target)):
        label_data[t] = Target[t][0]
    return label_data

def repeat_Samples(samples_to_be_repeated, X_train, y_train):

    # Step 1: Separate indices by class
    class0_indices = np.where(y_train < 1)[0]
    class1_indices = np.where(y_train >= 1)[0]
    # print(f'TRAIN: [0,1] : [{len(class0_indices), len(class1_indices)}]')

    # Step 2: Compute how many samples to repeat from each class
    repeat_per_class = samples_to_be_repeated // 2

    # If odd, one sample will be left out — you can choose what to do with that if needed

    # Step 3: Randomly sample with replacement
    sampled_class0 = np.random.choice(class0_indices, repeat_per_class, replace=True)
    sampled_class1 = np.random.choice(class1_indices, repeat_per_class, replace=True)

    # Step 4: Combine and concatenate
    sampled_indices = np.concatenate([sampled_class0, sampled_class1])
    X_repeat = X_train[sampled_indices]
    y_repeat = y_train[sampled_indices]

    # Final step: Concatenate to training set
    X_train = np.concatenate((X_train, X_repeat), axis=0)
    y_train = np.concatenate((y_train, y_repeat), axis=0)

    return X_train, y_train


def data_preparation(tasks_list):
    data_param_dict_for_specific_task = {}
    DataPath = f'Dataset/{datasetName.upper()}/Task_Splits'

    lengths = []
    for task_id in tasks_list:
        X_train = np.load(f'{DataPath}/{task_id}_X_train.npy')
        lengths.append(X_train.shape[0])

    if len(tasks_list) > 1:
        max_size = max(lengths)
        if max_size % 2 == 0:
            max_size += 1

        print(f'max size = {max_size}')

    for task_id in tasks_list:
        X_train = np.load(f'{DataPath}/{task_id}_X_train.npy')
        y_train = np.load(f'{DataPath}/{task_id}_y_train.npy')
        X_test = np.load(f'{DataPath}/{task_id}_X_test.npy')
        y_test = np.load(f'{DataPath}/{task_id}_y_test.npy')

        if len(tasks_list) > 1:
            samples_to_be_repeated = max_size - len(X_train)

            if samples_to_be_repeated > 0:
                # print(f'X_train: {X_train.shape}, samples_to_be_repeated = {samples_to_be_repeated}')
                X_train, y_train = repeat_Samples(samples_to_be_repeated, X_train, y_train)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        # print(f'shape of y_train = {y_train.shape}')

        data_param_dict_for_specific_task.update({f'Molecule_{task_id}_X_train': X_train})
        data_param_dict_for_specific_task.update({f'Molecule_{task_id}_y_train': y_train})
        data_param_dict_for_specific_task.update({f'Molecule_{task_id}_X_test': X_test})
        data_param_dict_for_specific_task.update({f'Molecule_{task_id}_y_test': y_test})

    return data_param_dict_for_specific_task



    # print(len(args))



def decay_lr(step, optimizer):
    if (step + 1) % 75 == 0:
        optimizer.lr = optimizer.lr / 2.
        # print('Decreasing the learning rate by 1/2. New Learning Rate: {}'.format(optimizer.lr))

def permute_list_limit(lst, max_len=2):
    """Returns all combinations of tasks in the task list."""
    task_lst = [t for t in lst]
    print(f'task_lst = {task_lst}')
    # task_lst.sort()
    rtn = []
    for group_len in range(1, max_len + 1):
        for task in itertools.combinations(task_lst, group_len):
            task = list(task)
            # task.sort()
            task = "_".join(task)
            rtn.append(task)
    print(f'rtn = {rtn}')
    return rtn

# Global dictionary to hold uncertainty variables across tasks
global_uncertainty_registry = {}

def get_uncertainty_weights(TASKS):
    global global_uncertainty_registry

    uncertainty_weights = {}
    for task in TASKS:
        if task not in global_uncertainty_registry:
            # Create a new trainable tf.Variable for this task
            global_uncertainty_registry[task] = tf.Variable(1.0, trainable=True, name=f"uncertainty_{task}")
        uncertainty_weights[task] = global_uncertainty_registry[task]

    return uncertainty_weights


def final_model(shared_hyperparameters, molecule_list, data_param_dict_for_specific_task, val = False):
    print(f'molecule_list = {molecule_list}')
    print(f'EXECUTING for METHOD NAME : {Method_name}')

    Final_Name = Method_name
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []

    for task_id in molecule_list:
        X_train_full = data_param_dict_for_specific_task[f'Molecule_{task_id}_X_train']
        y_train_full = data_param_dict_for_specific_task[f'Molecule_{task_id}_y_train']

        # Add a small validation split from the training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
        )

        train_data.append(X_train)
        train_label.append(y_train)
        val_data.append(X_val)
        val_label.append(y_val)

        test_data.append(data_param_dict_for_specific_task[f'Molecule_{task_id}_X_test'])
        test_label.append(data_param_dict_for_specific_task[f'Molecule_{task_id}_y_test'])

    class SharedEncoder(tf.keras.Model):
        def __init__(self):
            super(SharedEncoder, self).__init__()
            self.shared_layers = []
            for h in range(shared_hyperparameters['shared_FF_Layers']):
                self.shared_layers.append(Dense(shared_hyperparameters['shared_FF_Neurons'][h], activation='relu'))

        def call(self, inputs):
            x = inputs
            for layer in self.shared_layers:
                x = layer(x)
            return x

    class TaskDecoder(tf.keras.Model):
        def __init__(self):
            super(TaskDecoder, self).__init__()
            self.output_layer = Dense(1, activation='sigmoid')

        def call(self, shared_representation):
            return self.output_layer(shared_representation)

    # Create instances of the Shared Encoder and Task Decoders
    shared_encoder = SharedEncoder()
    task_decoders = {molecule: TaskDecoder() for molecule in molecule_list}

    # from tensorflow.keras.utils import plot_model
    #
    # # Make sure a directory exists
    # os.makedirs("model_images", exist_ok=True)
    # # Example input dimension (change this to match your actual input)
    # input_dim = 128  # <-- replace with actual feature size
    # num_tasks = len(molecule_list)
    #
    # # Define shared encoder input
    # shared_input = Input(shape=(input_dim,), name='shared_input')
    # shared_output = shared_encoder(shared_input)
    #
    # # Collect all task outputs
    # task_outputs = []
    # for mol in molecule_list:
    #     task_output = task_decoders[mol](shared_output)
    #     task_outputs.append(task_output)
    #
    # # Create full multi-output model
    # multi_task_model = Model(inputs=shared_input, outputs=task_outputs, name="MultiTaskModel")
    #
    # # Plot full model structure
    # plot_model(
    #     multi_task_model,
    #     to_file='model_images/full_model_structure_new.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     expand_nested=True,
    #     dpi=96
    # )
    #
    # exit(0)
    global_step = tf.Variable(0, trainable=False)
    init_lr = shared_hyperparameters['learning_rate']
    optimizer = tf.keras.optimizers.SGD(init_lr, momentum=0.9, nesterov=False)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    print(f'optimizer variables = {len(optimizer.variables())},')
    print(f'how many trainable variables = {len(shared_encoder.trainable_variables)}')

    if 'pcgrad' in Final_Name:
        lr_var = tf.Variable(shared_hyperparameters['learning_rate'])
        old_optimizer = tf.compat.v1.train.MomentumOptimizer(lr_var, momentum=0.9)
        optimizer = PCGrad(tf.compat.v1.train.MomentumOptimizer(lr_var, momentum=0.9))

    @tf.function
    def train_step(x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            shared_representations = [shared_encoder(input_data, training=True) for input_data in x_batch_train]
            predictions = [task_decoders[molecule](shared_rep, training=True) for molecule, shared_rep in
                           zip(molecule_list, shared_representations)]
            losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
            losses_dict = {task: loss for task, loss in zip(molecule_list, losses)}
            tot_loss = tf.reduce_sum(losses)

        gradients = tape.gradient(tot_loss, shared_encoder.trainable_variables + sum(
            [decoder.trainable_variables for decoder in task_decoders.values()], []))

        optimizer.apply_gradients(zip(gradients, shared_encoder.trainable_variables + sum(
            [decoder.trainable_variables for decoder in task_decoders.values()], [])))

        original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        original_decoder_weights = {molecule: [tf.identity(weight) for weight in decoder.trainable_weights]
                                    for molecule, decoder in task_decoders.items()}


        return tot_loss, original_shared_weights, original_decoder_weights

    @tf.function
    def test_step(x_batch_test, y_batch_test):
        shared_representations = [shared_encoder(input_data, training=False) for input_data in x_batch_test]
        predictions = [task_decoders[molecule](shared_rep, training=False) for molecule, shared_rep in
                       zip(molecule_list, shared_representations)]
        eval_losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_test, predictions)]
        eval_loss = tf.reduce_sum(eval_losses)
        return eval_loss, eval_losses, predictions

    @tf.function
    def train_pcgrad_step(x_batch_train, y_batch_train):

        shared_representations = [shared_encoder(input_data, training=True) for input_data in x_batch_train]
        predictions = [task_decoders[molecule](shared_rep, training=True) for molecule, shared_rep in
                       zip(molecule_list, shared_representations)]
        losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
        losses_dict = {task: loss for task, loss in zip(molecule_list, losses)}
        # tot_loss = tf.reduce_sum(losses)

        '''change'''
        # uncertainty_weights = get_uncertainty_weights(molecule_list)
        # # for task in molecule_list:
        # #     clip_uncertainty = tf.clip_by_value(uncertainty_weights[task], 0.01, 10.0)
        # #     losses[task] = losses[task] / tf.exp(2 * clip_uncertainty) + clip_uncertainty
        # for task in molecule_list:
        #     clip_uncertainty = tf.clip_by_value(uncertainty_weights[task], 0.01, 10.0)
        #     losses_dict[task] = losses_dict[task] / tf.exp(2 * clip_uncertainty) + clip_uncertainty

        # Total loss
        loss = tf.add_n(list(losses_dict.values()))

        # Ensure variable types are compatible
        for var in shared_encoder.trainable_weights:
            var.assign(tf.cast(var, tf.float32))  # Ensure the variable type is tf.float32

        # Compute gradients for ResBase
        base_gradvars = optimizer.compute_gradients(losses, shared_encoder.trainable_weights)
        base_gradvars = [(tf.cast(grad, tf.float32), var) for grad, var in base_gradvars]

        # Apply the gradients for ResBase
        old_optimizer.apply_gradients(base_gradvars)

        # Compute gradients for ResTowers tasks
        task_gradvars = [optimizer.compute_gradients([losses_dict[task]], decoder.trainable_weights) for task, decoder
                         in
                         task_decoders.items()]
        for gv in task_gradvars:
            old_optimizer.apply_gradients(gv)

        # print(f'losses = {losses_dict}')
        # before_update_losses = copy.deepcopy(losses_dict)
        # before_losses = {task: loss.numpy() for task, loss in before_update_losses.items()}
        # print(f'before_losses = {before_losses}')
        # # print(f'total loss = {tot_loss.numpy()}')
        # gradients = tape.gradient(tot_loss, shared_encoder.trainable_variables + sum(
        #     [decoder.trainable_variables for decoder in task_decoders.values()], []))
        #
        # optimizer.apply_gradients(zip(gradients, shared_encoder.trainable_variables + sum(
        #     [decoder.trainable_variables for decoder in task_decoders.values()], [])))

        # Update the uncertainty weight variables.
        # uw_gv = old_optimizer.compute_gradients(loss, list(uncertainty_weights.values()))
        # old_optimizer.apply_gradients(uw_gv)


        original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        original_decoder_weights = {molecule: [tf.identity(weight) for weight in decoder.trainable_weights]
                                    for molecule, decoder in task_decoders.items()}
        return loss, original_shared_weights, original_decoder_weights

    # @tf.function
    # def train_pcgrad_step(x_batch_train, y_batch_train):
    #     shared_representations = [shared_encoder(input_data, training=True) for input_data in x_batch_train]
    #     predictions = [task_decoders[mol](shared_rep, training=True) for mol, shared_rep in
    #                    zip(molecule_list, shared_representations)]
    #     losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
    #     losses_dict = {task: loss for task, loss in zip(molecule_list, losses)}
    #
    #     loss = tf.add_n(list(losses_dict.values()))
    #
    #     # Equal weighting
    #     equal_weighted_loss = tf.add_n(list(losses_dict.values())) / tf.cast(len(losses_dict), tf.float32)
    #
    #     # Sanity check: NaNs
    #     tf.debugging.check_numerics(equal_weighted_loss, message="Loss contains NaN")
    #
    #
    #     # Compute gradients for shared encoder
    #     base_gradvars = optimizer.compute_gradients(losses, shared_encoder.trainable_weights)
    #     base_gradvars = [(tf.cast(grad, tf.float32), var) for grad, var in base_gradvars]
    #     old_optimizer.apply_gradients(base_gradvars)
    #
    #     # Compute gradients for task decoders
    #     for task, decoder in task_decoders.items():
    #         task_loss = losses_dict[task]
    #         task_gradvars = optimizer.compute_gradients([task_loss], decoder.trainable_weights)
    #         old_optimizer.apply_gradients(task_gradvars)
    #
    #     # Update uncertainty weights (if applicable)
    #     # equal_weighted_loss = tf.add_n(losses) / len(losses)
    #     # uw_gv = old_optimizer.compute_gradients(loss)
    #     # old_optimizer.apply_gradients(uw_gv)
    #
    #     all_vars = shared_encoder.trainable_weights
    #     for decoder in task_decoders.values():
    #         all_vars += decoder.trainable_weights
    #
    #     uw_gv = old_optimizer.compute_gradients(equal_weighted_loss, var_list=all_vars)
    #     old_optimizer.apply_gradients(uw_gv)
    #
    #     # Save weights
    #     original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
    #     original_decoder_weights = {mol: [tf.identity(w) for w in dec.trainable_weights] for mol, dec in
    #                                 task_decoders.items()}
    #     return equal_weighted_loss, original_shared_weights, original_decoder_weights

    @tf.function
    def train_step_ITA(x_batch_train, y_batch_train, first_step=False):  # per-batch calculation
        task_gains = {task: {task: {} for task in TASKS}
                      for task in TASKS}

        '''can't see the output since it's a tf.function'''
        # print(f'first_step = {first_step} at epoch, batch_idx = {epoch, batch_idx}, {len(optimizer.variables())}')

        with tf.GradientTape(persistent=True) as tape:
            shared_representations = [shared_encoder(input_data, training=True) for input_data in x_batch_train]
            predictions = [task_decoders[molecule](shared_rep, training=True) for molecule, shared_rep in
                           zip(molecule_list, shared_representations)]
            losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
            losses_dict = {task: loss for task, loss in zip(TASKS, losses)}
            tot_loss = tf.reduce_sum(losses)

            single_task_specific_gradients = [
                (single_task, tape.gradient(losses_dict[single_task], shared_encoder.trainable_weights)) for
                single_task in TASKS]

        # Compute for regular model update
        all_tasks_gradients = [tf.add_n([task_gradient[i] for _, task_gradient in single_task_specific_gradients])
                               for i in range(len(shared_encoder.trainable_weights))]

        before_update_losses = {task: loss for task, loss in losses_dict.items()}
        original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        original_decoder_weights = {molecule: [tf.identity(weight) for weight in decoder.trainable_weights]
                                    for molecule, decoder in task_decoders.items()}

        # print(f'BEFORE AFFINITY_Calc: first parameter of shared_encoder = {shared_encoder.trainable_weights[0][0][:5]}')
        # print(f'before_update_losses = {before_update_losses}')

        for base_task, task_grad in single_task_specific_gradients:
            if first_step:
                # Regular update for the first step
                base_update = [optimizer.lr * grad for grad in task_grad]
                base_updated = [param - update for param, update in zip(shared_encoder.trainable_weights, base_update)]
            else:
                # Momentum-based update for later steps
                base_update = [(optimizer._momentum * optimizer.get_slot(param, 'momentum') - optimizer.lr * grad)
                               for param, grad in zip(shared_encoder.trainable_weights, task_grad)]
                base_updated = [param + update for param, update in zip(shared_encoder.trainable_weights, base_update)]

            # Recompute representation and losses using updated base (base_updated)

            # Temporarily update shared encoder weights with base_updated for AFFINITY_Calc computation
            for original_param, updated_param in zip(shared_encoder.trainable_weights, base_updated):
                original_param.assign(updated_param)

            shared_representations = [shared_encoder(input_data) for input_data in x_batch_train]
            predictions = [task_decoders[molecule](shared_rep, training=True) for molecule, shared_rep in
                           zip(molecule_list, shared_representations)]

            after_update_losses_list = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
            after_update_losses = {task: loss for task, loss in zip(TASKS, after_update_losses_list)}

            '''Compute task gain'''
            task_gain = {
                second_task: (1.0 - after_update_losses[second_task] / before_update_losses[second_task]) / optimizer.lr
                for second_task in TASKS}
            task_gains[base_task] = task_gain

            # Revert shared encoder weights back to the original parameters after AFFINITY_Calc computation
            for original_param, updated_param in zip(shared_encoder.trainable_weights, original_shared_weights):
                original_param.assign(updated_param)

        '''revert back the model to the previous state'''
        for original_weight, weight in zip(original_shared_weights, shared_encoder.trainable_weights):
            weight.assign(original_weight)

        for molecule, original_weights in original_decoder_weights.items():
            for original_weight, weight in zip(original_weights, task_decoders[molecule].trainable_weights):
                weight.assign(original_weight)

        '''apply regular model updates'''

        for task, decoder in task_decoders.items():
            task_grads = tape.gradient(losses_dict[task], decoder.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, decoder.trainable_weights))
        # print('update')
        # all_grads = tape.gradient(tot_loss, shared_encoder.trainable_weights)
        optimizer.apply_gradients(zip(all_tasks_gradients, shared_encoder.trainable_weights))

        '''save the original weights'''
        original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        original_decoder_weights = {molecule: [tf.identity(weight) for weight in decoder.trainable_weights]
                                    for molecule, decoder in task_decoders.items()}
        return tot_loss, task_gains, original_shared_weights, original_decoder_weights

    @tf.function
    def train_step_ETAP_Affinity(x_batch_train, y_batch_train, first_step=False):
        task_gains = {task: {task: {} for task in TASKS} for task in TASKS}

        with tf.GradientTape(persistent=True) as tape:
            shared_representations = [shared_encoder(input_data, training=True) for input_data in x_batch_train]
            predictions = [task_decoders[molecule](shared_rep, training=True) for molecule, shared_rep in
                           zip(molecule_list, shared_representations)]
            losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
            losses_dict = {task: loss for task, loss in zip(TASKS, losses)}
            # loss_val = {task: loss for task, loss in losses_dict.items()}
            tot_loss = tf.reduce_sum(losses)

            single_task_specific_gradients = [(single_task, tape.gradient(losses_dict[single_task], shared_encoder.trainable_weights)) for
                single_task in TASKS]

        # Compute for regular model update
        all_tasks_gradients = [tf.add_n([task_gradient[i] for _, task_gradient in single_task_specific_gradients])
                               for i in range(len(shared_encoder.trainable_weights))]

        # original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        # original_decoder_weights = {molecule: [tf.identity(weight) for weight in decoder.trainable_weights]
        #                             for molecule, decoder in task_decoders.items()}

        '''ITA Approximation-Calculation'''
        # get task-specific updates
        if w_momentum:
            task_gradient_updates = {}
            for single_task, task_gradient in single_task_specific_gradients:
                if first_step:
                    base_update = [optimizer.lr * grad for grad in task_gradient]
                    # base_updated = [param - update for param, update in zip(ResBase.trainable_weights, base_update)]
                else:
                    base_update = [
                        (optimizer.lr * grad - optimizer._momentum * optimizer.get_slot(param, 'momentum'))
                        for param, grad in zip(shared_encoder.trainable_weights, task_gradient)]

                task_gradient_updates[single_task] = base_update
        else:
            task_gradient_updates = {}
            for single_task, task_gradient in single_task_specific_gradients:
                base_update = [optimizer.lr * grad for grad in task_gradient]
                task_gradient_updates[single_task] = base_update

        # flatten and concatenate gradients for all tasks to get Jacobian matrix
        reshaped_gradients = [tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
                              for _, grads in single_task_specific_gradients]

        # Flatten and concatenate updates for all tasks to get update matrix
        reshaped_updates = [tf.concat([tf.reshape(update, [-1]) for update in updates], axis=0)
                            for _, updates in task_gradient_updates.items()]

        '''perform matrix multiplication for ITA approximation'''
        G = tf.convert_to_tensor(reshaped_gradients)  # Jacobian matrix - all gradients
        U = tf.convert_to_tensor(reshaped_updates)  # Update matrix -- all updates

        L = [losses_dict[task] for task in TASKS]  # Loss matrix
        L = tf.reshape(tf.convert_to_tensor(L), (-1, 1))

        ita_approximation_G_U = tf.matmul(G, U, transpose_b=True)
        # '''wo loss'''
        ita_approximation = tf.divide(ita_approximation_G_U, L)
        ita_approximation = tf.divide(ita_approximation, optimizer.lr)

        for idx, base_task in enumerate(TASKS):
            # Extract the ith column from ita_approximation of base task onto other tasks
            ita_per_task = ita_approximation[:, idx]
            task_gains[base_task] = {second_task: ita_per_task[TASKS.index(second_task)] for second_task in TASKS}

        # '''revert back the model to the previous state'''
        # for original_weight, weight in zip(original_shared_weights, shared_encoder.trainable_weights):
        #     weight.assign(original_weight)
        #
        # for molecule, original_weights in original_decoder_weights.items():
        #     for original_weight, weight in zip(original_weights, task_decoders[molecule].trainable_weights):
        #         weight.assign(original_weight)

        '''apply regular model updates'''

        for task, decoder in task_decoders.items():
            task_grads = tape.gradient(losses_dict[task], decoder.trainable_weights)
            optimizer.apply_gradients(zip(task_grads, decoder.trainable_weights))
        # print('update')
        # all_grads = tape.gradient(tot_loss, shared_encoder.trainable_weights)
        optimizer.apply_gradients(zip(all_tasks_gradients, shared_encoder.trainable_weights))

        '''save the original weights'''
        original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        original_decoder_weights = {molecule: [tf.identity(weight) for weight in decoder.trainable_weights]
                                    for molecule, decoder in task_decoders.items()}

        return tot_loss, task_gains, original_shared_weights, original_decoder_weights


    # @tf.function()
    def save_gradients():
        # Set the model to evaluation mode in TensorFlow
        # (In TF, there's no need for an explicit eval mode like PyTorch, just ensure dropout/batchnorm layers are in inference mode)
        task_gradients = {task: [] for task in molecule_list}

        for batch_idx in range(0, len(train_data[0]), batch_size):
            x_batch_train = [data[batch_idx:batch_idx + batch_size] for data in train_data]
            y_batch_train = [label[batch_idx:batch_idx + batch_size] for label in train_label]
            # optimizer = new_optimizer
            with tf.GradientTape(persistent=True) as tape:
                shared_representations = [shared_encoder(input_data, training = True) for input_data in x_batch_train]
                predictions = [task_decoders[molecule](shared_rep, training = True) for molecule, shared_rep in
                               zip(molecule_list, shared_representations)]
                losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch_train, predictions)]
                losses_dict = {task: loss for task, loss in zip(molecule_list, losses)}
                tot_loss = tf.reduce_sum(losses)

                # Compute the gradient of the task-specific loss w.r.t. the shared base.
                single_task_specific_gradients = [
                    (single_task, tape.gradient(losses_dict[single_task], shared_encoder.trainable_weights)) for
                    single_task in molecule_list]

            for task, tmp_gradients in single_task_specific_gradients:
                # for tmp_gradients in grads:
                '''flatten and concatenate gradients'''
                tmp_gradients = tf.concat([tf.reshape(g, [-1]) for g in tmp_gradients], axis=0).numpy()

                if tmp_gradients.size != project_matrix.shape[0]:
                    raise ValueError(
                        f"Gradient size {tmp_gradients.size} does not match expected size {project_matrix.shape[0]}")

                tmp_gradients = (tmp_gradients.reshape(1, -1) @ project_matrix).flatten()

                task_gradients[task].append(tmp_gradients)

        for task_name, gradients in task_gradients.items():
            np.save(f"{gradients_dir}/{task_name}_train_gradients.npy", gradients)

        del tape  # Clean up the persistent GradientTape

    Patience = 20
    min_loss_to_consider = math.inf

    TRAIN_SIZE = len(train_data[0])
    print(f'TRAIN_SIZE = {TRAIN_SIZE}')

    if Final_Name == 'ETAP_Affinity_Groups':
        gradient_metrics = {task: [] for task in permute_list_limit(molecule_list)}
    else:
        gradient_metrics = {task: [] for task in molecule_list}

    timeStart = time.time()
    velocity_trackers = {}
    for epoch in range(num_epochs):
        if epoch > 100:
            if 'pcgrad_mtl' in Method_name:
                if (epoch + 1) % 75 == 0:
                    lr_var = lr_var / 2
                old_optimizer = tf.compat.v1.train.MomentumOptimizer(lr_var, momentum=0.9)
                optimizer = PCGrad(tf.compat.v1.train.MomentumOptimizer(lr_var, momentum=0.9))
            else:
                decay_lr(epoch, optimizer)

        batch_grad_metrics = {combined_task: {task: 0. for task in molecule_list} for combined_task in
                              gradient_metrics}

        for batch_idx in range(0, len(train_data[0]), batch_size):
            x_batch_train = [data[batch_idx:batch_idx + batch_size] for data in train_data]
            y_batch_train = [label[batch_idx:batch_idx + batch_size] for label in train_label]


            if AFFINITY_Calc:
                if Final_Name == 'ITA':
                    train_loss, task_gains, shared_weights, decoder_weights = train_step_ITA(x_batch_train,
                                                                                             y_batch_train, first_step=(len(optimizer.variables()) == 0))
                    # print(f"batch_idx = {batch_idx}\tLoss: {train_loss.numpy()}")
                elif Final_Name == 'ETAP_Affinity':
                    train_loss, task_gains, shared_weights, decoder_weights = train_step_ETAP_Affinity(x_batch_train,
                                                                                                    y_batch_train,
                                                                                                    first_step=(
                                                                                                                len(optimizer.variables()) == 0))

                # Record batch-level training and gradient metrics.
                for first_task, task_gain_map in task_gains.items():
                    for second_task, gain in task_gain_map.items():
                        # print(f'first_task = {first_task}\tsecond_task = {second_task}\tgain = {gain}')
                        batch_grad_metrics[first_task][second_task] += gain.numpy() / (
                            math.ceil(TRAIN_SIZE / batch_size))
                        # print(f'first_task = {first_task}\tsecond_task = {second_task}\tgain = {gain.numpy()}, batch_grad_metrics = {batch_grad_metrics[first_task][second_task]}')

                # print(f'batch_grad_metrics = {batch_grad_metrics}')

                # exit(0)
            else:
                if 'pcgrad_mtl' in Final_Name:
                    train_loss, shared_weights, decoder_weights = train_pcgrad_step(x_batch_train, y_batch_train)
                if Final_Name == 'SimpleMTL':
                    train_loss, shared_weights, decoder_weights = train_step(x_batch_train, y_batch_train)
                # print(f"batch_idx = {batch_idx}\tLoss: {loss.numpy()}")

            # print(f"Loss: {loss.numpy()}")

        # print(f'One epoch done')
        # for source_task, task_gain_map in batch_grad_metrics.items():
        #     print(f'source_task = {source_task}\ttask_gain_map = {task_gain_map}')
        # exit(0)
        if val:
            ### Validation can be done here if needed, by evaluating on `val_data` and `val_label`
            val_loss, _, _ = test_step(val_data, val_label)
            loss_to_consider = val_loss
        else:
            loss_to_consider = train_loss

        if epoch % 20 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, loss = {train_loss.numpy()}, Patience = {Patience}')

        if loss_to_consider.numpy() < min_loss_to_consider:
            min_loss_to_consider = min(min_loss_to_consider, loss_to_consider.numpy())
            Patience = 20
            best_shared_weights = copy.deepcopy(shared_weights)
            best_decoder_weights = copy.deepcopy(decoder_weights)
        else:
            Patience -= 1
            if Patience == 0:
                print(f'Stopping Training at Epoch {epoch + 1}')
                break
        # if epoch % 20 == 0:
        #     for base_task, task_gain_map in batch_grad_metrics.items():
        #         print(f'base_task = {base_task}\tgain = {task_gain_map}')
        # exit(0)

        if AFFINITY_Calc:
            # print(f'\n\n***batch_grad_metrics epoch = {epoch}')
            # for source_task, task_gain_map in batch_grad_metrics.items():
            #     print(f'source_task = {source_task}\ttask_gain_map = {len(task_gain_map)}, task_gain_map = {task_gain_map}')
            # Record epoch-level training and gradient metrics.
            for combined_task, task_gain_map in batch_grad_metrics.items():
                gradient_metrics[combined_task].append(task_gain_map)

            # time_taken = time.time() - timeStart
            # print(f'Total time taken for epoch {epoch + 1} = {time.time() - timeStart}')

            if epoch % 100 == 0:
                print(f'epoch {epoch}, gradient_metrics = {len(gradient_metrics)}')


    time_taken = time.time() - timeStart
    # print(f'gradient_metrics = {gradient_metrics}')
    # load the original model
    for best_weight, curr_weight in zip(best_shared_weights, shared_encoder.trainable_weights):
        curr_weight.assign(best_weight)

    for molecule, decoder_specific_weights in best_decoder_weights.items():
        for best_weight, curr_weight in zip(decoder_specific_weights, task_decoders[molecule].trainable_weights):
            curr_weight.assign(best_weight)

    '''new parts'''
    if len(molecule_list)==len(TASKS):
        '''save best weights and model to a file'''
        model_base_dir = f'{datasetName}_model_weights'
        if not os.path.exists(model_base_dir):
            os.makedirs(model_base_dir)


        gradients_dir = f'{datasetName}_gradients_run_{run}'
        if not os.path.exists(gradients_dir):
            os.makedirs(gradients_dir)

        model_dir = f'{model_base_dir}/run_{run}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        shared_encoder.save_weights(f'{model_dir}/shared_encoder')
        for molecule, decoder in task_decoders.items():
            decoder.save_weights(f'{model_dir}/decoder_{molecule}')

        grad_params = []
        for params in shared_encoder.trainable_weights:
            grad_params.append(params)

        print(f'len(grad_params): {len(grad_params)}', end=' ')
        gradient_dim = 0
        for param in grad_params:
            gradient_dim += param.numpy().size
        print("Gradient Dim: {}".format(gradient_dim), end=' ')

        project_dim = 200
        project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_dim)) - 1).astype(float)
        project_matrix *= 1 / np.sqrt(project_dim)
        print("Project Dim: {}".format(project_dim))

        # Save gradients
        start_time_grad = time.time()
        save_gradients()
        end_time_grad = time.time()
        # print(f"Time taken for train gradients: {end_time_grad - start_time_grad}")

    test_loss, indiv_losses, y_pred = test_step(test_data, test_label)
    indiv_losses = [each_loss.numpy() for each_loss in indiv_losses]
    y_pred = [pred.numpy() for pred in y_pred]
    y_test = [label for label in test_label]
    y_pred = np.concatenate(y_pred, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    # print(f'y_pred = {y_pred[:50]}, y_test = {y_test[:50]}')

    predicted_val = (y_pred >= 0.75).astype(int)
    error_rate = np.mean(predicted_val != y_test)
    ap = average_precision_score(y_test, y_pred)

    print(f'test_loss = {test_loss}\terrorRate = {error_rate}\tap = {ap}')
    if AFFINITY_Calc:
        if 'Approx' in Final_Name:
            if w_momentum:
                ita_file = f'{ResultPath}/ITA/gradient_metrics_{Final_Name}_run_{run}_w_momentum_Arch_{Arch_Name}_{name_suffix}.csv'
            else:
                ita_file = f'{ResultPath}/ITA/gradient_metrics_{Final_Name}_run_{run}_wo_momentum_Arch_{Arch_Name}_{name_suffix}.csv'
        else:
            ita_file = f'{ResultPath}/ITA/gradient_metrics_{Final_Name}_run_{run}_Arch_{Arch_Name}_{name_suffix}.csv'

        with open(ita_file, 'w') as f:
            for key in gradient_metrics.keys():
                f.write("%s,%s\n" % (key, gradient_metrics[key]))
        f.close()

    # if os.path.exists(filepath + '_encoder'):
    #     os.remove(filepath + '_encoder')
    # for molecule in molecule_list:
    #     if os.path.exists(filepath + f'_decoder_{molecule}'):
    #         os.remove(filepath + f'_decoder_{molecule}')

    return test_loss.numpy(), indiv_losses, error_rate, time_taken


if __name__ == "__main__":
    datasetName = 'Chemical'
    DataPath = f'../Dataset/{datasetName.upper()}/'
    import sys

    w_momentum = False
    AFFINITY_Calc = 0
    if AFFINITY_Calc:
        Method_name = 'ETAP_Affinity'  # sys.argv[1] [Input: 'ITA' for TAG's ITA Score, 'ETAP_Affinity' for our Task-Affinity Score]
        group_type = sys.argv[1]
    else:
        Method_name = 'SimpleMTL'
        # Method_name = 'pcgrad_mtl_EW'
        group_type = sys.argv[1] #'G3' for groups of 3, 'G4' for groups of 4, 'ALL' for baseline MTL
        # group_type = 'ALL'
        if len(sys.argv) > 2:
            part = sys.argv[2]


    ResultPath = f'chem_results_new/'

    import sys

    TASKS = [83, 78, 84, 85, 76, 86, 81, 80, 87, 55]
    print(f'TASKS = {TASKS}')
    TASKS = [str(task) for task in TASKS]

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    STL_error = {}
    STL_AP = {}

    num_folds = 10

    Arch_Name = 'Arch_1'
    if Arch_Name == 'Arch_1':
        initial_shared_architecture = {'shared_FF_Layers': 2, 'shared_FF_Neurons': [32, 16],
                                       'learning_rate': 0.001}
        num_epochs = 1000
        batch_size = 264


    RUNS = [0,1,2,3,4,5,6,7,8,9,10]
    if AFFINITY_Calc:
        TASK_Group = [tuple(TASKS)]

    '''Pairs'''
    if group_type == 'PTL':
        pairs = list(itertools.combinations(TASKS, 2))
        print(f'pairs = {pairs}')
        TASK_Group = pairs
        name_suffix = 'pairs'

    '''STL'''
    if group_type == 'STL':
        Tasks_tuples = [tuple([task]) for task in TASKS]
        TASK_Group = Tasks_tuples
        name_suffix = 'STL'

    if group_type == 'GroundTruth':
        random_subsets = pd.read_csv(f'../RESULTS/{datasetName}_Random_Subsets_for_GroundTruth.csv', low_memory=False)
        print(len(random_subsets))
        TASK_Group = list(random_subsets['Random_Subsets'])
        TASK_Group = [ast.literal_eval(grp) for grp in TASK_Group]
        print(f'Total Groups : {len(TASK_Group)}')

        name_suffix = 'GroundTruth'

        print(f'part : {part}, total Groups : {len(TASK_Group)}')

    '''ALL'''
    if group_type == 'ALL':
        TASK_Group = [tuple(TASKS)]
        name_suffix = f'{Method_name}_ALL'
    if 'G' in group_type:
        len_group = int(group_type[-1:])
        TASK_Group = list(itertools.combinations(TASKS, len_group))
        name_suffix = f'{Method_name}_G{len_group}'


    name_suffix = 'FIXED_' + name_suffix
    if Method_name == 'pcgrad_mtl':
        name_suffix = f'{name_suffix}_EW'
    SEED = [2025, 2024, 2023, 2022, 2021, 2020, 2019]
    print(f'SEED = {SEED}')

    for run in RUNS:
        seed_value = SEED[run-1]
        random_seed = seed_value
        print(f'seed_value = {seed_value}')
        tf.random.set_seed(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)

        Task_group = []
        Total_Loss = []
        Individual_Group_Score = []
        Individual_Error_Rate = []
        Individual_AP = []
        Number_of_Groups = []
        Individual_Task_Score = []
        Prev_Groups = {}
        for count in range(len(TASK_Group)):
            print(f'Initial Training for {datasetName}-partition {count}, {TASK_Group[count]}')
            task_group = TASK_Group[count]
            args_tasks = []
            group_score = {}
            group_avg_err = {}
            group_avg_AP = {}
            tmp_task_score = []


            data_param_dict_for_specific_task = data_preparation(task_group)
            all_scores = final_model(initial_shared_architecture, task_group, data_param_dict_for_specific_task, val = True)
            print(all_scores)
            tot_loss, indi_scores = all_scores[0], all_scores[1]
            avg_error = all_scores[2]
            Total_time =  all_scores[-1]
            task_scores = {}
            for idx,task in enumerate(task_group):
                task_scores[f'molecule_{task}'] = indi_scores[idx]


            print(f'total_time = {Total_time}')
            print(f'avg time in minutes = {np.mean(Total_time) / 60}')

            # loss, task_scores, avg_error, AP = sort_Res(all_scores, task)


            print(f'tot_loss = {tot_loss}')
            # print(f'group_score = {group_score}')

            # exit(0)
            Task_group.append(task_group)
            # Number_of_Groups.append(len(task_group))
            Total_Loss.append(tot_loss)
            # Individual_Group_Score.append(group_score.copy())
            Individual_Error_Rate.append(avg_error)
            # Individual_AP.append(group_avg_AP.copy())
            Individual_Task_Score.append(copy.deepcopy(task_scores))
            # print(Individual_Group_Score)

            print(len(Total_Loss), len(Task_group), len(Individual_Task_Score), len(Individual_Error_Rate))
            # exit(0)

            temp_res = pd.DataFrame({'Total_Loss': Total_Loss,
                                     # 'Number_of_Groups': Number_of_Groups,
                                     'Task_group': Task_group,
                                     'Individual_Task_Score': Individual_Task_Score,
                                     # 'Individual_Group_Score': Individual_Group_Score,
                                     'Individual_Error_Rate': Individual_Error_Rate,
                                     # 'Individual_AP': Individual_AP
                                     })
            # run = seed_value
            if AFFINITY_Calc:
                temp_res.to_csv(f'{ResultPath}/{datasetName}_SimpleMTL_{Method_name}_run_{run}_SGD_Arch_{Arch_Name}_{name_suffix}.csv',
                                    index=False)

                print(f'total_time = {Total_time}')
                print(f'avg time in minutes = {np.mean(Total_time) / 60}')

                '''save time to txt file'''
                if w_momentum:
                    timefile = f'{ResultPath}/{datasetName}_{Method_name}_w_momentum_time_run_{run}_SGD_Arch_{Arch_Name}_{name_suffix}.txt'
                else:
                    timefile = f'{ResultPath}/{datasetName}_{Method_name}_time_run_{run}_SGD_Arch_{Arch_Name}_{name_suffix}.txt'
                with open(timefile, 'a') as f:
                    f.write(f'{np.mean(Total_time) / 60}\n')
                    f.write(f'{Total_time}\n')

                f.close()
            else:
                temp_res.to_csv(f'{ResultPath}/{datasetName}_{name_suffix}_run_{run}_SGD_Arch_{Arch_Name}.csv', index=False)

