import ast
import os

import pandas as pd
import copy
import numpy as np
import math
import os
import time
pd.set_option('display.max_columns', None)
'''use only GPU 1'''
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["OMP_NUM_THREADS"] = "4"  # Adjust based on CPU
os.environ["MKL_NUM_THREADS"] = "4"

import random
from sklearn.metrics import r2_score
from tensorflow.keras import Model
from tensorflow.keras.layers import (LayerNormalization,Embedding, Flatten,BatchNormalization)
import sys

sys.path.insert(0, '..')
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate

K.clear_session()


def prepare_data_sequences(df, past=10, target_col='load', seq_features=None, flat_features=None):
    """
    Prepare hybrid sequence data with (X_seq, X_flat), y for MTL or STL.

    Parameters:
    - df: input DataFrame sorted by stop_sequence or time
    - past: number of past timesteps to include in sequence
    - target_col: target label column name
    - seq_features: list of columns treated as sequential input
    - flat_features: list of columns treated as non-sequential (static per sample)

    Returns:
    - X_seq: np.array of shape (samples, past, seq_features)
    - X_flat: np.array of shape (samples, flat_features)
    - y: target labels of shape (samples,)
    """
    if seq_features is None:
        seq_features = [
            'stop_sequence',
            'darksky_temperature',
            'darksky_humidity',
            'darksky_precipitation_probability',
            'sched_hdwy'
        ]
    if flat_features is None:
        flat_features = [

            'time_window', 'month', 'day', 'hour',
            'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'dayofweek_7',
            # 'is_holiday_False',
            'is_holiday_True',
            # 'is_school_break_False',
            'is_school_break_True'
        ]

    X_seq, X_flat, y = [], [], []

    values = df[seq_features + flat_features + [target_col]].values
    for i in range(past, len(values)):
        seq_part = values[i - past:i, :len(seq_features)]
        flat_part = values[i, len(seq_features):len(seq_features) + len(flat_features)]
        label = values[i, -1]

        X_seq.append(seq_part)
        X_flat.append(flat_part)
        y.append(label)

    return np.array(X_seq), np.array(X_flat), np.array(y)

# --- Final Dataset Preparation for Model Feeding ---
def build_model_input_dataset(df, target_col='load', past=10):
    """
    Prepare final model-ready dataset with structured sequence and flat input tensors.
    Assumes previous trip features are included in df.

    Returns:
    - X_seq: np.array of shape (N, past, len(seq_features))
    - X_flat: np.array of shape (N, len(flat_features))
    - embed_inputs: Dict[str, np.array] for each embedding feature
    - y: np.array of shape (N,)
    """

    ALL_COLS = ['load', 'route_id_dir', 'stop_id_original', 'time_window', 'block_abbr',
       'year', 'month', 'day', 'hour', 'stop_sequence', 'darksky_temperature',
       'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy',
       'delay', 'Previous_Trip_1_Max_Occupancy',
       'Previous_Trip_1_Median_Occupancy', 'Previous_Trip_1_TimeDifference',
       'Previous_Trip_2_Max_Occupancy', 'Previous_Trip_2_Median_Occupancy',
       'Previous_Trip_2_TimeDifference', 'Previous_Trip_3_Max_Occupancy',
       'Previous_Trip_3_Median_Occupancy', 'Previous_Trip_3_TimeDifference',
       'Previous_Trip_4_Max_Occupancy', 'Previous_Trip_4_Median_Occupancy',
       'Previous_Trip_4_TimeDifference', 'Previous_Trip_5_Max_Occupancy',
       'Previous_Trip_5_Median_Occupancy', 'Previous_Trip_5_TimeDifference',
       'Previous_Trip_6_Max_Occupancy', 'Previous_Trip_6_Median_Occupancy',
       'Previous_Trip_6_TimeDifference', 'Previous_Trip_7_Max_Occupancy',
       'Previous_Trip_7_Median_Occupancy', 'Previous_Trip_7_TimeDifference',
       'Previous_Trip_8_Max_Occupancy', 'Previous_Trip_8_Median_Occupancy',
       'Previous_Trip_8_TimeDifference', 'Previous_Trip_9_Max_Occupancy',
       'Previous_Trip_9_Median_Occupancy', 'Previous_Trip_9_TimeDifference',
       'Previous_Trip_10_Max_Occupancy', 'Previous_Trip_10_Median_Occupancy',
       'Previous_Trip_10_TimeDifference', 'dayofweek_1', 'dayofweek_2',
       'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6',
       'dayofweek_7', 'is_holiday_False', 'is_holiday_True',
       'is_school_break_False', 'is_school_break_True']

    seq_features = [
    ]

    # Add previous trip context
    for i in range(1, past + 1):
        seq_features.extend([
            f'Previous_Trip_{i}_Max_Occupancy',
            f'Previous_Trip_{i}_Median_Occupancy',
            f'Previous_Trip_{i}_TimeDifference'
        ])

    flat_features = [
        'stop_sequence',
        'darksky_temperature',
        'darksky_humidity',
        'sched_hdwy',
        'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'dayofweek_7',
        'delay',
        # 'is_holiday_False',
        'is_holiday_True',
        # 'is_school_break_False',
        'is_school_break_True'
    ]

    embedding_features = {
        'month': 12,
        'day': 31,
        'hour': 23,
        'time_window': 100  # Approximate — you can update based on max()
    }

    df = df.dropna(subset=seq_features + flat_features + list(embedding_features.keys()) + [target_col])

    X_seq, X_flat, embed_inputs, y = [], [], {key: [] for key in embedding_features}, []

    values = df[seq_features + flat_features + list(embedding_features.keys()) + [target_col]].values
    for i in range(past, len(values)):
        seq_part = values[i - past:i, :len(seq_features)]
        flat_part = values[i, len(seq_features):len(seq_features) + len(flat_features)]
        embed_vals = values[i, len(seq_features) + len(flat_features):-1]  # embedding features
        label = values[i, -1]

        X_seq.append(seq_part)
        X_flat.append(flat_part)
        for j, key in enumerate(embedding_features.keys()):
            embed_inputs[key].append(embed_vals[j])
        y.append(label)

    embed_inputs = {k: np.array(v).reshape(-1, 1) for k, v in embed_inputs.items()}

    return np.array(X_seq), np.array(X_flat), embed_inputs, np.array(y)


def read_prep_Data(tasks_list):
    data_param_dictionary = {}
    lengths_dict = {}

    for taskName in tasks_list:
        for sample_type in ['TRAIN', 'VAL', 'TEST']:
            if sample_type not in lengths_dict.keys():
                lengths_dict[sample_type] = []
            data_samples = pd.read_parquet(f'../Dataset/RIDERSHIP/mtl_data/{taskName}_{sample_type}{LOAD_TYPE}.parquet')

            # X_seq, X_flat, labels = prepare_data_sequences(data_samples)
            X_seq, X_flat, embed_inputs, labels = build_model_input_dataset(data_samples, past=10)



            # print(f'taskName: {taskName}, before balance: shape of X_seq: {X_seq.shape}, shape of X_flat: {X_flat.shape}, shape of y: {labels.shape}')

            lengths_dict[sample_type].append(len(labels))

            data_param_dictionary.update({f'{taskName}_SEQ_Inputs_{sample_type}': X_seq})
            data_param_dictionary.update({f'{taskName}_FLAT_Inputs_{sample_type}': X_flat})
            data_param_dictionary.update({f'{taskName}_EMBED_Inputs_{sample_type}': embed_inputs})
            data_param_dictionary.update({f'{taskName}_Labels_{sample_type}': labels})

            # print(f'task: {taskName}, sample_type: {sample_type}, Unique class: {np.unique(labels, return_counts=True)}')

        '''*********************************'''
    print(lengths_dict)
    data_param_dict_for_specific_task = {}

    train_set_size = max(lengths_dict['TRAIN'])
    val_set_size = max(lengths_dict['VAL'])
    test_set_size = max(lengths_dict['TEST'])

    print(
        f'PER-TASK: train_set_size = {train_set_size}, test_set_size = {test_set_size}, val_set_size = {val_set_size}')
    print(f'TOTAL sum = {(train_set_size + test_set_size + val_set_size) * len(tasks_list)}')

    for taskName in tasks_list:
        X_train_seq = data_param_dictionary[f'{taskName}_SEQ_Inputs_TRAIN']
        X_train_flat = data_param_dictionary[f'{taskName}_FLAT_Inputs_TRAIN']
        X_train_embed = data_param_dictionary[f'{taskName}_EMBED_Inputs_TRAIN']
        X_val_seq = data_param_dictionary[f'{taskName}_SEQ_Inputs_VAL']
        X_val_flat = data_param_dictionary[f'{taskName}_FLAT_Inputs_VAL']
        X_val_embed = data_param_dictionary[f'{taskName}_EMBED_Inputs_VAL']
        X_test_seq = data_param_dictionary[f'{taskName}_SEQ_Inputs_TEST']
        X_test_flat = data_param_dictionary[f'{taskName}_FLAT_Inputs_TEST']
        X_test_embed = data_param_dictionary[f'{taskName}_EMBED_Inputs_TEST']

        y_train = data_param_dictionary[f'{taskName}_Labels_TRAIN']
        y_val = data_param_dictionary[f'{taskName}_Labels_VAL']
        y_test = data_param_dictionary[f'{taskName}_Labels_TEST']

        # fig,ax = plt.subplots(1,2)
        # ax[0].hist(y_val, bins=50, alpha=0.5, label='Validation')
        # ax[1].hist(y_test, bins=50, alpha=0.5, label='Test')
        # # plt.hist(y_test, bins=50, alpha=0.5, label='Test')
        # plt.legend()
        # plt.title("Distribution of Targets")
        # plt.show()

        # if len(tasks_list) > 1:
        #     X_train_seq, X_train_flat, X_train_embed, y_train = repeat_sample(train_set_size, X_train_seq, X_train_flat, X_train_embed, y_train)

        X_train = (X_train_seq, X_train_flat, X_train_embed)
        X_val = (X_val_seq, X_val_flat, X_val_embed)
        X_test = (X_test_seq, X_test_flat, X_test_embed)

        data_param_dict_for_specific_task.update({f'{taskName}_X_train': X_train})
        data_param_dict_for_specific_task.update({f'{taskName}_X_test': X_test})

        data_param_dict_for_specific_task.update({f'{taskName}_X_val': X_val})
        data_param_dict_for_specific_task.update({f'{taskName}_y_val': y_val})

        data_param_dict_for_specific_task.update({f'{taskName}_y_train': y_train})
        data_param_dict_for_specific_task.update({f'{taskName}_y_test': y_test})
    # exit(0)
    return data_param_dict_for_specific_task.copy()


def repeat_sample(SIZE, ARR_seq, ARR_flat, ARR_embed, LABELS):
    samples_to_be_repeated = SIZE - len(ARR_flat)
    if samples_to_be_repeated <= 0:
        return ARR_seq, ARR_flat, ARR_embed, LABELS

    # Sample uniformly at random (for regression)
    sampled_indices = np.random.choice(len(ARR_flat), size=samples_to_be_repeated, replace=True)

    # Concatenate to full data
    ARR_seq = np.concatenate([ARR_seq, ARR_seq[sampled_indices]], axis=0)
    ARR_flat = np.concatenate([ARR_flat, ARR_flat[sampled_indices]], axis=0)
    LABELS = np.concatenate([LABELS, LABELS[sampled_indices]], axis=0)

    # Handle embedding features
    ARR_embed_new = {}
    for key, val in ARR_embed.items():
        ARR_embed_new[key] = np.concatenate([val, val[sampled_indices]], axis=0)

    return ARR_seq, ARR_flat, ARR_embed_new, LABELS




# def dynamic_batch_sampler(data_dict, tasks_list, batch_size=32, seed=42):
#     np.random.seed(seed)
#
#     # Prepare per-task sample lists
#     task_samples = {}
#     task_labels = {}
#     max_len = 0
#
#     for task in tasks_list:
#         seq_X, flat_X, embed_dict = data_dict[f'{task}_X_train']
#         y = data_dict[f'{task}_y_train']
#
#         n = len(y)
#         max_len = max(max_len, n)
#
#         # Shuffle once
#         indices = np.random.permutation(n)
#         seq_X = seq_X[indices]
#         flat_X = flat_X[indices]
#         y = y[indices]
#         embed_dict = {k: v[indices] for k, v in embed_dict.items()}
#
#         task_samples[task] = list(zip(seq_X, flat_X, *[embed_dict[k] for k in embed_dict.keys()]))
#         task_labels[task] = list(y)
#
#     # Create task-specific iterators (cycled if necessary)
#     task_iters = {}
#     for task in tasks_list:
#         samples = task_samples[task]
#         labels = task_labels[task]
#         if len(samples) < max_len:
#             extra = max_len - len(samples)
#             samples = list(islice(cycle(samples), max_len))
#             labels = list(islice(cycle(labels), max_len))
#         task_iters[task] = list(zip(samples, labels))
#
#     # Interleave data for batching
#     batches = []
#     for i in range(0, max_len, batch_size):
#         x_batch = []
#         y_batch = []
#         for task in tasks_list:
#             task_batch = task_iters[task][i:i+batch_size]
#             for sample, label in task_batch:
#                 seq_x, flat_x, *embed_vals = sample
#                 embed_dict = {k: v.reshape(1,) for k, v in zip(data_dict[f'{task}_X_train'][2].keys(), embed_vals)}
#                 x_batch.append((seq_x, flat_x, embed_dict))
#                 y_batch.append(label)
#         batches.append((x_batch, y_batch))
#
#     return batches

# def decay_lr(epoch, optimizer):
#     if (epoch + 1) % 20 == 0:
#         optimizer.lr = optimizer.lr / 2.
#         print('Decreasing the learning rate by 1/2. New Learning Rate: {}'.format(optimizer.lr))

def decay_lr(epoch, optimizer, min_lr=1e-5):
    if (epoch + 1) % 20 == 0:
        current_lr = float(optimizer.lr.numpy())  # Get current value
        new_lr = max(current_lr / 2.0, min_lr)    # Apply decay with cap
        optimizer.lr.assign(new_lr)               # Update the tf.Variable
        print(f'Decreasing the learning rate by 1/2. New Learning Rate: {optimizer.lr.numpy()}')


def final_model(tasks_list, data_dict):
    # Shared Encoder class
    def build_shared_encoder(seq_input_shape, embed_shapes, ff_input_dim, lstm_units=64, embed_dim=4):
        # Sequential input
        seq_input = Input(shape=seq_input_shape, name='seq_input')
        x_seq = LSTM(lstm_units, return_sequences=True)(seq_input)
        x_seq = LSTM(lstm_units, return_sequences=False, dropout=0.3)(x_seq)
        x_seq = Dropout(0.2)(x_seq)

        # Flat numeric input
        ff_input = Input(shape=(ff_input_dim,), name='ff_input')
        x_ff = Dense(64)(ff_input)
        x_ff = BatchNormalization()(x_ff)
        x_ff = tf.keras.activations.relu(x_ff)
        x_ff = Dropout(0.2)(x_ff)
        x_ff = Dense(32, activation='relu')(x_ff)

        # Embedding categorical inputs
        embed_inputs = []
        embed_outputs = []
        for feature_name, vocab_size in embed_shapes.items():
            inp = Input(shape=(1,), name=f"{feature_name}_input")
            emb = Embedding(input_dim=vocab_size + 1, output_dim=embed_dim)(inp)
            emb = Flatten()(emb)
            embed_inputs.append(inp)
            embed_outputs.append(emb)

        # Combine everything
        x = Concatenate()([x_seq, x_ff] + embed_outputs)
        x = LayerNormalization()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        last_x = Dense(16, activation='relu')(x)

        model_inputs = [seq_input, ff_input] + embed_inputs
        return Model(inputs=model_inputs, outputs=last_x, name='shared_encoder')


    def build_task_decoder(latent_dim, task_name):
        inp = Input(shape=(latent_dim,))
        x = Dense(8, activation='relu')(inp)
        x = Dropout(0.2)(x)
        x = Dense(1, activation='linear', name=f"{task_name}_output")(x)  # use sigmoid if target normalized
        return Model(inputs=inp, outputs=x, name=f"{task_name}_decoder")

    seq_sample, ff_sample, embed_sample = data_dict[f'{tasks_list[0]}_X_train']

    seq_shape = seq_sample.shape[1:]  # (past, seq_feature_dim)
    flat_shape = ff_sample.shape[1]  # number of flat features
    print(f'seq_shape: {seq_shape}, flat_shape: {flat_shape}')


    embed_shapes = {
        'month': 12,
        'day': 31,
        'hour': 23,
        'time_window': 100  # Approximate — you can update based on max()
    }

    shared_encoder = build_shared_encoder(seq_shape, embed_shapes, flat_shape)

    task_decoders = {
        task: build_task_decoder(16, task) for task in tasks_list  # one decoder per route
    }
    learning_rate = 1e-3
    # global_step = tf.Variable(0, trainable=False)
    if Optimizer_Name == 'ADAM':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if Optimizer_Name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    loss_fn = tf.keras.losses.MeanSquaredError()

    # print(f'check for NaNs')
    # for task in tasks_list:
    #     y_train = data_dict[f'{task}_y_train']
    #     print(f"{task} - NaNs: {np.isnan(y_train).sum()}, Infs: {np.isinf(y_train).sum()}")

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            if multi_task_mode:
                # shared_outs = [
                #     shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=True)
                #     for (x_seq, x_ff, embed_inputs) in x_batch
                # ]
                #
                #
                # preds = [task_decoders[task](out, training=True) for task, out in zip(tasks_list, shared_outs)]
                # losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch, preds)]

                # ⬇️ Unpack and batch all sequences
                x_seqs = tf.concat([x[0] for x in x_batch], axis=0)  # shape [B_total, T, D]
                x_ffs = tf.concat([x[1] for x in x_batch], axis=0)  # shape [B_total, D_ff]
                embed_keys = x_batch[0][2].keys()
                embed_inputs = {
                    k: tf.concat([x[2][k] for x in x_batch], axis=0)
                    for k in embed_keys
                }

                # ⬇️ Run encoder once
                shared_outs_all = shared_encoder(
                    [x_seqs, x_ffs] + [embed_inputs[k] for k in embed_keys],
                    training=True
                )

                # ⬇️ Split outputs and targets per task
                batch_size = tf.shape(x_batch[0][0])[0]  # assuming fixed batch size
                shared_outs = tf.split(shared_outs_all, num_or_size_splits=len(tasks_list), axis=0)
                y_splits = y_batch  # list of [batch_size, 1] per task

                # ⬇️ Per-task decoders
                preds = [
                    task_decoders[task](shared_out, training=True)
                    for task, shared_out in zip(tasks_list, shared_outs)
                ]

                losses = [
                    loss_fn(y_true, y_pred)
                    for y_true, y_pred in zip(y_splits, preds)
                ]
            else:
                x_seq, x_ff, embed_inputs = x_batch[0]
                y_true = y_batch[0]
                shared_out = shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=True)
                y_pred = task_decoders[tasks_list[0]](shared_out, training=True)
                losses = [loss_fn(y_true, y_pred)]
                # print(f'TRAIN: here in STL MODE')

            total_loss = tf.reduce_mean(losses)

        all_vars = shared_encoder.trainable_variables + \
                   sum([decoder.trainable_variables for decoder in task_decoders.values()], [])

        grads = tape.gradient(total_loss, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))
        del tape

        return total_loss, losses

    # @tf.function
    def test_step(x_batch, y_batch):
        if multi_task_mode:
            shared_outs = [
                shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=False)
                for (x_seq, x_ff, embed_inputs) in x_batch
            ]
            preds = [task_decoders[task](out, training=False) for task, out in zip(tasks_list, shared_outs)]
            return [y.numpy() if isinstance(y, tf.Tensor) else y for y in y_batch], [p.numpy() if isinstance(p, tf.Tensor) else p for p in preds]
            # # ⬇️ Batch all inputs across tasks
            # x_seqs = tf.concat([x[0] for x in x_batch], axis=0)  # [num_tasks * batch_size, T, D]
            # x_ffs = tf.concat([x[1] for x in x_batch], axis=0)  # [num_tasks * batch_size, D_ff]
            # embed_keys = x_batch[0][2].keys()
            # embed_inputs = {
            #     k: tf.concat([x[2][k] for x in x_batch], axis=0)
            #     for k in embed_keys
            # }
            #
            # # ⬇️ Forward pass through shared encoder
            # shared_outs_all = shared_encoder(
            #     [x_seqs, x_ffs] + [embed_inputs[k] for k in embed_keys],
            #     training=False
            # )
            #
            # # ⬇️ Split back into per-task chunks
            # batch_size = tf.shape(x_batch[0][0])[0]
            # shared_outs = tf.split(shared_outs_all, num_or_size_splits=len(tasks_list), axis=0)
            #
            # preds = [
            #     task_decoders[task](shared_out, training=False)
            #     for task, shared_out in zip(tasks_list, shared_outs)
            # ]
            #
            # # ⬇️ Convert predictions and labels to numpy
            # y_true_out = [y.numpy() if isinstance(y, tf.Tensor) else y for y in y_batch]
            # y_pred_out = [p.numpy() if isinstance(p, tf.Tensor) else p for p in preds]
            #
            # return y_true_out, y_pred_out

        else:
            # print(f'len(x_batch) = {len(x_batch)}, len(y_batch) = {len(y_batch)}')
            x_seq, x_ff, embed_inputs = x_batch[0]
            y_true = y_batch[0]
            shared_out = shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=False)
            y_pred = task_decoders[tasks_list[0]](shared_out, training=False)

            # Ensure both are wrapped in lists of arrays
            if isinstance(y_true, tf.Tensor):
                y_true = y_true.numpy()
            if isinstance(y_pred, tf.Tensor):
                y_pred = y_pred.numpy()
            # print(f'here in STL MODE')

            return [np.array(y_true)], [np.array(y_pred)]

    @tf.function
    def train_step_ITA(x_batch_train, y_batch_train, first_step=False):

        task_gains = {task: {task: {} for task in TASKS}
                      for task in TASKS}
        with tf.GradientTape(persistent=True) as tape:
            shared_outs = [shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=True)
                           for (x_seq, x_ff, embed_inputs) in x_batch_train]
            preds = [task_decoders[task](out, training=True)
                     for task, out in zip(tasks_list, shared_outs)]
            losses = []
            for task, y_true, y_pred in zip(tasks_list, y_batch_train, preds):
                loss = loss_fn(y_true, y_pred)
                losses.append(loss)

            losses_dict = {task: loss for task, loss in zip(tasks_list, losses)}
            total_loss = tf.reduce_mean(losses)

            # Compute the gradient of the task-specific loss w.r.t. the shared base.
            single_task_specific_gradients = [
                (single_task, tape.gradient(losses_dict[single_task], shared_encoder.trainable_weights)) for
                single_task in tasks_list]  # Only has single tasks gradients for now
        # Compute for regular model update
        all_tasks_gradients = [tf.add_n([task_gradient[i] for _, task_gradient in single_task_specific_gradients])
                               for i in range(len(shared_encoder.trainable_weights))]

        before_update_losses = {task: loss for task, loss in losses_dict.items()}
        original_shared_weights = [tf.identity(weight) for weight in shared_encoder.trainable_weights]
        original_decoder_weights = {sch_id: [tf.identity(weight) for weight in decoder.trainable_weights]
                                    for sch_id, decoder in task_decoders.items()}

        # lr_val = optimizer.lr if not callable(optimizer.lr) else optimizer.lr(optimizer.iterations)
        lr_val = optimizer.lr
        import tqdm
        for base_task, task_gradient in tqdm.tqdm(single_task_specific_gradients):
            if first_step:
                # Regular update for the first step
                base_update = [lr_val * grad for grad in task_gradient]
                base_updated = [param - update for param, update in zip(shared_encoder.trainable_weights, base_update)]
            else:
                # Momentum-based update for later steps
                base_update = [(lr_val * grad - optimizer._momentum * optimizer.get_slot(param, 'momentum'))
                               for param, grad in zip(shared_encoder.trainable_weights, task_gradient)]
                base_updated = [param + update for param, update in zip(shared_encoder.trainable_weights, base_update)]

            # Temporarily update shared encoder weights with base_updated for AFFINITY_Calc computation
            for original_param, updated_param in zip(shared_encoder.trainable_weights, base_updated):
                original_param.assign(updated_param)

            shared_outs = [shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=False)
                           for (x_seq, x_ff, embed_inputs) in x_batch_train]
            preds = [task_decoders[task](out, training=False) for task, out in zip(tasks_list, shared_outs)]
            after_losses = []
            for y_true, y_pred in zip(y_batch_train, preds):
                loss = loss_fn(y_true, y_pred)
                after_losses.append(loss)
            after_update_losses = {task: loss for task, loss in zip(tasks_list, after_losses)}

            '''Compute task gain'''
            task_gain = {
                second_task: (1.0 - after_update_losses[second_task] / before_update_losses[second_task]) / optimizer.lr
                for second_task in TASKS}
            task_gains[base_task] = task_gain

            # Revert shared encoder weights back to the original parameters after AFFINITY_Calc computation
            for original_param, updated_param in zip(shared_encoder.trainable_weights, original_shared_weights):
                original_param.assign(updated_param)



        # Apply gradients to each decoder separately
        for task in tasks_list:
            grads = tape.gradient(losses_dict[task], task_decoders[task].trainable_weights)
            optimizer.apply_gradients(zip(grads, task_decoders[task].trainable_weights))

        all_tasks_gradients = [tf.add_n([task_gradient[idx] for _, task_gradient in single_task_specific_gradients])
                               for idx in range(len(shared_encoder.trainable_weights))]
        optimizer.apply_gradients(zip(all_tasks_gradients, shared_encoder.trainable_weights))

        # Save snapshot
        original_shared_weights = [tf.identity(w) for w in shared_encoder.trainable_weights]
        original_decoder_weights = {
            task: [tf.identity(w) for w in task_decoders[task].trainable_weights]
            for task in tasks_list
        }

        # global_step.assign_add(1)
        del tape
        return total_loss, task_gains, original_shared_weights, original_decoder_weights

    @tf.function
    def train_step_ITA_Approx(x_batch_train, y_batch_train, first_step=False):
        task_gains = {task: {other_task: 0.0 for other_task in tasks_list} for task in tasks_list}
        task_gains_approximation = {}
        with tf.GradientTape(persistent=True) as tape:
            shared_outs = [shared_encoder([x_seq, x_ff] + [embed_inputs[k] for k in embed_inputs.keys()], training=True)
                for (x_seq, x_ff, embed_inputs) in x_batch_train]
            preds = [task_decoders[task](out, training=True)
                     for task, out in zip(tasks_list, shared_outs)]
            losses = []
            for task, y_true, y_pred in zip(tasks_list, y_batch_train, preds):
                loss = loss_fn(y_true, y_pred)
                losses.append(loss)

            losses_dict = {task: loss for task, loss in zip(tasks_list, losses)}
            total_loss = tf.reduce_mean(losses)

            # Compute the gradient of the task-specific loss w.r.t. the shared base.
            single_task_specific_gradients = [
                (single_task, tape.gradient(losses_dict[single_task], shared_encoder.trainable_weights)) for
                single_task in tasks_list]  # Only has single tasks gradients for now

        '''flatten and concatenate gradients for all tasks to get Jacobian matrix'''
        reshaped_gradients = [tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
                              for _, grads in single_task_specific_gradients]

        # lr_val = optimizer.lr if not callable(optimizer.lr) else optimizer.lr(optimizer.iterations)
        lr_val = optimizer.lr
        task_gradient_updates = {}
        for single_task, task_gradient in single_task_specific_gradients:
            if w_momentum:
                if first_step:
                    base_update = [lr_val * grad for grad in task_gradient]
                else:
                    base_update = [(lr_val * grad - optimizer._momentum * optimizer.get_slot(param, 'momentum'))
                                   for param, grad in zip(shared_encoder.trainable_weights, task_gradient)]
            else:
                base_update = [lr_val * grad for grad in task_gradient]

            task_gradient_updates[single_task] = base_update

        # # Compute optimizer-like updates (ITA approx)
        # if w_momentum:
        #     task_updates = {}
        #     for task, grads in task_gradients.items():
        #         if first_step:
        #             updates = [lr * g for g in grads]
        #         else:
        #             updates = [
        #                 lr * g - optimizer.momentum * optimizer.get_slot(w, "momentum")
        #                 for g, w in zip(grads, shared_encoder.trainable_weights)
        #             ]
        #         task_updates[task] = updates
        # else:
        #     task_updates = {}
        #     for task, grads in task_gradients.items():
        #         updates = [lr * g for g in grads]
        #         task_updates[task] = updates

        # Flatten gradients and updates
        # flat_grads = {
        #     task: tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        #     for task, grads in task_gradients.items()
        # }
        # flat_updates = {
        #     task: tf.concat([tf.reshape(u, [-1]) for u in updates], axis=0)
        #     for task, updates in task_updates.items()
        # }

        '''flatten and concatenate updates for all tasks to get update matrix'''
        reshaped_updates = [tf.concat([tf.reshape(update, [-1]) for update in updates], axis=0)
                            for _, updates in task_gradient_updates.items()]
        # Construct G and U matrices
        # G = tf.stack([flat_grads[task] for task in tasks_list])  # [T, D]
        G = tf.convert_to_tensor(reshaped_gradients)  # Jacobian matrix - all gradients
        # U = tf.stack([flat_updates[task] for task in tasks_list])  # [T, D]
        U = tf.convert_to_tensor(reshaped_updates)  # Update matrix -- all updates
        L = [losses_dict[task] for task in tasks_list]  # Loss matrix
        # L = tf.reshape(tf.stack([losses_dict[task] for task in tasks_list]), (-1, 1))  # [T, 1]
        L = tf.reshape(tf.convert_to_tensor(L), (-1, 1))


        ita_approximation_G_U = tf.matmul(G, U, transpose_b=True)
        # '''wo loss'''
        ita_approximation = tf.divide(ita_approximation_G_U, L)
        ita_approximation = tf.divide(ita_approximation, lr_val)

        for idx, base_task in enumerate(tasks_list):
            # Extract the ith column from ita_approximation of base task onto other tasks
            ita_per_task = ita_approximation[:, idx]
            # Create a temp-dictionary mapping each task to its corresponding value in ita_per_task and Store the result in task_gains_approximation
            task_gains_approximation[base_task] = {task: ita_per_task[tasks_list.index(task)] for task in tasks_list}

        # Apply gradients to each decoder separately
        for task in tasks_list:
            grads = tape.gradient(losses_dict[task], task_decoders[task].trainable_weights)
            optimizer.apply_gradients(zip(grads, task_decoders[task].trainable_weights))

        all_tasks_gradients = [tf.add_n([task_gradient[idx] for _, task_gradient in single_task_specific_gradients])
                               for idx in range(len(shared_encoder.trainable_weights))]
        optimizer.apply_gradients(zip(all_tasks_gradients, shared_encoder.trainable_weights))

        # Save snapshot
        original_shared_weights = [tf.identity(w) for w in shared_encoder.trainable_weights]
        original_decoder_weights = {
            task: [tf.identity(w) for w in task_decoders[task].trainable_weights]
            for task in tasks_list
        }

        # global_step.assign_add(1)

        return total_loss, task_gains_approximation, original_shared_weights, original_decoder_weights

    def save_gradients(dynamic_batches, shared_encoder, task_decoders, loss_fn, tasks_list, project_matrix,
                       gradients_dir, batch_size=32):
        task_gradients = {task: [] for task in tasks_list}

        for batch_idx in range(len(dynamic_batches)):
            x_batch, y_batch = dynamic_batches[batch_idx]

            # Unpack the batched inputs as done in train_step
            x_seqs = tf.concat([x[0] for x in x_batch], axis=0)
            x_ffs = tf.concat([x[1] for x in x_batch], axis=0)
            embed_keys = x_batch[0][2].keys()
            embed_inputs = {
                k: tf.concat([x[2][k] for x in x_batch], axis=0)
                for k in embed_keys
            }

            with tf.GradientTape(persistent=True) as tape:
                shared_outs_all = shared_encoder(
                    [x_seqs, x_ffs] + [embed_inputs[k] for k in embed_keys],
                    training=True
                )
                shared_outs = tf.split(shared_outs_all, num_or_size_splits=len(tasks_list), axis=0)
                preds = [task_decoders[task](shared_out, training=True) for task, shared_out in
                         zip(tasks_list, shared_outs)]
                losses = [loss_fn(y_true, y_pred) for y_true, y_pred in zip(y_batch, preds)]

            for task, loss in zip(tasks_list, losses):
                grads = tape.gradient(loss, shared_encoder.trainable_weights)
                flat_grad = tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0).numpy()

                if flat_grad.size != project_matrix.shape[0]:
                    raise ValueError(
                        f"Gradient size {flat_grad.size} does not match expected size {project_matrix.shape[0]}")

                projected_grad = (flat_grad.reshape(1, -1) @ project_matrix).flatten()
                task_gradients[task].append(projected_grad)

            del tape  # Clean up
        for task_name, gradients in task_gradients.items():
            np.save(f"{gradients_dir}/{task_name}_train_gradients.npy", gradients)


    # Training loop
    best_weights = None
    best_loss = float('inf')
    patience = MAX_PATIENCE
    timeStart = time.time()

    def dynamic_batch_sampler(data_dict, tasks_list, batch_size=512, seed=42):

        task_data = {}

        # Step 1: Shuffle each task's data and track batch indices
        for t in tasks_list:
            seq_X, flat_X, embed_dict = data_dict[f'{t}_X_train']
            y = data_dict[f'{t}_y_train']

            indices = np.random.permutation(len(y))
            seq_X = seq_X[indices]
            flat_X = flat_X[indices]
            y = y[indices]
            embed_dict = {k: v[indices] for k, v in embed_dict.items()}

            num_batches = int(np.ceil(len(y) / batch_size))

            task_data[t] = {
                'seq': seq_X,
                'flat': flat_X,
                'embed': embed_dict,
                'y': y,
                'num_batches': num_batches,
                'pointer': 0
            }

        # Step 2: Determine max batches
        max_batches = max(task_data[t]['num_batches'] for t in tasks_list)
        for t in tasks_list:
            print(f'{t}: {task_data[t]["num_batches"]}', end=' ')

        # if Method_name == 'ITA_APPROX':
        #     max_batches = 300
        # print(f'max batches: {max_batches}')

        dynamic_batches = []

        for b in range(max_batches):
            x_batch, y_batch = [], []

            for t in tasks_list:
                d = task_data[t]
                start = d['pointer']
                end = start + batch_size

                # If we reach the end, sample new indices without replacement if possible
                if end > len(d['y']):
                    indices = np.random.permutation(len(d['y']))
                    d['seq'] = d['seq'][indices]
                    d['flat'] = d['flat'][indices]
                    d['y'] = d['y'][indices]
                    d['embed'] = {k: v[indices] for k, v in d['embed'].items()}
                    d['pointer'] = 0
                    start = 0
                    end = batch_size

                seq_b = d['seq'][start:end]
                flat_b = d['flat'][start:end]
                y_b = d['y'][start:end]
                embed_b = {k: v[start:end] for k, v in d['embed'].items()}

                d['pointer'] += batch_size
                x_batch.append((seq_b, flat_b, embed_b))
                y_batch.append(y_b)

            dynamic_batches.append((x_batch, y_batch))

        dynamic_batches_summary = {
            task: task_data[task]['num_batches'] for task in tasks_list
        }
        # print(dynamic_batches_summary)

        dynamic_batch_lengths = [len(x[0]) for x in dynamic_batches]
        # print(dynamic_batch_lengths[:10])  # Return a preview
        return dynamic_batches

    dynamic_batches = dynamic_batch_sampler(data_dict, tasks_list, batch_size=batch_size, seed=seed)
    # print(len(dynamic_batches))


    # # # Shuffle once and cache batches
    # task_batches_x = [[] for _ in tasks_list]
    # task_batches_y = [[] for _ in tasks_list]
    #
    # for i, t in enumerate(tasks_list):
    #     seq_X, flat_X, embed_dict = data_dict[f'{t}_X_train']
    #     y = data_dict[f'{t}_y_train']
    #
    #     indices = np.random.permutation(len(y))  # shuffle once
    #
    #     seq_X = seq_X[indices]
    #     flat_X = flat_X[indices]
    #     y = y[indices]
    #
    #     # Apply shuffling to each embedded feature separately
    #     shuffled_embed_dict = {
    #         k: v[indices] for k, v in embed_dict.items()
    #     }
    #
    #     num_batches = int(np.ceil(len(y) / batch_size))
    #     for b in range(num_batches):
    #         start = b * batch_size
    #         end = (b + 1) * batch_size
    #
    #         embed_batch = {
    #             k: v[start:end] for k, v in shuffled_embed_dict.items()
    #         }
    #
    #         task_batches_x[i].append((seq_X[start:end], flat_X[start:end], embed_batch))
    #         task_batches_y[i].append(y[start:end])


    # print(f'Number of Batches: {len(task_batches_x[0])}')
    # for i, task_batch in enumerate(task_batches_x):
    #     print(f"Task {tasks_list[i]}: {len(task_batch)} batches")

    epoch_losses = []
    gradient_metrics = {task: [] for task in tasks_list}
    TRAIN_SIZE = -math.inf
    for i, t in enumerate(tasks_list):
        seq_X, flat_X, embed_dict = data_dict[f'{t}_X_train']
        TRAIN_SIZE = max(len(seq_X),TRAIN_SIZE)
    print(f'TRAIN_SIZE: {TRAIN_SIZE}')
    print(f'math.ceil(TRAIN_SIZE/batch_size): {math.ceil(TRAIN_SIZE/batch_size)}')

    task_counts = {t: 0 for t in tasks_list}
    for x_batch, y_batch in dynamic_batches:
        for first_task in [t for t, y in zip(tasks_list, y_batch) if len(y) > 0]:
            task_counts[first_task] += 1
    print(f'task_counts: {task_counts}')
    for epoch in range(num_epochs):
        epoch_time = time.time()

        decay_lr(epoch,optimizer)

        batch_grad_metrics = {combined_task: {task: 0. for task in tasks_list} for combined_task in
                              gradient_metrics}
        # b_start = time.time()
        for batch_idx  in range(len(dynamic_batches)):

            x_batch, y_batch = dynamic_batches[batch_idx]

            if AFFINITY_Calc and Method_name == 'ITA_Approx':
                loss, task_gains, shared_weights, decoder_weights = train_step_ITA_Approx(x_batch, y_batch,
                                                                                          first_step=(len(optimizer.variables()) == 0))
            elif AFFINITY_Calc and Method_name == 'ITA':
                loss, task_gains, shared_weights, decoder_weights = train_step_ITA(x_batch, y_batch,
                                                                                          first_step=(len(optimizer.variables()) == 0))
            else:
                loss, _ = train_step(x_batch, y_batch)


            epoch_losses.append(loss.numpy())
            if AFFINITY_Calc:

                # # Record batch-level training and gradient metrics.
                for first_task, task_gain_map in task_gains.items():
                    for second_task, gain in task_gain_map.items():
                        # print(f'first_task = {first_task}\tsecond_task = {second_task}\tgain = {gain}')
                        batch_grad_metrics[first_task][second_task] += (gain.numpy()/ (math.ceil(TRAIN_SIZE / batch_size)))

                # print(f'batch_grad_metrics: {batch_grad_metrics}')

            # if batch_idx % 100 == 0:
            #     print(f"Batch {batch_idx}: {time.time() - b_start:.3f}s", end='\t')
            #     b_start = time.time()
        # Evaluate on full validation set
        y_true_val, y_pred_val = test_step(
            [data_dict[f'{t}_X_val'] for t in tasks_list],
            [data_dict[f'{t}_y_val'] for t in tasks_list]
        )
        y_true_all = np.concatenate([y.flatten() for y in y_true_val])
        y_pred_all = np.concatenate([y.flatten() for y in y_pred_val])
        val_loss = np.mean((y_pred_all - y_true_all) ** 2)

        if epoch % 5 == 0:
            print()
            print(
                f"Epoch {epoch + 1}: Train Loss = {epoch_losses[-1]:.6f}, Val Loss = {val_loss:.6f},\t patience = {patience}, TIME:  {epoch}: {time.time() - epoch_time:.2f}s")

        # Early stopping
        if val_loss < best_loss:
            best_loss = min(val_loss, best_loss)
            best_weights = (
                copy.deepcopy(shared_encoder.get_weights()),
                {t: dec.get_weights() for t, dec in task_decoders.items()}
            )
            patience = MAX_PATIENCE
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break



        if AFFINITY_Calc:
            # Record epoch-level training and gradient metrics.
            for combined_task, task_gain_map in batch_grad_metrics.items():
                gradient_metrics[combined_task].append(task_gain_map)

            # if epoch % 100 == 0:
            print(f'epoch {epoch}, gradient_metrics = {len(gradient_metrics)}')
            for source_task, task_gain_map in gradient_metrics.items():
                print(f'source_task = {source_task}\ttask_gain_map = {len(task_gain_map)}\n{task_gain_map[-1]}')
                break

    # Load best weights
    shared_encoder.set_weights(best_weights[0])
    for t in tasks_list:
        task_decoders[t].set_weights(best_weights[1][t])

    '''new parts'''
    if len(tasks_list) == len(TASKS):
        '''save best weights and model to a file'''
        model_base_dir = f'{datasetName}_model_weights'
        if not os.path.exists(model_base_dir):
            os.makedirs(model_base_dir)

        gradients_dir = f'{datasetName}_gradients_run_{seed}'
        if not os.path.exists(gradients_dir):
            os.makedirs(gradients_dir)

        model_dir = f'{model_base_dir}/run_{seed}'
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

        # Save gradients for each task on training data
        # gradients_dir = './gradients'
        # os.makedirs(gradients_dir, exist_ok=True)


        save_gradients(
            dynamic_batches=dynamic_batches,
            shared_encoder=shared_encoder,
            task_decoders=task_decoders,
            loss_fn=loss_fn,
            tasks_list=tasks_list,
            project_matrix=project_matrix,
            gradients_dir=gradients_dir
        )
        print("✅ Projected gradients saved for all tasks.")
    # --- Final Evaluation ---
    y_true_list, y_pred_list = test_step(
        [data_dict[f'{t}_X_test'] for t in tasks_list],
        [data_dict[f'{t}_y_test'] for t in tasks_list]
    )

    y_true_all = np.concatenate([y.flatten() for y in y_true_list])
    y_pred_all = np.concatenate([y.flatten() for y in y_pred_list])

    mse = np.mean((y_pred_all - y_true_all) ** 2)
    var = np.var(y_true_all)
    r2 = 1 - mse / var

    print("=== FINAL EVALUATION ===")
    print(f"Global MSE: {mse:.4f}")
    print(f"Global Var: {var:.4f}")
    print(f"Global R^2: {r2:.4f}\n")
    print(f'standard deviation: {np.std(y_pred_all)}')

    # Per-task R²
    task_r_square = {}
    task_mse = {}
    task_variance = {}
    for i, t in enumerate(tasks_list):
        y_t = y_true_list[i].flatten()
        y_p = y_pred_list[i].flatten()
        mse_t = np.mean((y_p - y_t) ** 2)
        var_t = np.var(y_t)

        r2_t = r2_score(y_t, y_p)
        task_r_square[t] = r2_t
        task_mse[t] = mse_t
        task_variance[t] = var_t
        print(f"[{t}] R² = {r2_t:.4f} | Manual R² = {1 - mse_t / var_t:.4f} | MSE = {mse_t:.4f} | Var = {var_t:.4f}")
    tot_loss = np.sum(list(task_mse.values()))
    print(f'task_mse: {task_mse}')
    print(f'task_r_square: {task_r_square}')

    if AFFINITY_Calc:
        if not w_momentum:
            ita_file = f'{ResultPath}/ITA/gradient_metrics_{Method_name}_seed_{seed}_{name_suffix}_PATIENCE_{MAX_PATIENCE}{LOAD_TYPE}.csv'
        else:
            ita_file = f'{ResultPath}/ITA/gradient_metrics_{Method_name}_w_momentum_seed_{seed}_{name_suffix}_PATIENCE_{MAX_PATIENCE}{LOAD_TYPE}.csv'
        with open(ita_file, 'w') as f:
            for key in gradient_metrics.keys():
                f.write("%s,%s\n" % (key, gradient_metrics[key]))
        f.close()






    time_elapsed = time.time() - timeStart
    return tot_loss, task_mse, task_r_square, time_elapsed, var


if __name__ == "__main__":
    datasetName = 'Occupancy'
    DataPath = f'../data/mtl_data/'
    import sys

    if len(sys.argv) == 1:
        AFFINITY_Calc = 0
        group_type = 'ALL'
    else:
        AFFINITY_Calc = int(sys.argv[1])  # AFFINITY_Calc' 0 or 1
        group_type = sys.argv[2]  # 'ALL', 'STL', 'PTL'
    print(f'total input: {sys.argv}')
    w_momentum = False
    # AFFINITY_Calc = 1
    if AFFINITY_Calc:
        Method_name = 'ITA'  # sys.argv[1]
        # group_type = 'ALL'
    else:
        Method_name = 'SimpleMTL'
        # group_type = 'ALL'

    print(sys.argv)

    ResultPath = '../RESULTS/'
    print(f'MTL for {datasetName} dataset with group {group_type}:')
    TASKS = ['10C_INBOUND', '10C_OUTBOUND', '13_INBOUND', '13_OUTBOUND', '16_INBOUND', '16_OUTBOUND',
             '1_INBOUND', '1_OUTBOUND', '7_INBOUND', '7_OUTBOUND']
    TASKS = sorted(TASKS)
    print(f'TASKS = {TASKS}')

    num_epochs = 1000
    batch_size = 512
    Optimizer_Name = 'SGD'
    MAX_PATIENCE = 8
    LOAD_TYPE = '_MEAN'

    import itertools

    if AFFINITY_Calc:
        TASK_Group = [tuple(TASKS)]
        name_suffix = 'ALL'
        # SEEDS = [2025, 2024, 2023]

    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
        SEEDS = [seed]
    else:
        # SEEDS = [2025,2024,2023]
        SEEDS = [0,1,2,3]
    '''Pairs'''
    if group_type == 'PTL':
        pairs = list(itertools.combinations(TASKS, 2))
        print(f'pairs = {pairs}')
        print(f'len(pairs): {len(pairs)}')
        TASK_Group = pairs
        name_suffix = 'pairs'


    '''STL'''
    if group_type == 'STL':
        Tasks_tuples = [tuple([task]) for task in TASKS]
        TASK_Group = Tasks_tuples
        name_suffix = 'STL'


    # '''ALL'''
    if group_type == 'ALL':
        TASK_Group = [tuple(TASKS)]
        name_suffix = 'ALL'



    if group_type in ['3','4','5','6','7','8','9']:
        SEEDS = [2025]
        gt = int(group_type)
        groups = list(itertools.combinations(TASKS, gt))
        TASK_Group = groups
        print(f'TASK_Group = {TASK_Group}')
        print(f'total TASK_Group: {len(TASK_Group)}')
        name_suffix = f'G{gt}'
        print(f'NAME of GROUPS: {name_suffix}')

    if group_type == 'OTHER_NEW':
        selected_grps = pd.read_csv(f'../RESULTS/Occupancy_Train_Groups_OTHER_NEW.csv')
        TASK_Group = selected_grps['Task_group'].tolist()
        TASK_Group = [ast.literal_eval(grp) for grp in TASK_Group]
        part = sys.argv[4]
        tot_len = len(TASK_Group) // 2
        if '1' in part:
            TASK_Group = TASK_Group[:tot_len]
        if '2' in part:
            TASK_Group = TASK_Group[tot_len:]
        # if '3' in part:
        #     TASK_Group = TASK_Group[tot_len*2:tot_len*3]
        # if '4' in part:
        #     TASK_Group = TASK_Group[tot_len*3:]
        print(f'Total TASK_Group: {len(TASK_Group)}')
        name_suffix = group_type + f'Part_{part}'

    if group_type == 'SELECTED_Rem':
        selected_grps = pd.read_csv(f'../RESULTS/selected_groups_for_train_remaining.csv')
        print(f'total remaining TASK_Group: {len(selected_grps)}')
        TASK_Group = selected_grps['Task_group'].tolist()
        TASK_Group = [ast.literal_eval(grp) for grp in TASK_Group]
        part = sys.argv[4]
        tot_len = len(TASK_Group) // 3
        if '1' in part:
            TASK_Group = TASK_Group[:tot_len]
        if '2' in part:
            TASK_Group = TASK_Group[tot_len:tot_len * 2]
        if '3' in part:
            TASK_Group = TASK_Group[tot_len * 2:tot_len * 3]
        #
        print(f'Total TASK_Group: {len(TASK_Group)}')
        name_suffix = group_type + f'Part_{part}'
        # SEEDS = [2024]


    print(f'first groyup = {TASK_Group[0]}')
    print(f'last group = {TASK_Group[-1]}')
    if len(sys.argv) > 1:
        print(f'Excecution of {sys.argv[0]} {sys.argv[1]} - Method_name: {Method_name}, SEEDS: {SEEDS}')

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        if group_type in ['3','4','5','6','7','8','9']:
            existing_file = f'../RESULTS/Occupancy_G{group_type}_seed_{seed}_SGD_Batch_512_REGRESSION_PREVTRIPS_PATIENCE_8{LOAD_TYPE}.csv'
            if os.path.exists(existing_file):
                existing_df = pd.read_csv(existing_file)
                Total_Loss = list(existing_df['Total_Loss'])
                Task_group = list(existing_df['Task_group'])
                Individual_Task_Score = list(existing_df['Individual_Task_Score'])
                AVG_RSQ = list(existing_df['AVG_RSQ'])
                Individual_RSquare =list(existing_df['Individual_RSquare'])
                TASK_VARIANCE =list(existing_df['TASK_VARIANCE'])
                TIME_TAKEN = list(existing_df['Time_Elapsed'])
                curr_len = len(Individual_Task_Score)

                print(f'total task grouops : {len(TASK_Group)}')
                TASK_Group = TASK_Group[curr_len:]
                print(f'Total Task Group Remaining : {len(TASK_Group)}')
                print(f'Already Trained : {len(Individual_Task_Score)}')

        elif group_type in ['OTHER']:
            existing_file = f'{ResultPath}/{datasetName}_{name_suffix}_seed_{seed}_{Optimizer_Name}_Batch_{batch_size}_REGRESSION_PREVTRIPS_PATIENCE_{MAX_PATIENCE}{LOAD_TYPE}.csv'
            if os.path.exists(existing_file):
                existing_df = pd.read_csv(existing_file)
                Total_Loss = list(existing_df['Total_Loss'])
                Task_group = list(existing_df['Task_group'])
                Individual_Task_Score = list(existing_df['Individual_Task_Score'])
                AVG_RSQ = list(existing_df['AVG_RSQ'])
                Individual_RSquare = list(existing_df['Individual_RSquare'])
                TASK_VARIANCE = list(existing_df['TASK_VARIANCE'])
                TIME_TAKEN = list(existing_df['Time_Elapsed'])
                curr_len = len(Individual_Task_Score)

                print(f'total task grouops : {len(TASK_Group)}')
                TASK_Group = TASK_Group[curr_len:]
                print(f'Total Task Group Remaining : {len(TASK_Group)}')
                print(f'Already Trained : {len(Individual_Task_Score)}')

            else:
                Task_group = []
                Total_Loss = []
                Individual_RSquare = []
                AVG_RSQ = []
                Individual_Task_Score = []
                TIME_TAKEN = []
                TASK_VARIANCE = []

        else:
            Task_group = []
            Total_Loss = []
            Individual_RSquare = []
            AVG_RSQ = []
            Individual_Task_Score = []
            TIME_TAKEN = []
            TASK_VARIANCE = []

        print(
            f'length of each list: {len(Total_Loss), len(Individual_Task_Score), len(AVG_RSQ), len(TASK_VARIANCE), len(TIME_TAKEN), len(Individual_RSquare)}')

        for count in range(len(TASK_Group)):
            print(f'Initial Training for {datasetName}-partition {count}, \nGROUP = {TASK_Group[count]}')
            task_group = TASK_Group[count]
            multi_task_mode = len(task_group) > 1
            print(f'Multi task mode: {multi_task_mode}')


            args_tasks = []
            group_score = {}
            group_avg_err = {}
            group_avg_AP = {}
            tmp_task_score = []

            data_dictionary = read_prep_Data(task_group)

            tot_loss, task_scores, task_rsquare, Total_time,variance = final_model(task_group, data_dictionary)
            print(f'tot_loss = {tot_loss}')

            print(f'total_time = {Total_time}')
            print(f'avg time in minutes = {np.mean(Total_time) / 60}')
            avg_r_sq = np.mean(list(task_rsquare.values()))
            Task_group.append(task_group)
            Total_Loss.append(tot_loss)
            AVG_RSQ.append(avg_r_sq)
            Individual_Task_Score.append(copy.deepcopy(task_scores))
            Individual_RSquare.append(copy.deepcopy(task_rsquare))
            TIME_TAKEN.append(Total_time)
            TASK_VARIANCE.append(copy.deepcopy(variance))

            print(len(Total_Loss), len(Task_group), len(Individual_Task_Score), len(Individual_RSquare))

            temp_res = pd.DataFrame({'Total_Loss': Total_Loss,
                                     'Task_group': Task_group,
                                     'Individual_Task_Score': Individual_Task_Score,
                                     'AVG_RSQ': AVG_RSQ,
                                     'Individual_RSquare': Individual_RSquare,
                                     'TASK_VARIANCE': TASK_VARIANCE,
                                     'Time_Elapsed': TIME_TAKEN,
                                     })

            if AFFINITY_Calc:
                temp_res.to_csv(
                    f'{ResultPath}/{datasetName}_SimpleMTL_{Method_name}_seed_{seed}_{Optimizer_Name}_{name_suffix}{LOAD_TYPE}.csv',
                    index=False)

                print(f'total_time = {Total_time}')
                print(f'avg time in minutes = {np.mean(Total_time) / 60}')

                '''save time to txt file'''
                timefile = f'{ResultPath}/{datasetName}_{Method_name}_time_seed_{seed}_{Optimizer_Name}_{name_suffix}{LOAD_TYPE}.txt'
                with open(timefile, 'a') as f:
                    f.write(f'{np.mean(Total_time) / 60}\n')
                    f.write(f'{Total_time}\n')

                f.close()
            else:
                temp_res.to_csv(
                    f'{ResultPath}/{datasetName}_{name_suffix}_seed_{seed}_{Optimizer_Name}_Batch_{batch_size}_REGRESSION_PREVTRIPS_PATIENCE_{MAX_PATIENCE}{LOAD_TYPE}.csv',
                    index=False)