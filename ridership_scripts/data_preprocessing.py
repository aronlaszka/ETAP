
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from collections import deque
from datetime import timedelta

pd.set_option('display.max_columns', None)

def prepare_linklevel(df, train_dates=None, val_dates=None, test_dates=None,
                      cat_columns=None, num_columns=None, ohe_columns=None,
                      feature_label='load', time_feature_used='arrival_time',
                      scaler='minmax',
                      prefit_ohe=None, prefit_label_encoders=None, prefit_scaler=None):
    # Decide scaler type
    scaler_obj = prefit_scaler or (MinMaxScaler() if scaler == 'minmax' else StandardScaler())

    # ========== One-Hot Encoding ==========
    if prefit_ohe is not None:
        ohe_encoder = prefit_ohe
    else:
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe_encoder.fit(df[ohe_columns])

    ohe_encoded = ohe_encoder.transform(df[ohe_columns])
    ohe_feature_names = ohe_encoder.get_feature_names_out(ohe_columns)
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df.index)

    df = pd.concat([df.drop(columns=ohe_columns), ohe_df], axis=1)

    if time_feature_used not in df.columns:
        raise ValueError(f"Column '{time_feature_used}' not found in DataFrame.")

    # ========== Date-Based Splitting ==========
    if train_dates is not None and val_dates is not None and test_dates is not None:

        # Normalize the input dates to date type
        train_start, train_end = train_dates[0], train_dates[1]
        val_start, val_end = val_dates[0], val_dates[1]
        test_start, test_end = test_dates[0], test_dates[1]

        df['_split_date'] = df[time_feature_used].dt.date
        df = df.sort_values(by='_split_date')  # after _split_date is created in prepare_linklevel

        train_df = df[df['_split_date'].between(train_start, train_end)].copy()
        val_df = df[df['_split_date'].between(val_start, val_end)].copy()
        test_df = df[df['_split_date'].between(test_start, test_end)].copy()

        print("Split column range:", df['_split_date'].min(), "to", df['_split_date'].max())
        print("Train range:", train_start, train_end)
        print("Val range:", val_start, val_end)
        print("Test range:", test_start, test_end)

        # Optionally drop this helper column after splitting
        df.drop(columns=['_split_date'], inplace=True)
        train_df.drop(columns=['_split_date'], inplace=True)
        val_df.drop(columns=['_split_date'], inplace=True)
        test_df.drop(columns=['_split_date'], inplace=True)

        print("Before encoding:", train_df.shape, val_df.shape, test_df.shape)
    else:
        train_df = val_df = test_df = None

    # ========== Label Encoding ==========
    label_encoders = prefit_label_encoders or {}
    for col in [c for c in cat_columns if c not in ohe_columns and c != feature_label]:
        if prefit_label_encoders is not None and col in prefit_label_encoders:
            le = prefit_label_encoders[col]
        else:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            label_encoders[col] = le

        # for subset_df in [x for x in [train_df, val_df, test_df] if x is not None]:
        #     subset_df.loc[:, col] = le.transform(subset_df[col].astype(str))

        # safer fallback using mapping — avoids unseen label errors
        classes = le.classes_
        class_to_index = {label: idx for idx, label in enumerate(classes)}

        for subset_df in [x for x in [train_df, val_df, test_df] if x is not None]:
            col_values = subset_df[col].astype(str)
            try:
                subset_df.loc[:, col] = le.transform(col_values)
            except ValueError:
                # map manually with fallback to -1 for unseen
                subset_df.loc[:, col] = col_values.map(class_to_index).fillna(len(class_to_index)).astype(int)

    # ========== Scaling ==========
    if prefit_scaler is None:
        scaler_obj.fit(df[num_columns])

    for subset_df in [x for x in [train_df, val_df, test_df] if x is not None]:
        subset_df[num_columns] = scaler_obj.transform(subset_df[num_columns])

    return ohe_encoder, label_encoders, scaler_obj, train_df, val_df, test_df


def setup_data(start_date, df):
    print(f'initial shape: {df.shape}')
    # Remove trips with invalid LOAD
    bad_trips = df[(df['load'] < 0) | (df['load'].isna())][['trip_id', 'transit_date']].drop_duplicates()
    bad_trips['marker'] = 1
    df = df.merge(bad_trips, on=['trip_id', 'transit_date'], how='left')
    df = df[df['marker'].isna()].drop(columns=['marker'])

    # Remove trips with extreme DELAY
    bad_trips = df[(df['delay'] < -900) | (df['delay'] > 900)][['trip_id', 'transit_date']].drop_duplicates()
    bad_trips['delay_marker'] = 1
    df = df.merge(bad_trips, on=['trip_id', 'transit_date'], how='left')
    df = df[df['delay_marker'].isna()].drop(columns=['delay_marker'])

    print(f'shapes after filtering: {df.shape}')

    df['transit_date'] = pd.to_datetime(df['transit_date'])

    # Filter dates
    dates = list(df.transit_date.dt.date)
    sorted_Dates = sorted(dates)
    end_date = sorted_Dates[-1]
    df = df[(df['transit_date'] >= pd.to_datetime(start_date)) & (df['transit_date'] <= pd.to_datetime(end_date))]

    # Filter by stop_sequence
    df = df.sort_values(by=['transit_date', 'arrival_time']).reset_index(drop=True)
    df = df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)
    sorted_df = []
    for ba in df['block_abbr'].dropna().unique():
        ba_df = df[df['block_abbr'] == ba]
        end_stop = ba_df['stop_sequence'].max()
        ba_df = ba_df[ba_df['stop_sequence'] != end_stop]
        sorted_df.append(ba_df)

    overall_df = pd.concat(sorted_df, ignore_index=True)
    overall_df = overall_df.drop(
        columns=[col for col in ['route_direction_name', 'route_id'] if col in overall_df.columns])

    overall_df['arrival_time'] = pd.to_datetime(overall_df['arrival_time'])
    overall_df['minute'] = overall_df['arrival_time'].dt.minute
    overall_df['minuteByWindow'] = overall_df['minute'] // 15
    overall_df['temp'] = overall_df['minuteByWindow'] + (overall_df['hour'] * 60 / 15)
    overall_df['time_window'] = np.floor(overall_df['temp']).astype(int)
    overall_df.drop(columns=['minute', 'minuteByWindow', 'temp'], inplace=True)

    print(f'overall shape: {overall_df.shape}')
    print(f'overall df: \n{overall_df.head(20)}')

    overall_df = overall_df.groupby(
        ['transit_date', 'route_id_dir', 'stop_id_original', 'time_window']
    ).agg({
        'trip_id': 'first',
        'block_abbr': 'first',
        'arrival_time': 'first',
        'year': 'first',
        'month': 'first',
        'day': 'first',
        'hour': 'first',
        'is_holiday': 'first',
        'is_school_break': 'first',
        'dayofweek': 'first',
        'stop_sequence': 'first',
        'darksky_temperature': 'mean',
        'darksky_humidity': 'mean',
        'darksky_precipitation_probability': 'mean',
        'sched_hdwy': 'max',
        'load': 'mean',
        'delay': 'mean'
    }).reset_index()
    # overall shape: (1467868, 22) before
    print(f'shape after grouping: {overall_df.shape}')
    # shape after grouping: (1448430, 21)
    print(f'overall df: \n{overall_df.head(20)}')
    # Sort and drop block/arrival_time like Spark
    overall_df = overall_df.sort_values(by=['block_abbr', 'arrival_time']).reset_index(drop=True)

    return overall_df


def remove_outliers(df, target_col='load', upper_percentile=99):
    threshold = np.percentile(df[target_col], upper_percentile)
    return df[df[target_col] <= threshold].copy()


def add_prev_trip_features(task_df, past=10):
    task_df = task_df.sort_values(by=['transit_date', 'arrival_time']).reset_index(drop=True)

    prev_max = {i + 1: [] for i in range(past)}
    prev_med = {i + 1: [] for i in range(past)}
    prev_time_diff = {i + 1: [] for i in range(past)}

    # Buffer of previous trips
    prev_trips = deque()

    for idx, row in task_df.iterrows():
        # Build occupancy list from prior trips
        prev_loads = [trip['load'] for trip in prev_trips]
        prev_arrivals = [trip['arrival_time'] for trip in prev_trips]

        for i in range(1, past + 1):
            if len(prev_loads) >= i:
                prev_max[i].append(max(prev_loads[-i:]))
                prev_med[i].append(np.median(prev_loads[-i:]))
                time_diff = (row['arrival_time'] - prev_arrivals[-i]).total_seconds() / 60.0
                prev_time_diff[i].append(time_diff)
            else:
                prev_max[i].append(0.0)
                prev_med[i].append(0.0)
                prev_time_diff[i].append(1440.0)  # 1 day gap default

        # Update the buffer
        prev_trips.append({'load': row['load'], 'arrival_time': row['arrival_time']})
        if len(prev_trips) > past:
            prev_trips.popleft()

    # Convert and assign
    for i in range(1, past + 1):
        task_df[f'Previous_Trip_{i}_Max_Occupancy'] = prev_max[i]
        task_df[f'Previous_Trip_{i}_Median_Occupancy'] = prev_med[i]
        task_df[f'Previous_Trip_{i}_TimeDifference'] = prev_time_diff[i]

    return task_df


if __name__ == '__main__':

    target = 'load'

    # Checking results with delay
    num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy']
    cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', 'time_window', 'delay']
    ohe_columns = ['dayofweek', 'is_holiday', 'is_school_break']  # 'route_id_dir',

    start_date = '2020-01-01'

    selected_route_df = pd.read_parquet(f'../Dataset/Ridership/Selected_data.parquet')

    unique_route_dir = list(selected_route_df.route_id_dir.unique())
    print(f'unique_route_dir: {unique_route_dir}\n{len(unique_route_dir)}')

    cleaned_entire_df = setup_data(start_date, selected_route_df)
    print(cleaned_entire_df.columns)
    print(f'shape: {cleaned_entire_df.shape}')
    print(cleaned_entire_df.head(10))
    df = cleaned_entire_df
    df['arrival_time'] = pd.to_datetime(df['arrival_time']).dt.tz_localize('UTC').dt.tz_convert('US/Central')
    df['new_hour'] = df['arrival_time'].dt.hour
    print(df[['arrival_time', 'new_hour', 'hour']].head())
    cleaned_entire_df.to_parquet(f'../Dataset/Ridership/cleaned_entire_df_correctedTimezone.parquet', index=False)

    cleaned_entire_df = df

    unique_route_dir = list(cleaned_entire_df.route_id_dir.unique())
    print(f'unique_route_dir: {unique_route_dir}\n{len(unique_route_dir)}')
    cleaned_entire_df.to_parquet(f'../Dataset/Ridership/cleaned_selected_data_MEAN.parquet', index=False)

    ohe, label_encs, scaler, _, _, _ = prepare_linklevel(
        df=cleaned_entire_df,
        train_dates=None,  # Just to pass through all rows
        val_dates=None,
        test_dates=None,
        cat_columns=cat_columns,
        num_columns=num_columns,
        ohe_columns=ohe_columns,
        scaler='standard',
        time_feature_used='transit_date'
    )

    print(f'global done')
    print(f'all columns: {cleaned_entire_df.columns}')

    # cleaned_entire_df= pd.read_csv(f'../data/mtl_data/cleaned_selected_data.csv', low_memory=False)

    unique_route_dir = list(cleaned_entire_df.route_id_dir.unique())
    print(f'unique_route_dir: {unique_route_dir}\n{len(unique_route_dir)}')

    for each_route_dir in unique_route_dir:
        cleaned_df = cleaned_entire_df[cleaned_entire_df['route_id_dir'] == each_route_dir]
        print(f'\n\n************ TASK: {each_route_dir}, SHAPE BEFORE: {cleaned_df.shape}')
        print(cleaned_df.head(5))
        # exit(0)

        # Ensure datetime types
        cleaned_df['transit_date'] = pd.to_datetime(cleaned_df['transit_date'])
        cleaned_df['arrival_time'] = pd.to_datetime(cleaned_df['arrival_time'])

        # Add features
        task_df = add_prev_trip_features(cleaned_df, past=10)

        # Save back if needed
        # task_df.to_parquet(f'../data/mtl_data/{each_route_dir}_with_prev_New.parquet', index=False)
        cleaned_df = task_df

        '''check if any columns is NAN'''
        columns = list(cleaned_df.columns)
        for col in columns:
            if col in cleaned_df.columns:
                if cleaned_df[col].isnull().sum() >= 0:
                    print(f'IsNAN: {cleaned_df[col].isnull().sum()}')

        ### ✅ NEW: date splitting based on sample distribution
        sample_counts_per_date = cleaned_df['transit_date'].value_counts().sort_index()
        cum_counts = sample_counts_per_date.cumsum()
        total_samples = cum_counts.iloc[-1]
        print(f'Total samples: {total_samples}')

        train_range = sample_counts_per_date[cum_counts <= 0.7 * total_samples].index
        val_range = sample_counts_per_date[
            (cum_counts > 0.7 * total_samples) & (cum_counts <= 0.85 * total_samples)].index
        test_range = sample_counts_per_date[cum_counts > 0.85 * total_samples].index

        if len(train_range) > 0:
            train_dates = (train_range[0].date(), train_range[-1].date())
        else:
            train_dates = (None, None)

        if len(val_range) > 0:
            val_dates = (val_range[0].date(), val_range[-1].date())
        else:
            val_dates = (None, None)

        if len(test_range) > 0:
            test_dates = (test_range[0].date(), test_range[-1].date())
        else:
            test_dates = (None, None)
        print(f'train_dates: {train_dates}')
        print(f'val_dates: {val_dates}')
        print(f'test_dates: {test_dates}')
        _, _, _, train_df, val_df, test_df = prepare_linklevel(
            df=cleaned_df,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            cat_columns=cat_columns,
            num_columns=num_columns,
            ohe_columns=ohe_columns,
            scaler='standard',
            prefit_ohe=ohe,
            prefit_label_encoders=label_encs,
            prefit_scaler=scaler,
            time_feature_used='transit_date'
        )

        for df_ in [train_df, val_df, test_df]:
            df_.drop(
                columns=[col for col in ['transit_date', 'trip_id', 'arrival_time', 'y_class', 'y_class_2']
                         if col in df_.columns], inplace=True)

        # Arrange columns (target first)
        target = 'load'
        train_df = train_df[[target] + [col for col in train_df.columns if col != target]]
        val_df = val_df[[target] + [col for col in val_df.columns if col != target]]
        test_df = test_df[[target] + [col for col in test_df.columns if col != target]]

        print(f'train_shape: {train_df.shape}, val_shape: {val_df.shape}, test_shape: {test_df.shape}')
        print(f'train columns : {train_df.columns}')

        train_df.to_parquet(f'../Dataset/Ridership/mtl_data/{each_route_dir}_TRAIN_MEAN.parquet', index=False)
        val_df.to_parquet(f'../Dataset/Ridership/mtl_data/{each_route_dir}_VAL_MEAN.parquet', index=False)
        test_df.to_parquet(f'../Dataset/Ridership/mtl_data/{each_route_dir}_TEST_MEAN.parquet', index=False)
