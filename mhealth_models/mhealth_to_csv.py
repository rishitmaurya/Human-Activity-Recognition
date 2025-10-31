"""
Script to combine all .log files from mhealth_dataset/MHEALTHDATASET into one CSV file.
Each .log file contains tab/space-separated sensor readings with 24 columns as described below.

Output: combined_mhealth.csv
"""

import os
import glob
import pandas as pd

# ---------------------- Configuration ----------------------
DATA_DIR = './mhealth_dataset/MHEALTHDATASET'  # Folder containing .log files
OUTPUT_FILE = 'combined_mhealth.csv'

# Column names as per the provided specification
COLUMNS = [
    'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
    'ecg_lead_1', 'ecg_lead_2',
    'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
    'gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z',
    'mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z',
    'acc_arm_x', 'acc_arm_y', 'acc_arm_z',
    'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
    'mag_arm_x', 'mag_arm_y', 'mag_arm_z',
    'label'
]

# ---------------------- Main Script ----------------------
def combine_logs_to_csv(data_dir=DATA_DIR, output_file=OUTPUT_FILE):
    log_files = sorted(glob.glob(os.path.join(data_dir, 'mHealth_subject*.log')))

    if not log_files:
        print(f'No .log files found in {data_dir}')
        return

    all_dataframes = []

    for file_path in log_files:
        print(f'Reading {file_path} ...')
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
            if df.shape[1] != len(COLUMNS):
                print(f'Warning: {file_path} has {df.shape[1]} columns, expected {len(COLUMNS)}. Skipping.')
                continue
            df.columns = COLUMNS
            # df['subject'] = os.path.basename(file_path).split('.')[0]  # add subject ID column
            all_dataframes.append(df)
        except Exception as e:
            print(f'Error reading {file_path}: {e}')

    if not all_dataframes:
        print('No valid data loaded. Exiting.')
        return

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f'Total combined shape: {combined_df.shape}')

    combined_df.to_csv(output_file, index=False)
    print(f'Combined CSV saved as: {output_file}')


if __name__ == '__main__':
    combine_logs_to_csv()
