import pandas as pd
import numpy as np

# =======================
# Load KARD dataset
# =======================
df = pd.read_csv('KARD_all_realworld.csv')

# =======================
# Map action_id to action_name
# =======================
action_map = {
    1: "Horizontal arm wave",
    2: "High arm wave",
    3: "Two hand wave",
    4: "Catch Cap",
    5: "High throw",
    6: "Draw X",
    7: "Draw Tick",
    8: "Toss Paper",
    9: "Forward Kick",
    10: "Side Kick",
    11: "Take Umbrella",
    12: "Bend",
    13: "Hand Clap",
    14: "Walk",
    15: "Phone Call",
    16: "Drink",
    17: "Sit down",
    18: "Stand up"
}

df['Activity'] = df['action_id'].map(action_map)

# =======================
# Dataset Summary
# =======================
num_subjects = df['subject_id'].nunique()
num_activities = df['Activity'].nunique()
num_samples = len(df)
num_joints = df['joint_name'].nunique()
sampling_rate = 30  # Assuming Kinect v2
avg_seq_length = df.groupby(['subject_id','repetition','action_id']).size().mean()

dataset_summary = pd.DataFrame({
    "Dataset Name": ["KARD"],
    "Sensor": ["Kinect v2"],
    "Number of Subjects": [num_subjects],
    "Number of Activities": [num_activities],
    "Number of Samples": [num_samples],
    "Sampling Rate (Hz)": [sampling_rate],
    "Avg Sequence Length (Frames)": [avg_seq_length],
    "Number of Skeletal Joints": [num_joints],
    "Joint Representation": ["3D Cartesian (x,y,z)"],
    "Skeleton Format": ["Kinect v2"],
    "Train/Val/Test Split (%)": ["70/15/15"],
    "Evaluation Protocol": ["Subject-Independent Split"],
    "Target Application": ["Human Activity Recognition"],
    "Benchmark Usage": ["Yes"]
})

print("\n=== Dataset Summary ===\n")
print(dataset_summary.to_string(index=False))

# =======================
# Activity Distribution
# =======================
activity_distribution = df.groupby('Activity')['frame'].count().reset_index()
activity_distribution.rename(columns={'frame':'Num Samples'}, inplace=True)

print("\n=== Activity Distribution ===\n")
print(activity_distribution.to_string(index=False))

# =======================
# Joint Statistics
# =======================
joint_stats = df.groupby('joint_name').agg({
    'x':['min','max','mean','std'],
    'y':['min','max','mean','std'],
    'z':['min','max','mean','std']
}).reset_index()

# Flatten multi-level columns
joint_stats.columns = ['Joint', 'x_min','x_max','x_mean','x_std',
                       'y_min','y_max','y_mean','y_std',
                       'z_min','z_max','z_mean','z_std']

print("\n=== Joint Statistics ===\n")
print(joint_stats.to_string(index=False))

# =======================
# Per-Subject Summary
# =======================
subject_summary = df.groupby('subject_id').agg(
    Num_Sequences=('repetition','nunique'),
    Num_Samples=('frame','count')
).reset_index()
subject_summary.rename(columns={'subject_id':'Person'}, inplace=True)

print("\n=== Per-Subject Summary ===\n")
print(subject_summary.to_string(index=False))

# =======================
# Activity Duration (in seconds)
# =======================
# Duration = number of frames / sampling rate
activity_duration = df.groupby('Activity').agg(
    Num_Samples=('frame','count')
).reset_index()
activity_duration['Duration_sec'] = activity_duration['Num_Samples'] / sampling_rate

print("\n=== Activity Duration (Seconds) ===\n")
print(activity_duration.to_string(index=False))





# for custom dataset 
# import pandas as pd
# import numpy as np

# # =========================
# # Load Dataset
# # =========================
# dataset_path = "combined_kinect_dataset.csv"  # Update this path
# df = pd.read_csv(dataset_path)

# # =========================
# # 1. Dataset Summary Table
# # =========================
# dataset_name = "Custom Skeletal Joint Dataset"
# sensor_type = "Kinect v2"
# num_subjects = df['Person'].nunique()
# num_activities = df['Activity'].nunique()
# num_samples = len(df)
# sampling_rate_hz = round(1 / df['timestamp'].diff().median(), 2)
# sequence_length_avg = df.groupby(['Person', 'Activity']).size().mean()

# # Identify joints
# joint_columns = [col for col in df.columns if '_' in col and col.split('_')[-1] in ['x','y','z']]
# num_joints = int(len(joint_columns)/3)

# dataset_summary = pd.DataFrame([{
#     "Dataset Name": dataset_name,
#     "Sensor": sensor_type,
#     "Number of Subjects": num_subjects,
#     "Number of Activities": num_activities,
#     "Number of Samples": num_samples,
#     "Sampling Rate (Hz)": sampling_rate_hz,
#     "Avg Sequence Length (Frames)": round(sequence_length_avg,2),
#     "Number of Skeletal Joints": num_joints,
#     "Joint Representation": "3D Cartesian (x,y,z)",
#     "Skeleton Format": "Kinect v2",
#     "Train/Val/Test Split (%)": "70/15/15",
#     "Evaluation Protocol": "Subject-Independent Split",
#     "Target Application": "Human Activity Recognition for Robot Imitation",
#     "Benchmark Usage": "Yes"
# }])

# # =========================
# # 2. Activity Distribution Table
# # =========================
# activity_distribution = df['Activity'].value_counts().reset_index()
# activity_distribution.columns = ['Activity', 'Num Samples']

# # =========================
# # 3. Joint Statistics Table
# # =========================
# joint_stats_list = []
# for joint in set([col.rsplit('_',1)[0] for col in joint_columns]):
#     stats = {}
#     stats["Joint"] = joint
#     for axis in ['x','y','z']:
#         col_name = f"{joint}_{axis}"
#         stats[f"{axis}_min"] = df[col_name].min()
#         stats[f"{axis}_max"] = df[col_name].max()
#         stats[f"{axis}_mean"] = df[col_name].mean()
#         stats[f"{axis}_std"] = df[col_name].std()
#     joint_stats_list.append(stats)

# joint_stats_df = pd.DataFrame(joint_stats_list)

# # =========================
# # 4. Per-Subject Summary Table
# # =========================
# subject_summary = df.groupby('Person').agg(
#     Num_Sequences=('Activity','nunique'),
#     Num_Samples=('timestamp','count')
# ).reset_index()

# # =========================
# # 5. Activity Duration Table
# # =========================
# activity_duration = df.groupby('Activity').agg(
#     Num_Samples=('timestamp','count'),
#     Duration_sec=('timestamp', lambda x: x.max() - x.min())
# ).reset_index()


# # =========================
# # Display Tables
# # =========================
# print("=== Dataset Summary Table ===")
# print(dataset_summary.to_string(index=False))

# print("\n=== Activity Distribution Table ===")
# print(activity_distribution.to_string(index=False))

# print("\n=== Joint Statistics Table (Min, Max, Mean, Std per Axis) ===")
# print(joint_stats_df.to_string(index=False))

# print("\n=== Per-Subject Summary Table ===")
# print(subject_summary.to_string(index=False))

# print("\n=== Activity Duration Table (Seconds) ===")
# print(activity_duration.to_string(index=False))

# # =========================
# # Optional: Save Tables as CSV
# # =========================
# dataset_summary.to_csv("dataset_summary_table.csv", index=False)
# activity_distribution.to_csv("activity_distribution_table.csv", index=False)
# joint_stats_df.to_csv("joint_statistics_table.csv", index=False)
# subject_summary.to_csv("subject_summary_table.csv", index=False)
# activity_duration.to_csv("activity_duration_table.csv", index=False)

