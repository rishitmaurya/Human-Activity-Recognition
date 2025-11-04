import os
import pandas as pd
import numpy as np


# --------- CONFIG ---------
ROOT_DIR = "KARD"   # path to your dataset root folder
OUTPUT_CSV = "KARD_all_realworld.csv"

# Skeleton joint order as per KARD specification
JOINT_NAMES = [
    "Head", "Neck", "Right Shoulder", "Right Elbow", "Right Hand",
    "Left Shoulder", "Left Elbow", "Left Hand", "Torso",
    "Right Hip", "Right Knee", "Right Foot",
    "Left Hip", "Left Knee", "Left Foot"
]

# --------- FUNCTION TO LOAD ONE FILE ---------
def load_realworld_file(file_path):
    """
    Loads a single realworld.txt file and returns a DataFrame
    with columns [action_id, subject_id, repetition, frame, joint_name, x, y, z].
    """
    # Extract IDs from filename (aAA_sSS_eNN_realworld.txt)
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    action_id = int(parts[0][1:])   # e.g., a04 -> 4
    subject_id = int(parts[1][1:])  # e.g., s03 -> 3
    repetition = int(parts[2][1:])  # e.g., e02 -> 2

    # Load numeric data
    data = np.loadtxt(file_path)
    num_frames = len(data) // 15

    # Reshape to (frames, joints, 3)
    data = data.reshape(num_frames, 15, 3)

    # Build DataFrame
    rows = []
    for f_idx in range(num_frames):
        for j_idx, joint in enumerate(JOINT_NAMES):
            x, y, z = data[f_idx, j_idx]
            rows.append([action_id, subject_id, repetition, f_idx+1, joint, x, y, z])

    df = pd.DataFrame(rows, columns=[
        "action_id", "subject_id", "repetition", "frame", "joint_name", "x", "y", "z"
    ])
    return df

# --------- MAIN MERGE LOOP ---------
def main():
    all_dfs = []

    # Walk through all subfolders and find *realworld.txt files
    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            if f.endswith("_realworld.txt"):
                file_path = os.path.join(root, f)
                try:
                    df = load_realworld_file(file_path)
                    all_dfs.append(df)
                except Exception as e:
                    print(f" Error reading {file_path}: {e}")

    # Combine all DataFrames
    if not all_dfs:
        print("No realworld.txt files found!")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Save as CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f" Merged CSV saved as: {OUTPUT_CSV}")
    print(f"Total rows: {len(final_df)}")

if __name__ == "__main__":
    main()
