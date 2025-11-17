import os
import pandas as pd

# Path to the main dataset directory
dataset_dir = "Dataset"   # change if needed

# List to collect data from all CSV files
all_data = []

# Loop through each Person folder
for person_folder in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_folder)

    if os.path.isdir(person_path):
        person_id = person_folder  # e.g., "Person_1"

        # Loop through each activity file inside the person's folder
        for file in os.listdir(person_path):
            if file.endswith(".csv"):
                activity_name = file.replace(".csv", "")  # e.g., "walking"
                file_path = os.path.join(person_path, file)

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Add identifying columns
                df["Person"] = person_id
                df["Activity"] = activity_name

                # Append to the global list
                all_data.append(df)

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

# Save final combined CSV
combined_df.to_csv("combined_kinect_dataset.csv", index=False)

print("Dataset combined successfully. Saved as combined_kinect_dataset.csv")
