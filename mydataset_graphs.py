import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset

csv_path = "combined_kinect_dataset.csv"  # change if needed
df = pd.read_csv(csv_path)

# Ensure output directory
output_dir = "analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

print("Dataset Loaded:", df.shape)
print("Columns:", len(df.columns))


# Activity Distribution

activity_counts = df["Activity"].value_counts()

plt.figure()
activity_counts.plot(kind="bar")
plt.xlabel("Activity")
plt.ylabel("Number of Samples")
plt.title("Activity Distribution")
plt.tight_layout()
plt.savefig(f"{output_dir}/activity_distribution.png")
plt.close()


# Person Participation Distribution

person_counts = df["Person"].value_counts()

plt.figure()
person_counts.plot(kind="bar")
plt.xlabel("Person")
plt.ylabel("Number of Samples")
plt.title("Participant Data Contribution")
plt.tight_layout()
plt.savefig(f"{output_dir}/person_distribution.png")
plt.close()

print("All analysis plots saved to:", output_dir)




# Create count matrix (Activity × Person)
count_matrix = pd.crosstab(df["Activity"], df["Person"])

# Convert to array
data = count_matrix.values

plt.figure(figsize=(10, 6))
plt.imshow(data, aspect="auto")
plt.colorbar(label="Number of Samples")

# X and Y ticks
plt.xticks(np.arange(len(count_matrix.columns)), count_matrix.columns, rotation=45)
plt.yticks(np.arange(len(count_matrix.index)), count_matrix.index)

plt.xlabel("Person")
plt.ylabel("Activity")
plt.title("Heatmap of Person vs Activity Sample Distribution")

plt.tight_layout()
plt.savefig(f"{output_dir}/person_activity_heatmap.png")
plt.close()

print("Saved:", f"{output_dir}/person_activity_heatmap.png")