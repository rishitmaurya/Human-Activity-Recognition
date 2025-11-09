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


# ================= Improved Vertical Stacked Bar Chart ==================
person_activity_counts = df.groupby(["Person", "Activity"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))   # wider figure
colors = plt.cm.tab20.colors[:len(person_activity_counts.columns)]

person_activity_counts.plot(kind="bar", stacked=True, edgecolor="black", color=colors, ax=ax)

ax.set_xlabel("Person", fontsize=14)
ax.set_ylabel("Number of Samples", fontsize=14)
ax.set_title("Stacked Bar Chart: Person vs Activity Distribution", fontsize=16, fontweight='bold')

ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=12)  # rotate and right-align
ax.grid(axis='y', linestyle='--', linewidth=0.4)

# Legend outside
ax.legend(title="Activity", bbox_to_anchor=(1, 1), loc='upper left')

plt.subplots_adjust(bottom=0.25)   # add extra space below labels
plt.tight_layout()
plt.savefig(f"{output_dir}/person_activity_stacked_bar_colored.png", bbox_inches="tight")
plt.close()

print("Updated stacked bar chart saved.")


