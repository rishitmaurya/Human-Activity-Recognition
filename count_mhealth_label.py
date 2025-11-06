import pandas as pd
import openpyxl

# Load your dataset
df = pd.read_csv("combined_mhealth.csv")

# Count and percentage
label_counts = df['label'].value_counts().sort_index()
label_percent = (label_counts / len(df)) * 100

# Create mapping table
data = {
    "Label": list(range(13)),  # 0 to 11
    "Label ID": ["L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","L12","L13"],
    "Activity Description": [
        "Standing still",
        "Sitting and relaxing",
        "Lying down",
        "Walking",
        "Climbing stairs",
        "Waist bends forward",
        "Frontal elevation of arms",
        "Knees bending (crouching)",
        "Cycling",
        "Jogging",
        "Running",
        "Jump front & back",
        "Not mentioned"
    ],
    "Duration / Repetitions": [
        "1 min","1 min","1 min","1 min","1 min",
        "20 repetitions","20 repetitions","20 repetitions",
        "1 min","1 min","1 min","20 repetitions","Not mentioned"
    ]
}

df_labels = pd.DataFrame(data)

# Add counts & percentages
df_labels["Count"] = df_labels["Label"].map(label_counts)
df_labels["Percentage (%)"] = df_labels["Label"].map(label_percent).round(4)

# Display table nicely
print(df_labels.to_string(index=False))

# Total rows in dataset
print("\nTotal Samples:", len(df))

# Export the final table to Excel
df_labels.to_excel("label_summary_mhealth.xlsx", index=False)

print("\nExcel file saved as: label_summary.xlsx")
