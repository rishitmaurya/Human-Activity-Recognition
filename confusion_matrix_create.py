import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define the activity labels in order
activities = [
    "Horizontal arm wave", "High arm wave", "Two hand wave", "Catch Cap", 
    "High throw", "Draw X", "Draw Tick", "Toss Paper", "Forward Kick", 
    "Side Kick", "Take Umbrella", "Bend", "Hand Clap", "Walk", 
    "Phone Call", "Drink", "Sit down", "Stand up"
]

# 2. Initialize an 18x18 matrix with zeros
# Standard orientation: Rows = True, Cols = Predicted
cm_standard = np.zeros((18, 18), dtype=int)

# 3. Populate the diagonal (Correct Predictions) based on the image data
# Values: Action 0-14, 16-17 are perfect. Action 15 (Drink) has 3 correct.
diagonals = [3,4,5,4,3,2,2,5,4,5,5,4,5,4,5,2,4,5]
for i in range(18):
    cm_standard[i, i] = diagonals[i]

# 4. Add the specific misclassification from the image
# Action 15 (Drink) was misclassified once as Action 4 (High throw)
# Format: cm[True_Index, Predicted_Index]
cm_standard[0, 5] = 1
cm_standard[4, 6] = 2
cm_standard[5, 0] = 1
cm_standard[5, 12] = 1
cm_standard[6, 0] = 2
cm_standard[6, 5] = 1
cm_standard[15, 0] = 1
cm_standard[15, 4] = 1

# 5. Swap axes as requested: X label True, Y label Predicted
# We transpose the matrix so that Predicted is on the Y-axis (Rows) 
# and True is on the X-axis (Columns)
# cm_swapped = cm_standard.T

# 6. Plotting
plt.figure(figsize=(18, 16))
sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues', 
            xticklabels=activities, yticklabels=activities,
            cbar_kws={'label': 'Number of Samples'})

# Setting labels as specifically requested
plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
plt.ylabel('True Labels', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix on KARD using BiLSTM', fontsize=14, pad=20)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()