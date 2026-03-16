import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the dataset
df = pd.read_csv('dataset_wearable/combined_har_dataset.csv')

# Function to extract numeric part from person_id for proper sorting
def extract_person_number(person_id):
    numbers = re.findall(r'\d+', person_id)
    return int(numbers[0]) if numbers else 0

# Sort person_ids numerically
unique_persons = sorted(df['person_id'].unique(), key=extract_person_number)

# Create a cross-tabulation of person_id and activity with proper sorting
person_activity_counts = pd.crosstab(df['person_id'], df['activity'])
person_activity_counts = person_activity_counts.reindex(unique_persons)

# Get the total samples per person
person_totals = person_activity_counts.sum(axis=1)

# Create the plot with more width between bars
fig, ax = plt.subplots(figsize=(18, 10))  # Increased figure size

# Set up bar positions with more spacing
x = np.arange(len(person_totals.index))
width = 0.6  # Reduced width to create more space between bars

# Create stacked bar chart
bottom = np.zeros(len(person_totals.index))
colors = plt.cm.Set3(np.linspace(0, 1, len(person_activity_counts.columns)))

# Plot each activity as a segment of the bar
for i, (activity, color) in enumerate(zip(person_activity_counts.columns, colors)):
    values = person_activity_counts[activity].values
    bars = ax.bar(x, values, width, label=activity, bottom=bottom, color=color, edgecolor='white', linewidth=0.5)
    bottom += values

# Add total samples above each bar with better positioning
max_height = person_totals.max()
for i, (person, total) in enumerate(person_totals.items()):
    # Calculate y-position (above the bar with more space)
    y_pos = total + (max_height * 0.03)  # 3% above the highest bar
    
    # Add text with rotation to prevent overlap
    ax.text(i, y_pos, f'Total: {int(total)}', 
            ha='center', va='bottom', 
            fontweight='bold', fontsize=9,
            rotation=45,  # Rotate text to prevent overlap
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

# Customize the plot
ax.set_xlabel('Person', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Activities per Person', fontsize=16, fontweight='bold')

# Set x-tick labels with proper sorting
ax.set_xticks(x)
ax.set_xticklabels(person_totals.index, rotation=45, ha='right', fontsize=10)

# Add legend with better placement
ax.legend(title='Activities', bbox_to_anchor=(1, 1), loc='upper left', fontsize=9, title_fontsize=10)

# Add grid for better readability (only horizontal lines)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Set y-axis limit to accommodate the rotated text labels
ax.set_ylim(0, max_height * 1.25)  # Increased top margin

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()