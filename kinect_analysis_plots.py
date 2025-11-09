"""
kinect_analysis_plots.py

Creates multiple 2D and 3D visualizations for a Kinect CSV dataset.

Features included (saved to `analysis_outputs`):
 1. Colored Activity distribution bar
 2. Horizontal Person contribution bar (less congested)
 3. Stacked bar: Person vs Activity (legend outside, totals labeled above bars)
 4. 3D scatter of a selected joint (SpineBase_x,y,z) colored by Person or Activity
 5. 3D trajectory for Activity_to_compare for selected Persons (timestamp -> 3D path)
 6. 3D PCA of selected joint coordinates (dimensionality reduction of joints) colored by Activity

Usage:
  - Place this script next to your `combined_kinect_dataset.csv` or update csv_path
  - Run: python kinect_analysis_plots.py

Requirements: pandas, numpy, matplotlib, scikit-learn
"""

import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib import cm
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# -------------------------- User settings ---------------------------------
csv_path = "combined_kinect_dataset.csv"  # change if needed
output_dir = "analysis_outputss"
os.makedirs(output_dir, exist_ok=True)

# Activity and persons requested by user
activity_to_compare = "bending"
persons_requested = [f"Person_{i}" for i in range(1, 31)]  # Person_1 .. Person_30

# Joint to use for simple 3D scatter / trajectory (you can change)
joint_prefix_for_3d = "SpineBase"  # will use SpineBase_x/y/z

# List of joint prefixes to use for PCA (will take their x,y,z columns)
joints_for_pca = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
]
# ---------------------------------------------------------------------------

print("Reading CSV:", csv_path)
df = pd.read_csv(csv_path)
print("Dataset loaded with shape:", df.shape)

# Normalize Person column strings (strip whitespace)
df['Person'] = df['Person'].astype(str).str.strip()

# Filter persons to those present
persons_available = sorted(df['Person'].unique())
persons = [p for p in persons_requested if p in persons_available]
if not persons:
    # fallback: use top 6 persons by count
    persons = df['Person'].value_counts().index[:6].tolist()
    print("No requested persons found—falling back to top persons:", persons)
else:
    print(f"Using {len(persons)} requested persons found in data")

# Ensure activity exists
activities_available = sorted(df['Activity'].unique())
if activity_to_compare not in activities_available:
    print(f"Warning: activity '{activity_to_compare}' not found. Available activities:\n  ", activities_available)

# -------------------- Helpers for colors & saving --------------------------

def get_color_map(categories, cmap_name='tab20'):
    cmap = cm.get_cmap(cmap_name)
    n = len(categories)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    return dict(zip(categories, colors))


def save_fig(fig, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('Saved:', path)

# -------------------- 1) Activity distribution (colored) ------------------
activity_counts = df['Activity'].value_counts()
activity_colors = get_color_map(activity_counts.index, cmap_name='tab10')

fig, ax = plt.subplots(figsize=(10, 6))
activity_counts.plot(kind='bar', ax=ax, color=[activity_colors[a] for a in activity_counts.index], edgecolor='black')
ax.set_xlabel('Activity', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Activity Distribution', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, xytext=(0, 4), textcoords='offset points')
save_fig(fig, '01_activity_distribution_colored.png')

# -------------------- 2) Person contribution (horizontal bar) -------------
person_counts = df['Person'].value_counts()
person_colors = get_color_map(person_counts.index, cmap_name='tab20')

fig, ax = plt.subplots(figsize=(10, max(6, len(person_counts) * 0.25)))
person_counts.plot(kind='barh', ax=ax, color=[person_colors[p] for p in person_counts.index], edgecolor='black')
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('Person', fontsize=12)
ax.set_title('Participant Data Contribution (horizontal)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, p in enumerate(ax.patches):
    ax.annotate(f"{int(p.get_width())}", (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=9, xytext=(4, 0), textcoords='offset points')
save_fig(fig, '02_person_distribution_horizontal.png')

# -------------------- 3) Stacked Bar: Person vs Activity ------------------
person_activity_counts = df.groupby(['Person', 'Activity']).size().unstack(fill_value=0)
# limit to persons we decided earlier (helps avoid label congestion)
person_activity_counts = person_activity_counts.reindex(persons).fillna(0)

fig, ax = plt.subplots(figsize=(14, 7))
activities = person_activity_counts.columns.tolist()
activity_color_map = get_color_map(activities, cmap_name='tab20')

bottom = np.zeros(len(person_activity_counts))
indices = np.arange(len(person_activity_counts))
for act in activities:
    vals = person_activity_counts[act].values
    ax.bar(indices, vals, bottom=bottom, label=act, color=activity_color_map[act], edgecolor='white')
    bottom = bottom + vals

ax.set_xticks(indices)
ax.set_xticklabels(person_activity_counts.index, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_xlabel('Person', fontsize=12)
ax.set_title('Stacked Bar: Person vs Activity', fontsize=16, fontweight='bold')
ax.grid(axis='y', linestyle='--', linewidth=0.4)

# Legend outside
ax.legend(title='Activity', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

# Annotate totals above each bar (outside the graph)
totals = person_activity_counts.sum(axis=1).values
for idx, total in enumerate(totals):
    ax.annotate(str(int(total)), xy=(idx, total), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Expand top margin so labels are outside
fig.subplots_adjust(top=0.88, right=0.78)
save_fig(fig, '03_person_activity_stacked_with_totals.png')

# -------------------- 4) 3D scatter of a joint (SpineBase_x,y,z) ----------
joint = joint_prefix_for_3d
cols = [f"{joint}_x", f"{joint}_y", f"{joint}_z"]
if not all(c in df.columns for c in cols):
    print(f"Joint columns {cols} not found in CSV. Skipping 3D scatter.")
else:
    # sample to reduce plotting time if dataset is large
    sample_df = df[df['Person'].isin(persons)].copy()
    # color by activity
    unique_acts = sample_df['Activity'].unique()
    act_colormap = get_color_map(unique_acts, cmap_name='tab10')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for act in unique_acts:
        sub = sample_df[sample_df['Activity'] == act]
        ax.scatter(sub[cols[0]], sub[cols[1]], sub[cols[2]], label=act, alpha=0.7, s=18, color=act_colormap[act])
    ax.set_xlabel(f'{joint}_x')
    ax.set_ylabel(f'{joint}_y')
    ax.set_zlabel(f'{joint}_z')
    ax.set_title(f'3D Scatter of {joint} positions colored by Activity', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    save_fig(fig, f'04_3d_scatter_{joint}_by_activity.png')

# -------------------- 5) 3D trajectory for activity_to_compare ------------
act = activity_to_compare
if act not in activities_available:
    print(f"Activity '{act}' not present; skipping 3D trajectory.")
else:
    traj_df = df[df['Activity'] == act].copy()
    traj_df = traj_df[traj_df['Person'].isin(persons)]
    # attempt to sort by timestamp if present
    time_col = None
    for candidate in ['timestamp', 'time', 'Timestamp']:
        if candidate in traj_df.columns:
            time_col = candidate
            break
    if time_col is not None:
        traj_df = traj_df.sort_values(time_col)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # draw separate trajectories per person
    person_colors = get_color_map(traj_df['Person'].unique(), cmap_name='tab20')
    for p in traj_df['Person'].unique():
        sub = traj_df[traj_df['Person'] == p]
        if cols[0] not in sub.columns:
            continue
        ax.plot(sub[cols[0]].values, sub[cols[1]].values, sub[cols[2]].values, label=p, linewidth=1.5, alpha=0.9, color=person_colors[p])
        # mark start and end
        if len(sub) > 0:
            ax.scatter(sub[cols[0]].iloc[0], sub[cols[1]].iloc[0], sub[cols[2]].iloc[0], marker='o', s=30, color=person_colors[p])
            ax.scatter(sub[cols[0]].iloc[-1], sub[cols[1]].iloc[-1], sub[cols[2]].iloc[-1], marker='X', s=30, color=person_colors[p])

    ax.set_xlabel(f'{joint}_x')
    ax.set_ylabel(f'{joint}_y')
    ax.set_zlabel(f'{joint}_z')
    ax.set_title(f'3D Trajectories of {joint} for activity: {act}', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    save_fig(fig, f'05_3d_trajectory_{joint}_{act}.png')

# -------------------- 6) PCA 3D of selected joint coordinates --------------
# Build a features matrix using the listed joints (x,y,z for each joint)
feature_cols = []
for jp in joints_for_pca:
    for axis in ['_x', '_y', '_z']:
        feature_cols.append(jp + axis)

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print('Some PCA feature columns missing; skipping PCA 3D. Missing cols sample:', missing[:6])
else:
    pca_df = df.dropna(subset=feature_cols + ['Activity']).copy()
    X = pca_df[feature_cols].values
    # do a quick standardization (mean centering)
    X = X - X.mean(axis=0)
    pca = PCA(n_components=3)
    Xp = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    acts = pca_df['Activity'].unique()
    act_colors = get_color_map(acts, cmap_name='tab10')
    for a in acts:
        sel = pca_df['Activity'] == a
        ax.scatter(Xp[sel, 0], Xp[sel, 1], Xp[sel, 2], label=a, alpha=0.6, s=10, color=act_colors[a])
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title('PCA (3D) of joint coordinates colored by Activity', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    save_fig(fig, '06_pca_3d_joints_by_activity.png')

print('\nAll plots completed and saved to folder:', output_dir)
