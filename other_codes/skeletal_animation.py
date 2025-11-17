import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.animation import PillowWriter

# ------------------------------------
# Load dataset & filter for one person and one activity
# ------------------------------------
csv_path = "combined_kinect_dataset.csv"
df = pd.read_csv(csv_path)

PERSON = "Person_1"
ACTIVITY = "running"

df = df[(df["Person"] == PERSON) & (df["Activity"] == ACTIVITY)]
df = df.reset_index(drop=True)

print("Frames loaded:", len(df))


# ------------------------------------
# Define Joint Groups (Kinect Layout)
# ------------------------------------
joints = [
    "Head", "Neck", "SpineShoulder", "SpineMid", "SpineBase",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight"
]

# Skeleton Connection Edges
edges = [
    ("Head", "Neck"), ("Neck", "SpineShoulder"), ("SpineShoulder", "SpineMid"), ("SpineMid", "SpineBase"),
    ("SpineShoulder", "ShoulderLeft"), ("ShoulderLeft", "ElbowLeft"), ("ElbowLeft", "WristLeft"), ("WristLeft", "HandLeft"),
    ("SpineShoulder", "ShoulderRight"), ("ShoulderRight", "ElbowRight"), ("ElbowRight", "WristRight"), ("WristRight", "HandRight"),
    ("SpineBase", "HipLeft"), ("HipLeft", "KneeLeft"), ("KneeLeft", "AnkleLeft"), ("AnkleLeft", "FootLeft"),
    ("SpineBase", "HipRight"), ("HipRight", "KneeRight"), ("KneeRight", "AnkleRight"), ("AnkleRight", "FootRight")
]

# ------------------------------------
# Extract skeleton coordinates frame-wise
# ------------------------------------
def get_frame_xyz(frame_index):
    row = df.iloc[frame_index]
    coords = {}
    for j in joints:
        coords[j] = np.array([row[f"{j}_x"], row[f"{j}_z"], row[f"{j}_y"]])
    return coords

# ------------------------------------
# Prepare Matplotlib Animation
# ------------------------------------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")

# Set consistent axis limits for visibility
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(0, 2)
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.title(f"Skeleton Animation: {PERSON} - {ACTIVITY}")

lines = []

# Initialize skeleton lines
for edge in edges:
    line, = ax.plot([], [], [], linewidth=3)
    lines.append(line)

# ------------------------------------
# Animation Update Function
# ------------------------------------
def update(frame):
    coords = get_frame_xyz(frame)
    for i, (j1, j2) in enumerate(edges):
        p1 = coords[j1]
        p2 = coords[j2]
        lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
        lines[i].set_3d_properties([p1[2], p2[2]])
    return lines

# ------------------------------------
# Run Animation
# ------------------------------------
ani = FuncAnimation(fig, update, frames=len(df), interval=40, blit=False)
ani.save("skeletal_animation/running_skeleton_animation.gif", writer=PillowWriter(fps=25))

plt.show()
