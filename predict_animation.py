import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

# ---------------------------
# Load dataset for Person_1
# ---------------------------
df = pd.read_csv("combined_kinect_dataset.csv")
df = df[df['Person'] == 'Person_1']

# Joints used during training
joints = [
    'SpineBase','SpineMid','Neck','Head',
    'ShoulderLeft','ElbowLeft','WristLeft','HandLeft',
    'ShoulderRight','ElbowRight','WristRight','HandRight',
    'HipLeft','KneeLeft','AnkleLeft','FootLeft',
    'HipRight','KneeRight','AnkleRight','FootRight',
    'SpineShoulder','HandTipLeft','ThumbLeft','HandTipRight','ThumbRight'
]

# Feature columns
feature_cols = []
for joint in joints:
    feature_cols += [f"{joint}_x", f"{joint}_z", f"{joint}_y"]

X_raw = df[feature_cols].values
timestamps = df['timestamp'].values

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = r"Dataset_models_kinect\ms_grs_bilstm\ms_grs_lstm_model.h5"
model = load_model(MODEL_PATH)

# ---------------------------
# Recreate Label Encoder (same as training)
# ---------------------------
activity_classes = ['sitting','bending','walking','jumping','squats','running','standing']
le = LabelEncoder()
le.fit(activity_classes)

# ---------------------------
# Normalize features using StandardScaler
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)  # use your own scaler from training if available
X = X_scaled.astype(np.float32)

# ---------------------------
# Skeleton connections
# ---------------------------
connections = [
    ('SpineBase','SpineMid'), ('SpineMid','SpineShoulder'), ('SpineShoulder','Neck'), ('Neck','Head'),
    ('SpineShoulder','ShoulderLeft'), ('ShoulderLeft','ElbowLeft'), ('ElbowLeft','WristLeft'), ('WristLeft','HandLeft'), ('HandLeft','HandTipLeft'), ('HandLeft','ThumbLeft'),
    ('SpineShoulder','ShoulderRight'), ('ShoulderRight','ElbowRight'), ('ElbowRight','WristRight'), ('WristRight','HandRight'), ('HandRight','HandTipRight'), ('HandRight','ThumbRight'),
    ('SpineBase','HipLeft'), ('HipLeft','KneeLeft'), ('KneeLeft','AnkleLeft'), ('AnkleLeft','FootLeft'),
    ('SpineBase','HipRight'), ('HipRight','KneeRight'), ('KneeRight','AnkleRight'), ('AnkleRight','FootRight')
]

# ---------------------------
# Sliding window parameters
# ---------------------------
SEQ_LEN = 64
num_frames = len(X)

# ---------------------------
# Animation setup
# ---------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.5)
ax.set_zlim(0, 1)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
title_text = ax.set_title("")

lines = [ax.plot([], [], [], 'o-', lw=2)[0] for _ in connections]

# Store recent predictions to smooth flickering
recent_preds = []

def update(frame):
    # Skip until we have full sequence
    if frame < SEQ_LEN-1:
        return lines + [title_text]

    # Get the last 64-frame window
    seq_input = X[frame-SEQ_LEN+1:frame+1]  # (64, 75)
    seq_input = np.expand_dims(seq_input, axis=0)  # (1, 64, 75)

    # Predict activity
    pred_probs = model.predict(seq_input, verbose=0)
    pred_idx = np.argmax(pred_probs, axis=1)[0]
    recent_preds.append(pred_idx)
    # Keep last 5 predictions for smoothing
    if len(recent_preds) > 5:
        recent_preds.pop(0)
    # Most frequent prediction in recent window
    smoothed_idx = max(set(recent_preds), key=recent_preds.count)
    activity_name = le.inverse_transform([smoothed_idx])[0]

    title_text.set_text(f"Predicted Activity: {activity_name}")

    # Draw current skeleton
    frame_data = X_raw[frame]  # use raw for visualization
    for line, (j1, j2) in zip(lines, connections):
        idx1 = joints.index(j1)
        idx2 = joints.index(j2)
        x = [frame_data[idx1*3], frame_data[idx2*3]]
        y = [frame_data[idx1*3+1], frame_data[idx2*3+1]]
        z = [frame_data[idx1*3+2], frame_data[idx2*3+2]]
        line.set_data(x, y)
        line.set_3d_properties(z)
    return lines + [title_text]

ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
# ani.save("skeletal_animation/prediction_skeleton_new.gif", writer=PillowWriter(fps=25))
plt.show()
