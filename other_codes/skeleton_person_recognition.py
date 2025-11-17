import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

# ---------------------------
# Load your model definition
# ---------------------------
class MultiScaleGRSBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_scales, n_blocks_per_scale, num_classes, dropout=0.3):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.LSTM(input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=n_blocks_per_scale,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=True)
            for _ in range(n_scales)
        ])
        self.fc = nn.Linear(n_scales * hidden_dim * 2, num_classes)

    def forward(self, x):
        outputs = []
        for lstm in self.scales:
            out, _ = lstm(x)
            outputs.append(out[:, -1])  # last timestep
        combined = torch.cat(outputs, dim=1)
        return self.fc(combined)


# ---------------------------
# Helper: extract joints XYZ
# ---------------------------
def get_joint_groups(feature_cols):
    joints = sorted(list(set([c.rsplit("_", 1)[0] for c in feature_cols])))
    return joints

# ---------------------------
# Skeleton connections (example)
# ---------------------------
SKELETON_EDGES = [
    ("head", "neck"),
    ("neck", "spine_shoulder"),
    ("spine_shoulder", "spine_mid"),
    ("spine_mid", "spine_base"),
    ("spine_shoulder", "shoulder_left"),
    ("spine_shoulder", "shoulder_right"),
    ("shoulder_left", "elbow_left"),
    ("elbow_left", "wrist_left"),
    ("shoulder_right", "elbow_right"),
    ("elbow_right", "wrist_right"),
    ("spine_base", "hip_left"),
    ("hip_left", "knee_left"),
    ("knee_left", "ankle_left"),
    ("spine_base", "hip_right"),
    ("hip_right", "knee_right"),
    ("knee_right", "ankle_right"),
]

# ---------------------------
# Animation Script
# ---------------------------
def animate_skeleton(csv_path, model_path, activity_name, seq_len=64, stride=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    le = ckpt["label_encoder"]
    feature_cols = ckpt["feature_cols"]
    scaler = ckpt["scaler"]
    model_state = ckpt["model_state"]

    input_dim = len(feature_cols)

    # Build model
    model = MultiScaleGRSBiLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        n_scales=3,
        n_blocks_per_scale=2,
        num_classes=len(le.classes_),
        dropout=0.3
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # Load dataset
    df = pd.read_csv(csv_path)

    # Filter activity
    df = df[df["Activity"] == activity_name].reset_index(drop=True)
    if len(df) < seq_len:
        print("Not enough frames for the selected activity.")
        return

    # Extract features
    X = df[feature_cols].values.astype(np.float32)
    X = scaler.transform(X)

    # For skeleton joints
    joints = get_joint_groups(feature_cols)

    # Build animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"Activity: {activity_name}")

    # Prepare initial empty plot
    lines = []
    for _ in SKELETON_EDGES:
        line, = ax.plot([], [], [], linewidth=3)
        lines.append(line)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    text_pred = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # Animation update
    def update(frame_idx):
        start = max(0, frame_idx - seq_len)
        window = X[start:frame_idx]

        # Pad if needed
        if len(window) < seq_len:
            pad = np.zeros((seq_len - len(window), X.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])

        inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(inp)
            pred_label = le.inverse_transform([pred.argmax().item()])[0]

        text_pred.set_text(f"Predicted Person: {pred_label}")

        frame = df.iloc[frame_idx]

        # Extract joints for plotting
        xyz = {}
        for joint in joints:
            x = frame.get(f"{joint}_x", np.nan)
            y = frame.get(f"{joint}_y", np.nan)
            z = frame.get(f"{joint}_z", np.nan)
            xyz[joint] = (x, y, z)

        # Update skeleton lines
        for i, (j1, j2) in enumerate(SKELETON_EDGES):
            if j1 in xyz and j2 in xyz:
                x1, y1, z1 = xyz[j1]
                x2, y2, z2 = xyz[j2]
                lines[i].set_data([x1, x2], [y1, y2])
                lines[i].set_3d_properties([z1, z2])

        return lines + [text_pred]

    anim = FuncAnimation(fig, update, frames=len(df), interval=50, blit=False)
    plt.show()


# -----------------------------
# Argument Parser
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="combined_kinect_dataset.csv")
    parser.add_argument("--model", type=str, default="best_model.pt")
    parser.add_argument("--activity", type=str, required=True, help="Which activity to animate")
    parser.add_argument("--seq_len", type=int, default=64)
    args = parser.parse_args()

    animate_skeleton(args.csv, args.model, args.activity, seq_len=args.seq_len)
