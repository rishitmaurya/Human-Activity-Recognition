import pandas as pd
import numpy as np
import os
import time
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =========================================================
# GPU SETUP (THIS IS WHAT ACTUALLY ENABLES GPU)
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# 1. PATH SETUP
# =========================================================
dataset_path = r'wearable_sensors/dataset_wearable/combined_har_dataset.csv'
output_dir = r'wearable_sensors/ms_grs_bilstm_wear/testing'
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 2. DATA LOADING & FEATURE ENGINEERING
# =========================================================
print("Step 1: Engineering Features and Magnitudes...")
df = pd.read_csv(dataset_path)

df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

features = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','acc_mag','gyro_mag']
X_raw = df[features].values
y_raw = df['activity'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# =========================================================
# 3. DATA AUGMENTATION & WINDOWING
# =========================================================
def augment_data(data):
    noise = np.random.normal(0, 0.01, data.shape)
    return data + noise

def create_windows(X, y, window_size=128, step_size=32):
    X_win, y_win = [], []
    for i in range(0, len(X) - window_size, step_size):
        win = X[i:i + window_size]
        X_win.append(win)
        y_win.append(np.argmax(np.bincount(y[i:i + window_size])))
        X_win.append(augment_data(win))
        y_win.append(y_win[-1])
    return np.array(X_win), np.array(y_win)

X_seq, y_seq = create_windows(X_scaled, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42
)

hyperparams = {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "window_size": 128,
    "step_size": 32,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss",
    "dropout": 0.4,
}

with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)


# =========================================================
# 4. CLASS WEIGHTS
# =========================================================
weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.att(x), dim=1)
        return (x * weights).sum(dim=1)


# =========================================================
# 5. MODEL ARCHITECTURE (MS-GRS-BiLSTM)
# =========================================================
class MSGRSBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv3 = nn.Conv1d(8, 64, 3, padding=1)
        self.conv5 = nn.Conv1d(8, 64, 5, padding=2)
        self.conv7 = nn.Conv1d(8, 64, 7, padding=3)

        self.bn = nn.BatchNorm1d(192)
        self.gate = nn.Conv1d(192, 192, 1)
        self.shortcut = nn.Conv1d(8, 192, 1)

        self.lstm1 = nn.LSTM(192, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        self.attention = Attention(128)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        ms = torch.cat([
            torch.relu(self.conv3(x)),
            torch.relu(self.conv5(x)),
            torch.relu(self.conv7(x))
        ], dim=1)

        ms = self.bn(ms)
        gate = torch.sigmoid(self.gate(ms))
        x = torch.relu(ms * gate + self.shortcut(x))

        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.attention(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MSGRSBiLSTM(num_classes).to(device)

# =========================================================
# 6. TRAINING SETUP
# =========================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ),
    batch_size=64, shuffle=True, pin_memory=True
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    ),
    batch_size=64, pin_memory=True
)

# =========================================================
# 7. TRAINING LOOP
# =========================================================
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

for epoch in range(50):
    model.train()
    correct = total = loss_sum = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct += (outputs.argmax(1) == yb).sum().item()
        total += yb.size(0)

    history['accuracy'].append(correct / total)
    history['loss'].append(loss_sum / len(train_loader))

    model.eval()
    correct = total = loss_sum = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            loss_sum += loss.item()
            correct += (outputs.argmax(1) == yb).sum().item()
            total += yb.size(0)

    history['val_accuracy'].append(correct / total)
    history['val_loss'].append(loss_sum / len(test_loader))

    print(f"Epoch {epoch+1}/50 | "
          f"Acc: {history['accuracy'][-1]:.4f} | "
          f"Val Acc: {history['val_accuracy'][-1]:.4f}")

# =========================================================
# 8. EVALUATION
# =========================================================
model.eval()
y_pred = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        y_pred.extend(model(xb).argmax(1).cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_), digits=6)

# =========================================================
# 9. CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(11, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Confusion Matrix (MS-GRS-BiLSTM)", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.show()


num_params = sum(p.numel() for p in model.parameters())
model_size_mb = num_params * 4 / (1024 ** 2)

print(f"Model Parameters: {num_params:,}")
print(f"Model Size: {model_size_mb:.4f} MB")

# ---- Batch inference time ----
start = time.time()
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        _ = model(xb)
batch_time = (time.time() - start) / len(test_loader)

# ---- Single sample inference time ----
sample = torch.tensor(X_test[0:1], dtype=torch.float32).to(device)
start = time.time()
with torch.no_grad():
    _ = model(sample)
single_time = time.time() - start

print(f"Avg Batch Inference Time: {batch_time:.6f} sec")
print(f"Single Sample Inference Time: {single_time:.6f} sec")


plt.figure(figsize=(10, 6))

plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)

plt.plot(history['loss'], '--', label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], '--', label='Val Loss', linewidth=2)

plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training & Validation Accuracy + Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(os.path.join(output_dir, "loss_accuracy_curve.png"))
plt.show()


# =========================================================
# 10. SAVE EVERYTHING
# =========================================================
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model_98.pt'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))

with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
    json.dump(history, f)

print("\nAll models, scalers, and labels saved.")
