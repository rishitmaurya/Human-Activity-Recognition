# --- train_kard_bilstm.py ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json, pickle, os, time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
#  Load and preprocess dataset
# =====================================================
csv_path = "KARD_all_realworld.csv"
print(f"Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# Sort and pivot
df = df.sort_values(["action_id", "subject_id", "repetition", "frame", "joint_name"])
pivoted = (
    df.pivot_table(
        index=["action_id", "subject_id", "repetition", "frame"],
        columns="joint_name",
        values=["x", "y", "z"]
    )
    .reset_index()
)
pivoted.columns = ["_".join(col).strip("_") for col in pivoted.columns.values]

# Group sequences
seqs, labels = [], []
for (a, s, r), group in pivoted.groupby(["action_id", "subject_id", "repetition"]):
    feat_cols = [c for c in group.columns if c not in ["action_id", "subject_id", "repetition", "frame"]]
    arr = group[feat_cols].values
    seqs.append(arr)
    labels.append(a - 1)

# Pad/truncate sequences
max_len = max(len(x) for x in seqs)
feature_dim = seqs[0].shape[1]
X = np.zeros((len(seqs), max_len, feature_dim))
for i, seq in enumerate(seqs):
    L = len(seq)
    X[i, :L, :] = seq
y = np.array(labels)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, feature_dim)).reshape(X.shape)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(" Data shapes:",
      "\n  X_train:", X_train.shape,
      "\n  X_val:", X_val.shape,
      "\n  X_test:", X_test.shape)

# =====================================================
#  Gated Residual Skip BiLSTM Model Definition
# =====================================================
class GatedResidualSkipBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, num_classes=18, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Primary BiLSTM
        self.bilstm_main = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                   batch_first=True, bidirectional=True, dropout=dropout)

        # Skip connections (downsampled temporal resolutions)
        self.bilstm_skip1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                    batch_first=True, bidirectional=True, dropout=dropout)
        self.bilstm_skip2 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                    batch_first=True, bidirectional=True, dropout=dropout)

        # Gating mechanism
        self.gate_main = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.gate_skip1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.gate_skip2 = nn.Linear(hidden_size * 2, hidden_size * 2)

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out_main, _ = self.bilstm_main(x)

        # Downsampled skip connections (simulate multi-scale)
        x_skip1 = x[:, ::2, :]     # half temporal resolution
        x_skip2 = x[:, ::4, :]     # quarter temporal resolution

        out_skip1, _ = self.bilstm_skip1(x_skip1)
        out_skip2, _ = self.bilstm_skip2(x_skip2)

        # Mean pooling each stream
        out_main = out_main.mean(dim=1)
        out_skip1 = out_skip1.mean(dim=1)
        out_skip2 = out_skip2.mean(dim=1)

        # Gated fusion
        g_main = torch.sigmoid(self.gate_main(out_main))
        g_skip1 = torch.sigmoid(self.gate_skip1(out_skip1))
        g_skip2 = torch.sigmoid(self.gate_skip2(out_skip2))

        fused = g_main * out_main + g_skip1 * out_skip1 + g_skip2 * out_skip2

        # Residual connection + normalization
        out = self.norm(fused + out_main)
        out = self.dropout(out)
        return self.fc(out)


# =====================================================
#  Prepare DataLoaders
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_size = X_train.shape[2]
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long)),
                          batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.long)),
                        batch_size=32)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                       torch.tensor(y_test, dtype=torch.long)),
                         batch_size=32)

# =====================================================
#  Train the BiLSTM Model
# =====================================================
model = GatedResidualSkipBiLSTM(input_size=input_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

best_acc, patience, counter = 0, 10, 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(20):
    model.train()
    total, correct, train_loss = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss / len(train_loader))
    history["val_loss"].append(val_loss / len(val_loader))
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "kard_grs_bilstm_best.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# =====================================================
#  Final Test Evaluation
# =====================================================
model.load_state_dict(torch.load("kard_grs_bilstm_best.pt"))
model.eval()
correct, total, test_loss = 0, 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        test_loss += criterion(preds, yb).item()
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
test_acc = correct / total
print(f"\n Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {test_acc*100:.2f}%")

# =====================================================
#  Save Outputs
# =====================================================
torch.save(model.state_dict(), "kard_grs_bilstm_final.pt")
with open("kard_grs_bilstm_training_history.json", "w") as f:
    json.dump(history, f)
pickle.dump({"labels": np.unique(y_train)}, open("kard_grs_bilstm_labels.pkl", "wb"))

print("\n Training completed successfully.")
print("Saved:")
print("  • kard_grs_bilstm_best.pt")
print("  • kard_grs_bilstm_final.pt")
print("  • kard_grs_bilstm_training_history.json")
print("  • kard_grs_bilstm_labels.pkl")

# =====================================================
#  Evaluation, Curves, Inference Time, Model Size
# =====================================================
with open("kard_grs_bilstm_training_history.json", "r") as f:
    history = json.load(f)
with open("kard_grs_bilstm_labels.pkl", "rb") as f:
    label_data = pickle.load(f)
class_names = [f"Action_{int(l)}" for l in label_data["labels"]]

# Hyperparameters
hyperparams = {
    "model_type": "BiLSTM",
    "input_size": input_size,
    "hidden_size": 32,
    "num_layers": 1,
    "num_classes": 18,
    "dropout": 0.4,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau",
    "epochs_trained": len(history["train_acc"]),
    "patience": patience
}
print("\n===== Model Hyperparameters =====")
for k, v in hyperparams.items():
    print(f"{k}: {v}")
print("=================================\n")

# Model size
model_size = os.path.getsize("kard_grs_bilstm_best.pt") / (1024 ** 2)
print(f"Model size: {model_size:.2f} MB")

# Inference time
batch_times, sample_times = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        start = time.time()
        preds = model(xb)
        end = time.time()
        batch_times.append(end - start)
        sample_times.extend([(end - start) / xb.size(0)] * xb.size(0))
avg_batch_time = np.mean(batch_times)
avg_sample_time = np.mean(sample_times)
print(f"\nAverage inference time per batch: {avg_batch_time*1000:.6f} ms")
print(f"Average inference time per sample: {avg_sample_time*1000:.6f} ms")

# Predictions
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        all_preds.extend(preds.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# Classification report
print("\n===== Classification Report =====")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - KARD BiLSTM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Accuracy & Loss curves
epochs = range(1, len(history["train_acc"]) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=2)
plt.plot(epochs, history["val_acc"], label="Val Accuracy", linewidth=2)
plt.plot(epochs, history["train_loss"], '--', label="Train Loss", linewidth=2)
plt.plot(epochs, history["val_loss"], '--', label="Val Loss", linewidth=2)
plt.title("Accuracy & Loss Curves - BiLSTM")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
