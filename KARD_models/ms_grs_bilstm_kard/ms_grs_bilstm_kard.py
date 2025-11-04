# --- train_kard_ms_bilstm.py ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json, pickle

# =====================================================
#  Load and preprocess dataset
# =====================================================
csv_path = "KARD_all_realworld.csv"
print(f"Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# Sort for consistency
df = df.sort_values(["action_id", "subject_id", "repetition", "frame", "joint_name"])

# Pivot each frame into one row of 45 (15 joints × 3 coords)
pivoted = (
    df.pivot_table(
        index=["action_id", "subject_id", "repetition", "frame"],
        columns="joint_name",
        values=["x", "y", "z"]
    )
    .reset_index()
)
pivoted.columns = ["_".join(col).strip("_") for col in pivoted.columns.values]

# Group by each sequence
seqs, labels = [], []
for (a, s, r), group in pivoted.groupby(["action_id", "subject_id", "repetition"]):
    feat_cols = [c for c in group.columns if c not in ["action_id", "subject_id", "repetition", "frame"]]
    arr = group[feat_cols].values
    seqs.append(arr)
    labels.append(a - 1)  # 0-based labels

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
X_reshaped = X.reshape(-1, feature_dim)
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# Split
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
#  Model Definition
# =====================================================
class MultiScaleResBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=18, dropout=0.3):
        super().__init__()
        self.scale1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.scale2 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.scale3 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.reduce = nn.Linear(hidden_size * 6, hidden_size * 2)   #  projection for skip add
        self.res_fc = nn.Linear(hidden_size * 6, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x1, _ = self.scale1(x)
        x2, _ = self.scale2(x[:, ::2, :])
        x3, _ = self.scale3(x[:, ::4, :])
        x1, x2, x3 = x1.mean(dim=1), x2.mean(dim=1), x3.mean(dim=1)
        cat = torch.cat([x1, x2, x3], dim=-1)
        cat_proj = self.reduce(cat)
        res = torch.relu(self.res_fc(cat))
        out = self.norm(cat_proj + res)
        out = self.dropout(out)
        return self.classifier(out)


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
#  Train the Model
# =====================================================
model = MultiScaleResBiLSTM(input_size=input_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

best_acc, patience, counter = 0, 10, 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(21):
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
        torch.save(model.state_dict(), "kard_ms_bilstm_best.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# =====================================================
#  Final Test Evaluation
# =====================================================
model.load_state_dict(torch.load("kard_ms_bilstm_best.pt"))
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
torch.save(model.state_dict(), "kard_ms_bilstm_final.pt")
with open("kard_training_history.json", "w") as f:
    json.dump(history, f)
pickle.dump({"labels": np.unique(y_train)}, open("kard_labels.pkl", "wb"))

print("\n Training completed successfully.")
print("Saved:")
print("  • kard_ms_bilstm_best.pt")
print("  • kard_ms_bilstm_final.pt")
print("  • kard_training_history.json")
print("  • kard_labels.pkl")

# =====================================================
#  Evaluate Model: Print Hyperparameters, Report & Confusion Matrix
# =====================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ------------------ Hyperparameters ------------------
hyperparams = {
    "input_size": input_size,
    "hidden_size": 128,
    "num_classes": 18,
    "dropout": 0.3,
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

# ------------------ Evaluation ------------------
model.load_state_dict(torch.load("kard_ms_bilstm_best.pt"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        all_preds.extend(preds.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# ------------------ Classification Report ------------------
print("===== Classification Report =====")
print(classification_report(all_labels, all_preds, digits=4))

# ------------------ Confusion Matrix ------------------
cm = confusion_matrix(all_labels, all_preds)
print("\n===== Confusion Matrix =====")
print(cm)

# Optional: visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - KARD Multi-Scale Residual BiLSTM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

