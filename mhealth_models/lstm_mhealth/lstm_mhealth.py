# -------------------------------------------------------------
# Simple LSTM for mHealth Activity Recognition using combined_mhealth.csv
# Saves model, history (JSON), and label classes
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
import random
import os

# ---------------- Configuration ----------------
CSV_PATH = "./combined_mhealth.csv"
SEQ_LEN = 100
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
RANDOM_SEED = 42

OUTPUT_DIR = "./outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "mhealth_simple_lstm.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")
LABELS_PATH = os.path.join(OUTPUT_DIR, "label_classes.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Dataset ----------------
class MHealthDataset(Dataset):
    """Prepares sliding window sequences for LSTM from standardized data."""
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len - 1]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.long)

# ---------------- Model ----------------
class SimpleLSTM(nn.Module):
    """Single-layer LSTM for activity classification."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# ---------------- Load and Prepare Data ----------------
data = pd.read_csv(CSV_PATH)
print("Dataset shape:", data.shape)

# Last column = label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.astype(int)

# Store label classes
label_classes = sorted(np.unique(y).tolist())
with open(LABELS_PATH, "w") as f:
    json.dump({"label_classes": label_classes}, f, indent=4)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

train_dataset = MHealthDataset(X_train, y_train, SEQ_LEN)
test_dataset = MHealthDataset(X_test, y_test, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ---------------- Initialize Model ----------------
input_size = X.shape[1]
num_classes = len(label_classes)
hidden_size = 128

model = SimpleLSTM(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- Training ----------------
print("\nStarting training...\n")
best_acc = 0.0
history = {"epoch": [], "loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(y_batch.numpy())

    acc = accuracy_score(trues, preds)
    epoch_loss = total_loss / len(train_loader.dataset)
    history["epoch"].append(epoch)
    history["loss"].append(round(epoch_loss, 4))
    history["val_acc"].append(round(acc, 4))

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "seq_len": SEQ_LEN
        }, MODEL_PATH)

# ---------------- Save History ----------------
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=4)

print(f"\nTraining completed. Best validation accuracy: {best_acc*100:.2f}%")
print(f"Model saved to: {MODEL_PATH}")
print(f"History saved to: {HISTORY_PATH}")
print(f"Label classes saved to: {LABELS_PATH}")

# ---------------- Final Evaluation ----------------
print("\nFinal Evaluation:")
report = classification_report(trues, preds, zero_division=0)
print(report)
