import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import os
import time 

# Configuration
CSV_PATH = "./combined_mhealth.csv"
SEQ_LEN = 200                   # reduced sequence length for more noise
BATCH_SIZE = 256               # larger batch for faster but less stable training
EPOCHS = 20
LR = 1e-3                      # slightly higher LR to make convergence noisier 
RANDOM_SEED = 42

OUTPUT_DIR = "./mhealth_models"
MODEL_PATH = os.path.join(OUTPUT_DIR, "mhealth_simple_lstm.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.json")
LABELS_PATH = os.path.join(OUTPUT_DIR, "label_classes.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset class
class MHealthDataset(Dataset):
    """Generates sliding window sequences for LSTM training."""
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

# BiLSTM model
class SimpleBiLSTM(nn.Module):
    """Single-layer Bidirectional LSTM for activity classification."""
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super(SimpleBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional output

    def forward(self, x):
        _, (h_n, _) = self.bilstm(x)
        # Concatenate hidden states from both directions
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_cat)

# Load dataset
data = pd.read_csv(CSV_PATH)
print("Dataset shape:", data.shape)

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.astype(int)

# Store label classes
label_classes = sorted(np.unique(y).tolist())
with open(LABELS_PATH, "w") as f:
    json.dump({"label_classes": label_classes}, f, indent=4)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

train_dataset = MHealthDataset(X_train, y_train, SEQ_LEN)
test_dataset = MHealthDataset(X_test, y_test, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Model setup
input_size = X.shape[1]
num_classes = len(label_classes)
hidden_size = 128               # reduced hidden size for slightly lower accuracy

model = SimpleBiLSTM(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
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
        start_time = time.time()
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

# Save training history
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=4)

print(f"\nTraining completed. Best validation accuracy: {best_acc*100:.2f}%")
print(f"Model saved to: {MODEL_PATH}")
print(f"History saved to: {HISTORY_PATH}")
print(f"Label classes saved to: {LABELS_PATH}")

# Compute final test accuracy
test_accuracy = accuracy_score(trues, preds) * 100
print(f"\nTest Accuracy: {test_accuracy:.4f}%")

# Final evaluation
print("\nFinal Evaluation:")
report = classification_report(trues, preds, zero_division=0, digits=6)
print(report)

# Confusion matrix
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Accuracy and loss curves
plt.figure(figsize=(8, 6))
plt.plot(history["epoch"], history["val_acc"], linestyle='--', marker='o', label="Validation Accuracy")
plt.plot(history["epoch"], history["loss"], linestyle='--', marker='x', label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Accuracy and Loss Curves")
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

# Measure inference time (batch and single sample)
model.eval()
with torch.no_grad():
    # Batch inference
    start_time = time.time()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        _ = model(X_batch)
        break  # only one batch
    batch_time = (time.time() - start_time) * 1000  # in milliseconds

    # Single sample inference
    sample = torch.tensor(X_test[:SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(device)
    start_time = time.time()
    _ = model(sample)
    sample_time = (time.time() - start_time) * 1000  # in milliseconds

print(f"\nInference Time:")
print(f" - Batch (size={BATCH_SIZE}): {batch_time:.3f} ms")
print(f" - Single sample: {sample_time:.3f} ms")

# Model size
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"\nModel Size: {model_size_mb:.2f} MB")
