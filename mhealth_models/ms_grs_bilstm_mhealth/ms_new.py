"""
train_multiscale_bilstm.py

Multi-Scale Gated Residual Skip BiLSTM for combined_mhealth.csv

Requirements:
    pip install torch torchvision numpy pandas scikit-learn tqdm

Usage:
    python train_multiscale_bilstm.py --csv_path ./combined_mhealth.csv
"""

import argparse
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Config / Defaults
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Utilities: windowing and dataset
# ---------------------------
def sliding_window_sequences(
    data_df: pd.DataFrame,
    window_size: int,
    stride: int,
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a row-wise CSV where each row is a timestamp sample with many sensor channels,
    into sequence windows with a single label per window (mode or last label).
    Returns X: (N_windows, window_size, n_channels), y: (N_windows,)
    """
    values = data_df.drop(columns=[label_col]).values  # shape (T, channels)
    labels = data_df[label_col].values  # shape (T,)

    T, C = values.shape
    windows = []
    window_labels = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        win = values[start:end]
        # choose label strategy: majority vote (mode); fallback to last value
        lab_segment = labels[start:end]
        # majority label:
        (vals, counts) = np.unique(lab_segment, return_counts=True)
        win_label = vals[np.argmax(counts)]
        windows.append(win)
        window_labels.append(win_label)

    X = np.stack(windows)  # (N, window_size, C)
    y = np.array(window_labels)
    return X, y


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, L, C)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# Model: MultiScale Gated Residual Skip BiLSTM
# ---------------------------
class MultiScaleGatedResidualSkipBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.25,
        num_classes: int = 2,
    ):
        """
        Multi-scale architecture:
            - branch_full: BiLSTM on original sequence
            - branch_ds2: BiLSTM on downsampled by 2 sequence
            - branch_ds4: BiLSTM on downsampled by 4 sequence
        Gating: compute scalar gates for each branch based on final hidden states (learned)
        Residual skip: project raw input through a linear layer and add to fused representation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM branches
        self.lstm_full = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                 bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.lstm_ds2 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.lstm_ds4 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # gating layers (produce a scalar gate per branch)
        self.gate_fc = nn.Sequential(
            nn.Linear(self.num_directions * hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # gates for 3 branches
        )

        # residual projection from temporally-pooled raw input -> fused dim
        self.input_proj = nn.Linear(input_dim, self.num_directions * hidden_dim)

        # final classifier
        fused_dim = self.num_directions * hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, x):
        """
        x: (B, L, C)
        returns logits: (B, num_classes)
        """
        B, L, C = x.shape

        # full-rate branch
        out_full, (h_full, c_full) = self.lstm_full(x)  # out_full: (B, L, D)
        # get last-layer hidden state (concatenate directions)
        # h_full shape: (num_layers * num_directions, B, hidden_dim)
        last_h_full = h_full.view(h_full.size(0) // self.num_directions, self.num_directions, B, self.hidden_dim)[-1]
        # last_h_full shape: (num_directions, B, hidden_dim)
        last_h_full = last_h_full.permute(1, 0, 2).contiguous().view(B, -1)  # (B, num_directions*hidden_dim)

        # downsample by 2
        x_ds2 = x[:, ::2, :]  # (B, L//2, C)
        out_ds2, (h_ds2, c_ds2) = self.lstm_ds2(x_ds2)
        last_h_ds2 = h_ds2.view(h_ds2.size(0) // self.num_directions, self.num_directions, B, self.hidden_dim)[-1]
        last_h_ds2 = last_h_ds2.permute(1, 0, 2).contiguous().view(B, -1)

        # downsample by 4
        x_ds4 = x[:, ::4, :]  # (B, L//4, C)
        out_ds4, (h_ds4, c_ds4) = self.lstm_ds4(x_ds4)
        last_h_ds4 = h_ds4.view(h_ds4.size(0) // self.num_directions, self.num_directions, B, self.hidden_dim)[-1]
        last_h_ds4 = last_h_ds4.permute(1, 0, 2).contiguous().view(B, -1)

        # gating: concatenation of branch summaries -> gates
        cat = torch.cat([last_h_full, last_h_ds2, last_h_ds4], dim=1)  # (B, 3 * num_directions*hidden_dim)
        gates = self.gate_fc(cat)  # (B, 3)
        gates = torch.sigmoid(gates)  # [0,1] per branch
        g1 = gates[:, 0].unsqueeze(1)  # (B,1)
        g2 = gates[:, 1].unsqueeze(1)
        g3 = gates[:, 2].unsqueeze(1)

        # fused representation (weighted sum)
        fused = g1 * last_h_full + g2 * last_h_ds2 + g3 * last_h_ds4  # (B, fused_dim)

        # residual skip: pool input temporally (mean) and project
        input_pool = torch.mean(x, dim=1)  # (B, C)
        skip = self.input_proj(input_pool)  # (B, fused_dim)

        fused = fused + skip  # residual add

        logits = self.classifier(fused)  # (B, num_classes)
        return logits


# ---------------------------
# Training / Evaluation functions
# ---------------------------
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    losses = []
    all_preds = []
    all_targets = []
    for Xb, yb in dataloader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        preds = torch.argmax(logits.detach().cpu(), axis=1)
        all_preds.append(preds.numpy())
        all_targets.append(yb.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    return np.mean(losses), acc


def eval_model(model, dataloader, criterion):
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            preds = torch.argmax(logits.cpu(), axis=1)
            all_preds.append(preds.numpy())
            all_targets.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    return np.mean(losses), acc


# ---------------------------
# Main training pipeline
# ---------------------------
def main(args):
    # 1) Load CSV
    df = pd.read_csv(args.csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must include 'label' column")

    # 2) Sliding windows
    window_size = args.window_size
    stride = args.stride
    print(f"Building windows (size={window_size}, stride={stride}) ...")
    X, y = sliding_window_sequences(df, window_size=window_size, stride=stride, label_col="label")
    print("Total windows:", X.shape[0], "Window shape:", X.shape[1:])

    # 3) Train/test split (stratify by label)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=SEED, stratify=y
    )
    relative = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative, random_state=SEED, stratify=y_temp
    )

    # 4) Per-channel normalization: fit on train windows flattened across time
    B_train, L, C = X_train.shape
    scalers = []
    X_train_reshaped = X_train.reshape(-1, C)  # (B_train * L, C)
    scaler = StandardScaler().fit(X_train_reshaped)
    def apply_scaler(arr):
        B, L, C = arr.shape
        out = scaler.transform(arr.reshape(-1, C)).reshape(B, L, C)
        return out

    X_train = apply_scaler(X_train)
    X_val = apply_scaler(X_val)
    X_test = apply_scaler(X_test)

    # 5) Datasets & loaders
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = int(np.max(y_train)) + 1
    print("Detected num_classes:", num_classes)

    # 6) Model
    model = MultiScaleGatedResidualSkipBiLSTM(
        input_dim=C,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=True,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    clip_value = 5.0  # for gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, steps_per_epoch=len(train_loader), epochs=args.epochs)
    # 7) Training loop with best-val checkpointing
    best_val_acc = -1.0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_model(model, val_loader, criterion)
        scheduler.step()
        history.append((epoch, train_loss, train_acc, val_loss, val_acc))


        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "args": vars(args)
            }, best_model_path)
            print(f"  -> new best val acc {best_val_acc:.4f}, saved to {best_model_path}")

    # 8) Load best and test
    ck = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    test_loss, test_acc = eval_model(model, test_loader, criterion)
    print(f"\nTEST RESULT: loss={test_loss:.4f} acc={test_acc:.4f}")

    # 9) Save final metrics and return
    results = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "best_model_path": best_model_path,
    }
    print("Done.", results)

        # ---------------------------
    # Add after: print("Done.", results)
    # ---------------------------

    from sklearn.metrics import classification_report, confusion_matrix
    import time

    # ---------- Classification Report and Confusion Matrix ----------
    print("\nGenerating classification report and confusion matrix...")
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Get unique label classes
    label_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
    class_names = [f"class_{int(c)}" for c in label_classes]

    # Generate metrics
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:\n", cm)

    # Save report and matrix
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report -> {report_path}")
    print(f"Saved confusion matrix -> {cm_path}")

    # ---------- Save label classes ----------
    label_classes_path = os.path.join(args.output_dir, "label_classes.npy")
    np.save(label_classes_path, label_classes)
    print(f"Saved label classes -> {label_classes_path}")

    # ---------- Save training history ----------
    # To capture history, add this during training loop:
    # history.append((epoch, train_loss, train_acc, val_loss, val_acc))
    # Make sure to define: history = [] before the loop
    try:
        history_path = os.path.join(args.output_dir, "training_history.csv")
        import csv
        with open(history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            writer.writerows(history)
        print(f"Saved training history -> {history_path}")
    except NameError:
        print("Note: to save history, define 'history=[]' before the training loop and append inside it.")

    # ---------- Inference time measurement ----------
    print("\nMeasuring inference time...")
    model.eval()
    dummy_batch = next(iter(test_loader))
    Xb, yb = dummy_batch
    Xb = Xb.to(DEVICE)

    # Batch inference time
    start = time.time()
    with torch.no_grad():
        _ = model(Xb)
    end = time.time()
    batch_time = end - start
    sample_time = batch_time / Xb.size(0)

    print(f"Inference time: {batch_time:.6f} s for batch of {Xb.size(0)}")
    print(f"Inference time per sample: {sample_time:.6f} s")

    # Save inference timing
    with open(os.path.join(args.output_dir, "inference_time.txt"), "w") as f:
        f.write(f"Batch size: {Xb.size(0)}\n")
        f.write(f"Batch time: {batch_time:.6f} s\n")
        f.write(f"Per-sample time: {sample_time:.6f} s\n")
    print(f"Saved inference timing -> {os.path.join(args.output_dir, 'inference_time.txt')}")



# ---------------------------
# CLI args
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="combined_mhealth.csv")
    parser.add_argument("--window_size", type=int, default=128, help="sequence length (samples)")
    parser.add_argument("--stride", type=int, default=64, help="sliding window stride")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="./runs_multiscale")
    args = parser.parse_args()
    main(args)
