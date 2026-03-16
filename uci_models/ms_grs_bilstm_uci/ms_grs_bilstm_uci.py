# --- train_uci_ms_grs_bilstm_v3.py ---
# Changes to achieve ~98% test accuracy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import json, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score,
                             precision_recall_curve, roc_curve, auc,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =====================================================
#  Create output directory
# =====================================================
OUTPUT_DIR = "uci_models/ms_grs_bilstm_uci/test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# =====================================================
#  Load UCI HAR dataset
# =====================================================
csv_path = "UCI_HAR_Dataset.csv"
print(f"Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns (first 10): {df.columns.tolist()[:10]}")
print(f"Columns (last 10): {df.columns.tolist()[-10:]}")
print(f"\nActivity distribution:")
print(df['activity_name'].value_counts())
print(f"\nSet type distribution:")
print(df['set_type'].value_counts())
print(f"\nSubject distribution:")
print(f"Total subjects: {df['subject_id'].nunique()}")

# =====================================================
#  Use ALL 561 features
# =====================================================
meta_cols = ['subject_id', 'activity_id', 'activity_name', 'set_type']
all_feature_cols = [col for col in df.columns if col not in meta_cols]
print(f"\nTotal feature columns: {len(all_feature_cols)}")
print(f"Using ALL {len(all_feature_cols)} features")

# =====================================================
#  Prepare Data
# =====================================================
X = df[all_feature_cols].values
y_raw = df['activity_id'].values
set_types = df['set_type'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
print(f"\nNumber of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

activity_names = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}
class_names = [activity_names[c] for c in label_encoder.classes_]
print(f"Class names: {class_names}")

# =====================================================
#  KEY CHANGE #1: Keep original train/test split intact
#  Use very small validation set to maximize training data
# =====================================================
train_mask = set_types == 'train'
test_mask = set_types == 'test'

X_train_full = X[train_mask]
y_train_full = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"\nOriginal split:")
print(f"  Train: {X_train_full.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Very small validation (5%) to keep more training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.05,  # Only 5% for validation
    stratify=y_train_full, 
    random_state=42
)

print(f"\nFinal split:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val: {X_val.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# =====================================================
#  Normalize Features
# =====================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

noise_std = 0.015
X_train = X_train + np.random.normal(0, noise_std, X_train.shape)

# =====================================================
#  FEATURE-VIEW SEQUENCE CREATION
#  561 features are precomputed statistics, not raw time steps
# =====================================================
n_features = X_train.shape[1]  # 561
seq_len = 3  # [original, permuted-group, averaged-view]

def create_augmented_sequence(X):
    X_orig = X.copy()
    
    # grouped permutation instead of simple roll
    idx = np.arange(X.shape[1])
    np.random.seed(42)
    perm = np.random.permutation(idx)
    X_perm = X[:, perm]
    
    X_avg = 0.5 * (X_orig + X_perm)
    return np.stack([X_orig, X_perm, X_avg], axis=1)

X_train = create_augmented_sequence(X_train)
X_val = create_augmented_sequence(X_val)
X_test = create_augmented_sequence(X_test)

features_per_step = X_train.shape[2]

print(f"\nReshaped data:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  Sequence length: {seq_len}")
print(f"  Features per step: {features_per_step}")

# =====================================================
#  MODEL: Better suited for precomputed features
# =====================================================
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        return self.norm(h2 + self.skip(x))


class GatedResidualUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h_out = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        out = g * h_out + (1 - g) * residual
        return self.norm(out)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        weighted = torch.sum(x * weights, dim=1)
        return weighted, weights


class MSGRSBiLSTM_V4(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=6, dropout=0.15):
        super().__init__()
        self.hidden_size = hidden_size

        self.feature_extractor = nn.Sequential(
            FeatureExtractor(input_size, hidden_size, dropout),
            FeatureExtractor(hidden_size, hidden_size, dropout)
        )

        self.bilstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.gru = GatedResidualUnit(hidden_size, hidden_size // 2, dropout)
        self.temporal_attention = TemporalAttention(hidden_size)
        self.skip_proj = nn.Linear(hidden_size, hidden_size)
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, return_features=False):
        batch_size, seq_len, feat_dim = x.shape

        x_flat = x.reshape(batch_size * seq_len, feat_dim)
        feat = self.feature_extractor(x_flat)
        feat = feat.reshape(batch_size, seq_len, -1)

        skip = self.skip_proj(feat.mean(dim=1))

        lstm_out, _ = self.bilstm(feat)
        gru_out = self.gru(lstm_out)

        attended, _ = self.temporal_attention(gru_out)
        features = torch.cat([attended, skip], dim=-1)

        logits = self.classifier(features)
        aux_logits = self.aux_classifier(skip)

        if return_features:
            return logits, features
        return logits, aux_logits


# =====================================================
#  KEY CHANGE #4: Mixup Data Augmentation
# =====================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =====================================================
#  Prepare DataLoaders
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

# =====================================================
#  KEY CHANGE #5: Optimized Hyperparameters
# =====================================================
BATCH_SIZE = 64
HIDDEN_SIZE = 224
NUM_LAYERS = 2
DROPOUT = 0.20
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 8e-4
NUM_EPOCHS = 50
MIXUP_ALPHA = 0.10

print(f"\nHyperparameters:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Num Layers: {NUM_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Mixup Alpha: {MIXUP_ALPHA}")
print(f"  Epochs: {NUM_EPOCHS}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

# =====================================================
#  Initialize Model
# =====================================================
input_size = X_train.shape[2]

model = MSGRSBiLSTM_V4(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

print("\nModel Architecture:")
print(model)

# =====================================================
#  KEY CHANGE #6: Focal Loss for hard examples
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, 
                                               weight=self.alpha, 
                                               reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        loss = (-true_dist * log_preds).sum(dim=1)

        if self.weight is not None:
            sample_weights = self.weight[target]
            loss = loss * sample_weights

        return loss.mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.05, weight=class_weights)

# =====================================================
#  Optimizer with weight decay
# =====================================================
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos'
)

# =====================================================
#  Training Loop with Mixup
# =====================================================
best_val_acc = 0
best_val_loss = float('inf')
best_test_acc = 0
best_val_f1 = 0
patience = 30
patience_counter = 0
history = {
    "train_loss": [], "val_loss": [], 
    "train_acc": [], "val_acc": [],
    "train_f1": [], "val_f1": [],
    "test_acc": [],
    "lr": []
}

print("\n" + "="*70)
print("Starting Training - Robust MS-GRS-BiLSTM with Mixup")
print("="*70)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Training with Mixup
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_preds = []
    train_labels = []
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Apply mixup
        if MIXUP_ALPHA > 0 and epoch < 40 and np.random.random() > 0.5:
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs, aux_outputs = model(mixed_x)
            loss_main = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss_aux = mixup_criterion(criterion, aux_outputs, y_a, y_b, lam)
            loss = loss_main + 0.5 * loss_aux
            loss.backward()

            # For accuracy calculation, use original predictions
            with torch.no_grad():
                orig_outputs, _ = model(batch_x)
                _, predicted = orig_outputs.max(1)
        else:
            optimizer.zero_grad()
            outputs, aux_outputs = model(batch_x)
            loss_main = criterion(outputs, batch_y)
            loss_aux = criterion(aux_outputs, batch_y)
            loss = loss_main + 0.5 * loss_aux
            loss.backward()
            _, predicted = outputs.max(1)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        train_total += batch_y.size(0)
        train_correct += predicted.eq(batch_y).sum().item()
        
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(batch_y.cpu().numpy())
    
    
    
    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    train_f1 = f1_score(train_labels, train_preds, average='macro')
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, aux_outputs = model(batch_x)
            loss_main = criterion(outputs, batch_y)
            loss_aux = criterion(aux_outputs, batch_y)
            loss = loss_main + 0.5 * loss_aux
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += predicted.eq(batch_y).sum().item()
            
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())
    
    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    # Track test accuracy only for logging every 5 epochs
    if (epoch + 1) % 5 == 0:
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()
        current_test_acc = 100. * test_correct / test_total
    else:
        current_test_acc = history["test_acc"][-1] if len(history["test_acc"]) > 0 else 0.0
    
    current_lr = optimizer.param_groups[0]['lr']
    
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)
    history["test_acc"].append(current_test_acc)
    history["lr"].append(current_lr)
    
    print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | "
          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Test: {current_test_acc:.2f}% | "
          f"LR: {current_lr:.2e}")
    
    # Save based on validation accuracy
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        best_test_acc = current_test_acc
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'test_acc': current_test_acc,
        }, os.path.join(OUTPUT_DIR, "uci_ms_grs_bilstm_best.pt"))
        print(f"  → New best! Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {current_test_acc:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

training_time = time.time() - start_time
print("\n" + "="*70)
print(f"Training completed in {training_time:.2f} seconds!")
print(f"Best Test Acc: {best_test_acc:.2f}%")
print("="*70)

# =====================================================
#  Final Test Evaluation
# =====================================================
print("\nLoading best model for final evaluation...")
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "uci_ms_grs_bilstm_best.pt"), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_correct = 0
test_total = 0
test_loss = 0
all_preds = []
all_labels = []
all_probs = []
all_features = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs, features = model(batch_x, return_features=True)
        loss = criterion(outputs, batch_y)
        
        probs = torch.softmax(outputs, dim=1)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += batch_y.size(0)
        test_correct += predicted.eq(batch_y).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_features.extend(features.cpu().numpy())

test_acc = 100. * test_correct / test_total
avg_test_loss = test_loss / len(test_loader)

all_probs = np.array(all_probs)
all_features = np.array(all_features)

print(f"\n{'='*70}")
print(f"FINAL TEST RESULTS")
print(f"{'='*70}")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"{'='*70}")

# =====================================================
#  Save all artifacts
# =====================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "uci_ms_grs_bilstm_final.pt"))

with open(os.path.join(OUTPUT_DIR, "uci_training_history.json"), "w") as f:
    json.dump(history, f)

with open(os.path.join(OUTPUT_DIR, "uci_preprocessing.pkl"), "wb") as f:
    pickle.dump({
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_cols': all_feature_cols,
        'seq_len': seq_len,
        'features_per_step': features_per_step,
        'class_names': class_names
    }, f)

model_path = os.path.join(OUTPUT_DIR, "uci_ms_grs_bilstm_best.pt")
model_size_bytes = os.path.getsize(model_path)
model_size_kb = model_size_bytes / 1024
model_size_mb = model_size_kb / 1024

hyperparams = {
    "model_name": "MSGRSBiLSTM_V4",
    "dataset": "UCI HAR",
    "input_size": input_size,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "num_classes": num_classes,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "weight_decay": WEIGHT_DECAY,
    "scheduler": "OneCycleLR",
    "mixup_alpha": MIXUP_ALPHA,
    "loss": "LabelSmoothingCrossEntropy",
    "label_smoothing": 0.05,
    "seq_len": seq_len,
    "features_per_step": features_per_step,
    "num_features": len(all_feature_cols),
    "epochs_trained": len(history['train_loss']),
    "best_val_acc": best_val_acc,
    "best_val_loss": best_val_loss,
    "best_test_acc": best_test_acc,
    "test_acc": test_acc,
    "training_time_seconds": training_time,
    "total_params": total_params,
    "model_size_mb": model_size_mb
}

with open(os.path.join(OUTPUT_DIR, "uci_hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=2)

print(f"\nSaved files to {OUTPUT_DIR}")

# =====================================================
#  Detailed Evaluation
# =====================================================
print("\n" + "="*70)
print("DETAILED EVALUATION")
print("="*70)

print(f"\nModel size:")
print(f"  Bytes: {model_size_bytes:,}")
print(f"  KB: {model_size_kb:.2f}")
print(f"  MB: {model_size_mb:.4f}")

# Inference timing
batch_times = []
model.eval()
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        model(batch_x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        model(batch_x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        batch_times.append(end - start)

avg_batch_time = np.mean(batch_times) * 1000
avg_sample_time = avg_batch_time / BATCH_SIZE

print(f"\nInference timing:")
print(f"  Average batch time: {avg_batch_time:.2f} ms")
print(f"  Average sample time: {avg_sample_time:.4f} ms")
print(f"  Throughput: {1000/avg_sample_time:.0f} samples/sec")

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# Calculate metrics
macro_f1 = f1_score(all_labels, all_preds, average='macro')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
macro_precision = precision_score(all_labels, all_preds, average='macro')
macro_recall = recall_score(all_labels, all_preds, average='macro')
kappa = cohen_kappa_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

# Per-class metrics
print("\nPer-class Metrics:")
print("-" * 60)
per_class_acc = []
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

for i, class_name in enumerate(class_names):
    class_mask = np.array(all_labels) == i
    if class_mask.sum() > 0:
        class_preds = np.array(all_preds)[class_mask]
        class_acc = (class_preds == i).sum() / class_mask.sum() * 100
        per_class_acc.append(class_acc)
        print(f"  {class_name}: {class_acc:.2f}% ({class_mask.sum()} samples)")
    else:
        per_class_acc.append(0)

precisions = [report_dict[name]['precision'] * 100 for name in class_names]
recalls = [report_dict[name]['recall'] * 100 for name in class_names]
f1_scores_list = [report_dict[name]['f1-score'] * 100 for name in class_names]

# =====================================================
#  VISUALIZATIONS (ALL 21)
# =====================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

# 1. Combined Accuracy and Loss Plot
fig, ax = plt.subplots(figsize=(12, 8))
epochs_range = range(1, len(history['train_acc']) + 1)

ax.plot(epochs_range, history['train_acc'], 'b-', linewidth=2.5, label='Train Accuracy', marker='o', markersize=2)
ax.plot(epochs_range, history['val_acc'], 'r-', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=2)
ax.plot(epochs_range, history['test_acc'], 'g-', linewidth=2.5, label='Test Accuracy', marker='^', markersize=2)

ax2 = ax.twinx()
ax2.plot(epochs_range, history['train_loss'], 'b--', linewidth=2, label='Train Loss', alpha=0.7)
ax2.plot(epochs_range, history['val_loss'], 'r--', linewidth=2, label='Validation Loss', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax.set_title('Training Progress - Robust MS-GRS-BiLSTM with Mixup', fontsize=16)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_accuracy_loss_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_accuracy_loss_curves.png")

# 2. Normalized Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 12})
plt.title(f'Normalized Confusion Matrix - Test Accuracy: {test_acc:.2f}%', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_confusion_matrix.png")

# 3. Raw Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 12})
plt.title('Confusion Matrix (Raw Counts)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_confusion_matrix_raw.png")

# 4. Learning Rate Plot
plt.figure(figsize=(10, 6))
plt.plot(history['lr'], linewidth=2.5, color='green')
plt.fill_between(range(len(history['lr'])), history['lr'], alpha=0.3, color='green')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedule (CosineAnnealingWarmRestarts)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_learning_rate.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_learning_rate.png")

# 5. Per-class Accuracy Bar Plot
plt.figure(figsize=(12, 8))
bar_colors = ['green' if acc >= 95 else 'orange' if acc >= 90 else 'red' for acc in per_class_acc]
bars = plt.bar(range(len(class_names)), per_class_acc, color=bar_colors, edgecolor='black', linewidth=1.2)
plt.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
plt.axhline(y=np.mean(per_class_acc), color='blue', linestyle='-.', linewidth=2, 
            label=f'Mean ({np.mean(per_class_acc):.1f}%)')
plt.xlabel('Activity', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=14)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 105])

for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_per_class_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_per_class_accuracy.png")

# 6. Precision, Recall, F1-Score Comparison
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#3498db', edgecolor='black')
bars3 = ax.bar(x + width, f1_scores_list, width, label='F1-Score', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Activity', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Precision, Recall, and F1-Score by Class', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim([0, 110])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_precision_recall_f1.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_precision_recall_f1.png")

# 7. Training Subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(epochs_range, history['train_acc'], 'b-', linewidth=2, label='Train', marker='o', markersize=2)
axes[0, 0].plot(epochs_range, history['val_acc'], 'r-', linewidth=2, label='Validation', marker='s', markersize=2)
axes[0, 0].plot(epochs_range, history['test_acc'], 'g-', linewidth=2, label='Test', marker='^', markersize=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Accuracy Over Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 105])

axes[0, 1].plot(epochs_range, history['train_loss'], 'b-', linewidth=2, label='Train', marker='o', markersize=2)
axes[0, 1].plot(epochs_range, history['val_loss'], 'r-', linewidth=2, label='Validation', marker='s', markersize=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss Over Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs_range, history['train_f1'], 'b-', linewidth=2, label='Train', marker='o', markersize=2)
axes[1, 0].plot(epochs_range, history['val_f1'], 'r-', linewidth=2, label='Validation', marker='s', markersize=2)
axes[1, 0].axhline(y=macro_f1, color='g', linestyle='--', linewidth=2, label=f'Test ({macro_f1:.4f})')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Macro F1 Score')
axes[1, 0].set_title('F1 Score Over Epochs')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
axes[1, 1].plot(epochs_range, gap, 'purple', linewidth=2, marker='d', markersize=2)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 1].fill_between(epochs_range, gap, alpha=0.3, color='purple')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
axes[1, 1].set_title('Generalization Gap')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_training_subplots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_training_subplots.png")

# 8. Class Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

train_counts = pd.Series(y_train).value_counts().sort_index()
val_counts = pd.Series(y_val).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

axes[0].bar(range(num_classes), [train_counts.get(i, 0) for i in range(num_classes)], color=colors, edgecolor='black')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Training Set (n={len(y_train)})')
axes[0].set_xticks(range(num_classes))
axes[0].set_xticklabels(class_names, rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(range(num_classes), [val_counts.get(i, 0) for i in range(num_classes)], color=colors, edgecolor='black')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Validation Set (n={len(y_val)})')
axes[1].set_xticks(range(num_classes))
axes[1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(range(num_classes), [test_counts.get(i, 0) for i in range(num_classes)], color=colors, edgecolor='black')
axes[2].set_xlabel('Class')
axes[2].set_ylabel('Count')
axes[2].set_title(f'Test Set (n={len(y_test)})')
axes[2].set_xticks(range(num_classes))
axes[2].set_xticklabels(class_names, rotation=45, ha='right')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_class_distribution.png")

# 9. ROC Curves
plt.figure(figsize=(12, 10))
y_test_bin = label_binarize(all_labels, classes=range(num_classes))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_roc_curves.png")

# 10. Precision-Recall Curves
plt.figure(figsize=(12, 10))
for i in range(num_classes):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
    pr_auc = auc(recall_curve, precision_curve)
    plt.plot(recall_curve, precision_curve, color=colors[i], linewidth=2, label=f'{class_names[i]} (AUC={pr_auc:.3f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_precision_recall_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_precision_recall_curves.png")

# 11. t-SNE Visualization
print("Computing t-SNE visualization...")
max_samples = 2000
if len(all_features) > max_samples:
    indices = np.random.choice(len(all_features), max_samples, replace=False)
    features_sample = all_features[indices]
    labels_sample = np.array(all_labels)[indices]
else:
    features_sample = all_features
    labels_sample = np.array(all_labels)

try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
except TypeError:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

features_tsne = tsne.fit_transform(features_sample)

plt.figure(figsize=(12, 10))
for i in range(num_classes):
    mask = labels_sample == i
    plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], c=[colors[i]], label=class_names[i], 
                alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.title('t-SNE Visualization of Learned Features', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_tsne_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_tsne_features.png")

# 12. PCA Visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_sample)

plt.figure(figsize=(12, 10))
for i in range(num_classes):
    mask = labels_sample == i
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], c=[colors[i]], label=class_names[i], 
                alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('PCA Visualization of Learned Features', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_pca_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_pca_features.png")

# 13. Confidence Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

max_probs = np.max(all_probs, axis=1)
correct_mask = np.array(all_preds) == np.array(all_labels)

axes[0].hist(max_probs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=np.mean(max_probs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_probs):.3f}')
axes[0].set_xlabel('Prediction Confidence', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Prediction Confidence', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(max_probs[correct_mask], bins=30, alpha=0.7, label=f'Correct (n={correct_mask.sum()})', color='green', edgecolor='black')
axes[1].hist(max_probs[~correct_mask], bins=30, alpha=0.7, label=f'Incorrect (n={(~correct_mask).sum()})', color='red', edgecolor='black')
axes[1].set_xlabel('Prediction Confidence', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_confidence_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_confidence_distribution.png")

# 14. Confidence Box Plot
plt.figure(figsize=(12, 8))
confidence_by_class = [max_probs[np.array(all_labels) == i] for i in range(num_classes)]
bp = plt.boxplot(confidence_by_class, labels=class_names, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.xlabel('Activity', fontsize=12)
plt.ylabel('Prediction Confidence', fontsize=12)
plt.title('Prediction Confidence by Class', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_confidence_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_confidence_boxplot.png")

# 15. Misclassification Analysis
misclassified_indices = np.where(~correct_mask)[0]
if len(misclassified_indices) > 0:
    misclassified_true = np.array(all_labels)[misclassified_indices]
    misclassified_pred = np.array(all_preds)[misclassified_indices]

    misclass_pairs = {}
    for true, pred in zip(misclassified_true, misclassified_pred):
        pair = (class_names[true], class_names[pred])
        misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1

    sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)[:10]

    if len(sorted_pairs) > 0:
        plt.figure(figsize=(12, 8))
        pair_labels = [f"{p[0][0]}→{p[0][1]}" for p in sorted_pairs]
        pair_counts = [p[1] for p in sorted_pairs]

        bars = plt.barh(range(len(pair_labels)), pair_counts, color='coral', edgecolor='black')
        plt.yticks(range(len(pair_labels)), pair_labels)
        plt.xlabel('Number of Misclassifications', fontsize=12)
        plt.ylabel('True → Predicted', fontsize=12)
        plt.title('Most Common Misclassifications', fontsize=14)
        plt.gca().invert_yaxis()

        for bar, count in zip(bars, pair_counts):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'uci_misclassification_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: uci_misclassification_analysis.png")
else:
    print("✓ No misclassifications to analyze")

# 16. Metrics Radar Chart
def create_radar_chart(categories, values, title, filename):
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = list(values) + [values[0]]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 100)
    ax.set_title(title, size=14, y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

metrics_names = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Weighted F1']
metrics_values = [test_acc, macro_f1*100, macro_precision*100, macro_recall*100, weighted_f1*100]
create_radar_chart(metrics_names, metrics_values, 'Model Performance Metrics', 'uci_metrics_radar.png')
print("✓ Saved: uci_metrics_radar.png")

# 17. Error Rate by Class
plt.figure(figsize=(12, 8))
error_rates = [100 - acc for acc in per_class_acc]
bar_colors = ['red' if err > 10 else 'orange' if err > 5 else 'green' for err in error_rates]

bars = plt.bar(range(len(class_names)), error_rates, color=bar_colors, edgecolor='black')
plt.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Target Error (5%)')
plt.axhline(y=np.mean(error_rates), color='blue', linestyle='-.', linewidth=2, label=f'Mean Error ({np.mean(error_rates):.1f}%)')
plt.xlabel('Activity', fontsize=12)
plt.ylabel('Error Rate (%)', fontsize=12)
plt.title('Error Rate by Class', fontsize=14)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

for bar, err in zip(bars, error_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{err:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_error_rate.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_error_rate.png")

# 18. Sample Predictions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

sample_indices = np.random.choice(len(all_labels), 6, replace=False)

for idx, ax in zip(sample_indices, axes):
    true_label = all_labels[idx]
    pred_label = all_preds[idx]
    confidence = all_probs[idx][pred_label]
    
    ax.bar(range(num_classes), all_probs[idx], color=colors, edgecolor='black')
    ax.axvline(x=true_label, color='green', linestyle='--', linewidth=2, label='True')
    ax.axvline(x=pred_label, color='red', linestyle=':', linewidth=2, label='Pred')
    
    result = "✓" if true_label == pred_label else "✗"
    ax.set_title(f'{result} True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({confidence:.2f})', 
                 fontsize=10, color='green' if true_label == pred_label else 'red')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([c[:8] for c in class_names], rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Sample Predictions with Probability Distributions', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_sample_predictions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_sample_predictions.png")

# 19. Metrics Heatmap
metrics_matrix = np.array([per_class_acc, precisions, recalls, f1_scores_list]).T

plt.figure(figsize=(10, 8))
sns.heatmap(metrics_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            yticklabels=class_names, vmin=0, vmax=100, annot_kws={'size': 12})
plt.title('Per-Class Metrics Heatmap', fontsize=14)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Activity', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_metrics_heatmap.png")

# 20. Training Progress
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(epochs_range, history['val_acc'], 'b-', linewidth=2, marker='o', markersize=3, label='Val Acc')
ax1.plot(epochs_range, history['test_acc'], 'g-', linewidth=2, marker='s', markersize=3, label='Test Acc')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim([0, 105])
ax1.legend(loc='lower left')

ax2 = ax1.twinx()
ax2.plot(epochs_range, np.cumsum(history['train_loss']), 'r--', linewidth=2)
ax2.set_ylabel('Cumulative Training Loss', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Training Progress: Accuracy vs Cumulative Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_training_progress.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_training_progress.png")

# 21. Model Architecture Diagram
plt.figure(figsize=(10, 14))
plt.text(0.5, 0.95, 'Robust MS-GRS-BiLSTM Architecture', fontsize=16, fontweight='bold', 
         ha='center', transform=plt.gca().transAxes)

architecture_text = f"""
┌─────────────────────────────────────────┐
│           Input Layer                   │
│      (batch, {seq_len}, {features_per_step})                    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Feature Extractor Block 1        │
│  Linear → BN → GELU → Dropout           │
│  + Residual Skip                        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Feature Extractor Block 2        │
│  Linear → BN → GELU → Dropout           │
│  + Residual Skip                        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        BiLSTM (2 layers)                │
│    Hidden: {HIDDEN_SIZE} total representation         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Gated Residual Unit              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Temporal Attention               │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │ Skip Projection   │
        │ Aux Classifier    │
        └─────────┬─────────┘
                  │
┌─────────────────▼───────────────────────┐
│           Main Classifier               │
│  LayerNorm → Linear → GELU → Dropout    │
│  → Linear → GELU → Dropout → Linear     │
└─────────────────────────────────────────┘

Training tricks:
- Mixup (alpha={MIXUP_ALPHA})
- Label smoothing = 0.05
- OneCycleLR
- Feature noise std = {noise_std}

Total Parameters: {total_params:,}
Model Size: {model_size_mb:.4f} MB
"""

plt.text(0.5, 0.45, architecture_text, fontsize=9, family='monospace',
         ha='center', va='center', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uci_model_architecture.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: uci_model_architecture.png")

# =====================================================
#  Final Summary
# =====================================================
summary_text = f"""
{'='*70}
SUMMARY - Robust MS-GRS-BiLSTM with Mixup
{'='*70}
Model: MSGRSBiLSTM_V4
Dataset: UCI Human Activity Recognition
Features Used: {len(all_feature_cols)} features (ALL)
Sequence Length: {seq_len}
Features per Step: {features_per_step}
Number of Classes: {num_classes}
Class Names: {class_names}

MODEL SPECIFICATIONS:
---------------------
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Model Size: {model_size_bytes:,} bytes ({model_size_kb:.2f} KB, {model_size_mb:.4f} MB)
Hidden Size: {HIDDEN_SIZE}
Number of Layers: {NUM_LAYERS}
Dropout: {DROPOUT}
Scales: [3, 5, 7]

REGULARIZATION TECHNIQUES:
--------------------------
- Mixup Augmentation (alpha={MIXUP_ALPHA})
- Label Smoothing Cross Entropy (smoothing=0.05)
- Weight Decay: {WEIGHT_DECAY}
- Spatial Dropout: 0.2
- Multiple Dropout layers: {DROPOUT}-{DROPOUT+0.1}

TRAINING CONFIGURATION:
-----------------------
Learning Rate: {LEARNING_RATE}
Batch Size: {BATCH_SIZE}
Optimizer: AdamW
Weight Decay: {WEIGHT_DECAY}
Scheduler: OneCycleLR
Epochs Trained: {len(history['train_loss'])}
Training Time: {training_time:.2f} seconds
Feature Noise Std: {noise_std}
Best Validation Loss: {best_val_loss:.4f}

DATA SPLIT:
-----------
Training Samples: {len(y_train)}
Validation Samples: {len(y_val)}
Test Samples: {len(y_test)}

PERFORMANCE METRICS:
--------------------
Best Test Accuracy: {best_test_acc:.2f}%
Final Test Accuracy: {test_acc:.2f}%
Macro F1-Score: {macro_f1:.4f}
Weighted F1-Score: {weighted_f1:.4f}
Macro Precision: {macro_precision:.4f}
Macro Recall: {macro_recall:.4f}
Cohen's Kappa: {kappa:.4f}
Matthews Correlation Coefficient: {mcc:.4f}

INFERENCE SPEED:
----------------
Avg Batch Time: {avg_batch_time:.2f} ms
Avg Sample Time: {avg_sample_time:.4f} ms
Throughput: {1000/avg_sample_time:.0f} samples/sec

PER-CLASS ACCURACY:
-------------------
"""

for name, acc in zip(class_names, per_class_acc):
    summary_text += f"  {name}: {acc:.2f}%\n"

summary_text += f"""
VISUALIZATIONS GENERATED: 21
{'='*70}
"""

print(summary_text)

with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\n✅ All results saved to: {OUTPUT_DIR}")
print(f"📊 Total visualizations generated: 21")
print(f"📁 Model size: {model_size_mb:.4f} MB")
print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")