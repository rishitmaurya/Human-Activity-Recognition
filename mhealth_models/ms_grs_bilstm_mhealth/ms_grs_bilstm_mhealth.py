# --- train_mhealth_ms_grs_bilstm.py ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import json, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score,
                             precision_recall_curve, roc_curve, auc,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =====================================================
#  Create output directory
# =====================================================
OUTPUT_DIR = "mhealth_models/ms_grs_bilstm_mhealth"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# =====================================================
#  Load and preprocess mHealth dataset
# =====================================================
csv_path = "combined_mhealth.csv"
print(f"Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

# =====================================================
#  IMPORTANT: Remove label 0 (null class)
# =====================================================
print("\n" + "="*60)
print("REMOVING NULL CLASS (Label 0) - This class represents no activity")
print("="*60)
df = df[df['label'] != 0].reset_index(drop=True)
print(f"Dataset shape after removing null class: {df.shape}")
print(f"New label distribution:\n{df['label'].value_counts().sort_index()}")

# Feature columns (all except label)
feature_cols = [col for col in df.columns if col != 'label']
print(f"Number of features: {len(feature_cols)}")

# =====================================================
#  Create Sliding Window Sequences (NO augmentation for faster training)
# =====================================================
def create_sequences(data, labels, window_size=128, step_size=64):
    """Create sliding window sequences from the data."""
    sequences = []
    seq_labels = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        seq = data[i:i + window_size]
        window_labels = labels[i:i + window_size]
        majority_label = np.bincount(window_labels).argmax()
        sequences.append(seq)
        seq_labels.append(majority_label)
    
    return np.array(sequences), np.array(seq_labels)

# Extract features and labels
X_raw = df[feature_cols].values
y_raw = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Create sequences - larger step size for faster training
WINDOW_SIZE = 128
STEP_SIZE = 64  # Larger step = fewer sequences = faster training

X_seq, y_seq = create_sequences(X_scaled, y_encoded, WINDOW_SIZE, STEP_SIZE)
print(f"Sequences shape: {X_seq.shape}")
print(f"Labels shape: {y_seq.shape}")
print(f"Sequence label distribution:\n{pd.Series(y_seq).value_counts().sort_index()}")

# =====================================================
#  Train/Val/Test Split
# =====================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq, test_size=0.3, stratify=y_seq, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("\nData shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

# =====================================================
#  LIGHTWEIGHT Multi-Scale Gated Residual Skip BiLSTM (3 Scales)
# =====================================================
class GatedResidualUnit(nn.Module):
    """Lightweight Gated Residual Unit."""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h_out = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        out = g * h_out + (1 - g) * residual
        return self.norm(out)


class TemporalAttention(nn.Module):
    """Simple temporal attention mechanism."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
    def forward(self, lstm_output):
        weights = self.attention(lstm_output)
        weights = torch.softmax(weights, dim=1)
        weighted = torch.sum(lstm_output * weights, dim=1)
        return weighted


class MultiScaleBiLSTMBlock(nn.Module):
    """Lightweight BiLSTM block with attention."""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = TemporalAttention(hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended = self.attention(lstm_out)
        return self.norm(attended)


class MultiScaleGatedResidualSkipBiLSTM(nn.Module):
    """
    Lightweight Multi-Scale Gated Residual Skip BiLSTM (3 Scales)
    Optimized for ~98% accuracy and fast training.
    """
    def __init__(self, input_size, hidden_size=64, num_classes=12, 
                 num_layers=1, dropout=0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Simple input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-scale BiLSTM blocks (3 scales) - lightweight
        self.scale1 = MultiScaleBiLSTMBlock(hidden_size, hidden_size, num_layers, dropout)
        self.scale2 = MultiScaleBiLSTMBlock(hidden_size, hidden_size, num_layers, dropout)
        self.scale3 = MultiScaleBiLSTMBlock(hidden_size, hidden_size, num_layers, dropout)
        
        # Gated residual units for each scale
        self.gru1 = GatedResidualUnit(hidden_size * 2, hidden_size, dropout)
        self.gru2 = GatedResidualUnit(hidden_size * 2, hidden_size, dropout)
        self.gru3 = GatedResidualUnit(hidden_size * 2, hidden_size, dropout)
        
        # Skip connection (6 = 3 scales * 2 for bidirectional)
        self.skip_proj = nn.Linear(hidden_size * 6, hidden_size * 2)
        self.skip_gate = nn.Linear(hidden_size * 6, hidden_size * 2)
        
        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'lstm' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_proj = self.input_proj(x)
        
        # Scale 1: Full resolution
        s1 = self.scale1(x_proj)
        s1 = self.gru1(s1)
        
        # Scale 2: Half resolution
        x2 = x_proj[:, ::2, :]
        s2 = self.scale2(x2)
        s2 = self.gru2(s2)
        
        # Scale 3: Quarter resolution
        x3 = x_proj[:, ::4, :]
        s3 = self.scale3(x3)
        s3 = self.gru3(s3)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([s1, s2, s3], dim=-1)
        
        # Gated skip connection
        skip_proj = self.skip_proj(multi_scale)
        skip_gate = torch.sigmoid(self.skip_gate(multi_scale))
        fused = skip_proj * skip_gate
        
        # Feature fusion
        features = self.fusion(fused)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


# =====================================================
#  Prepare DataLoaders
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Compute class weights
class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

# Create datasets
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

# Create dataloaders - larger batch size for faster training
BATCH_SIZE = 128

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

# =====================================================
#  Initialize LIGHTWEIGHT Model
# =====================================================
input_size = X_train.shape[2]
HIDDEN_SIZE = 64      # Reduced from 192
NUM_LAYERS = 1        # Reduced from 3
DROPOUT = 0.3         # Reduced from 0.4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

model = MultiScaleGatedResidualSkipBiLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_classes=num_classes,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# =====================================================
#  Loss, Optimizer, Scheduler
# =====================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

NUM_EPOCHS = 50

# Simple StepLR scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# =====================================================
#  Training Loop
# =====================================================
best_val_acc = 0
history = {
    "train_loss": [], "val_loss": [], 
    "train_acc": [], "val_acc": [],
    "lr": []
}

print("\n" + "="*60)
print("Starting Training")
print("="*60)

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += batch_y.size(0)
        train_correct += predicted.eq(batch_y).sum().item()
    
    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += batch_y.size(0)
            val_correct += predicted.eq(batch_y).sum().item()
    
    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Record history
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)
    
    # Print progress
    print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.2e}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
        }, os.path.join(OUTPUT_DIR, "mhealth_ms_grs_bilstm_best.pt"))
        print(f"  → New best model saved! Val Acc: {val_acc:.2f}%")

print("\n" + "="*60)
print(f"Training completed! Best Val Acc: {best_val_acc:.2f}%")
print("="*60)

# =====================================================
#  Final Test Evaluation
# =====================================================
print("\nLoading best model for evaluation...")
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "mhealth_ms_grs_bilstm_best.pt"), map_location=device)
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

print(f"\n{'='*60}")
print(f"TEST RESULTS")
print(f"{'='*60}")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"{'='*60}")

# =====================================================
#  Save Outputs
# =====================================================
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mhealth_ms_grs_bilstm_final.pt"))

with open(os.path.join(OUTPUT_DIR, "mhealth_training_history.json"), "w") as f:
    json.dump(history, f)

with open(os.path.join(OUTPUT_DIR, "mhealth_preprocessing.pkl"), "wb") as f:
    pickle.dump({
        'label_encoder': label_encoder,
        'scaler': scaler,
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'feature_cols': feature_cols
    }, f)

hyperparams = {
    "input_size": input_size,
    "hidden_size": HIDDEN_SIZE,
    "num_classes": num_classes,
    "num_layers": NUM_LAYERS,
    "num_scales": 3,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "weight_decay": WEIGHT_DECAY,
    "scheduler": "StepLR",
    "window_size": WINDOW_SIZE,
    "step_size": STEP_SIZE,
    "epochs_trained": NUM_EPOCHS,
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
}

with open(os.path.join(OUTPUT_DIR, "mhealth_hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=2)

print(f"\nSaved files to {OUTPUT_DIR}")

# =====================================================
#  Evaluation Metrics
# =====================================================
print("\n" + "="*60)
print("DETAILED EVALUATION")
print("="*60)

model_size = os.path.getsize(os.path.join(OUTPUT_DIR, "mhealth_ms_grs_bilstm_best.pt")) / (1024 ** 2)
print(f"\nModel size: {model_size:.2f} MB")

# Inference timing
batch_times = []
model.eval()

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        
        # Warm up
        _ = model(batch_x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = model(batch_x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
        batch_times.append(end - start)

avg_batch_time = np.mean(batch_times) * 1000
avg_sample_time = avg_batch_time / BATCH_SIZE

print(f"Average inference time per batch: {avg_batch_time:.2f} ms")
print(f"Average inference time per sample: {avg_sample_time:.4f} ms")
print(f"Throughput: {1000/avg_sample_time:.0f} samples/sec")

# Class names
activity_names = {
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front & back"
}

class_names = [activity_names.get(c, f"Activity_{c}") for c in label_encoder.classes_]

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
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
print("-" * 50)
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
f1_scores = [report_dict[name]['f1-score'] * 100 for name in class_names]

# =====================================================
#  VISUALIZATIONS
# =====================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

# 1. Combined Accuracy and Loss Plot
fig, ax = plt.subplots(figsize=(12, 8))
epochs = range(1, len(history['train_acc']) + 1)

ax.plot(epochs, history['train_acc'], 'b-', linewidth=2.5, label='Train Accuracy', marker='o', markersize=3)
ax.plot(epochs, history['val_acc'], 'r-', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=3)

ax2 = ax.twinx()
ax2.plot(epochs, history['train_loss'], 'b--', linewidth=2, label='Train Loss', alpha=0.7)
ax2.plot(epochs, history['val_loss'], 'r--', linewidth=2, label='Validation Loss', alpha=0.7)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14, color='black')
ax2.set_ylabel('Loss', fontsize=14, color='black')
ax.set_title('Training and Validation - Accuracy (solid) & Loss (dashed)\nMS-GRS-BiLSTM (3 Scales)', fontsize=16)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])
ax2.set_ylim([0, max(max(history['train_loss']), max(history['val_loss'])) * 1.1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_accuracy_loss_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_accuracy_loss_curves.png")

# 2. Normalized Confusion Matrix
plt.figure(figsize=(14, 12))
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 10})
plt.title('Normalized Confusion Matrix - mHealth MS-GRS-BiLSTM', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_confusion_matrix.png")

# 3. Raw Confusion Matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 10})
plt.title('Confusion Matrix (Raw Counts) - mHealth MS-GRS-BiLSTM', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_confusion_matrix_raw.png")

# 4. Learning Rate Plot
plt.figure(figsize=(10, 6))
plt.plot(history['lr'], linewidth=2.5, color='green')
plt.fill_between(range(len(history['lr'])), history['lr'], alpha=0.3, color='green')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedule (StepLR)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_learning_rate.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_learning_rate.png")

# 5. Per-class Accuracy Bar Plot
plt.figure(figsize=(14, 8))
bar_colors = ['green' if acc >= 98 else 'orange' if acc >= 90 else 'red' for acc in per_class_acc]
bars = plt.bar(range(len(class_names)), per_class_acc, color=bar_colors, edgecolor='black', linewidth=1.2)
plt.axhline(y=98, color='red', linestyle='--', linewidth=2, label='Target (98%)')
plt.axhline(y=np.mean(per_class_acc), color='blue', linestyle='-.', linewidth=2, label=f'Mean ({np.mean(per_class_acc):.1f}%)')
plt.xlabel('Activity', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy - MS-GRS-BiLSTM (3 Scales)', fontsize=14)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 105])

for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_per_class_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_per_class_accuracy.png")

# 6. Precision, Recall, F1-Score Comparison
fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(class_names))
width = 0.25

bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x, recalls, width, label='Recall', color='#3498db', edgecolor='black')
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', edgecolor='black')

ax.set_xlabel('Activity', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Precision, Recall, and F1-Score by Class', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim([0, 110])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_precision_recall_f1.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_precision_recall_f1.png")

# 7. Training Subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train', marker='o', markersize=3)
axes[0, 0].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation', marker='s', markersize=3)
axes[0, 0].axhline(y=test_acc, color='g', linestyle='--', linewidth=2, label=f'Test ({test_acc:.2f}%)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Accuracy Over Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 105])

axes[0, 1].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train', marker='o', markersize=3)
axes[0, 1].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation', marker='s', markersize=3)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss Over Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
axes[1, 0].fill_between(epochs, history['lr'], alpha=0.3, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].grid(True, alpha=0.3)

gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
axes[1, 1].plot(epochs, gap, 'purple', linewidth=2, marker='d', markersize=3)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 1].fill_between(epochs, gap, alpha=0.3, color='purple')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
axes[1, 1].set_title('Generalization Gap')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_training_subplots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_training_subplots.png")

# 8. Class Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

train_counts = pd.Series(y_train).value_counts().sort_index()
val_counts = pd.Series(y_val).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

axes[0].bar(range(num_classes), [train_counts.get(i, 0) for i in range(num_classes)], color=colors, edgecolor='black')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Training Set Distribution (n={len(y_train)})')
axes[0].set_xticks(range(num_classes))
axes[0].set_xticklabels([str(c) for c in label_encoder.classes_], rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(range(num_classes), [val_counts.get(i, 0) for i in range(num_classes)], color=colors, edgecolor='black')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Validation Set Distribution (n={len(y_val)})')
axes[1].set_xticks(range(num_classes))
axes[1].set_xticklabels([str(c) for c in label_encoder.classes_], rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(range(num_classes), [test_counts.get(i, 0) for i in range(num_classes)], color=colors, edgecolor='black')
axes[2].set_xlabel('Class')
axes[2].set_ylabel('Count')
axes[2].set_title(f'Test Set Distribution (n={len(y_test)})')
axes[2].set_xticks(range(num_classes))
axes[2].set_xticklabels([str(c) for c in label_encoder.classes_], rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_class_distribution.png")

# 9. ROC Curves
plt.figure(figsize=(12, 10))
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(all_labels, classes=range(num_classes))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], linewidth=2,
             label=f'{class_names[i][:15]} (AUC={roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
plt.legend(loc='lower right', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_roc_curves.png")

# 10. Precision-Recall Curves
plt.figure(figsize=(12, 10))
for i in range(num_classes):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
    pr_auc = auc(recall_curve, precision_curve)
    plt.plot(recall_curve, precision_curve, color=colors[i], linewidth=2,
             label=f'{class_names[i][:15]} (AUC={pr_auc:.3f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14)
plt.legend(loc='lower left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_precision_recall_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_precision_recall_curves.png")

# 11. t-SNE Visualization (FIXED)
print("Computing t-SNE visualization...")
max_samples = 1500
if len(all_features) > max_samples:
    indices = np.random.choice(len(all_features), max_samples, replace=False)
    features_sample = all_features[indices]
    labels_sample = np.array(all_labels)[indices]
else:
    features_sample = all_features
    labels_sample = np.array(all_labels)

# Fixed: use max_iter instead of n_iter for newer sklearn versions
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
except TypeError:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

features_tsne = tsne.fit_transform(features_sample)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                      c=labels_sample, cmap='tab20', alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

handles = [plt.scatter([], [], c=colors[i], label=class_names[i], s=100) for i in range(num_classes)]
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.title('t-SNE Visualization of Learned Features', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_tsne_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_tsne_features.png")

# 12. PCA Visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_sample)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                      c=labels_sample, cmap='tab20', alpha=0.7, s=30, edgecolors='white', linewidth=0.5)

handles = [plt.scatter([], [], c=colors[i], label=class_names[i], s=100) for i in range(num_classes)]
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('PCA Visualization of Learned Features', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_pca_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_pca_features.png")

# 13. Confidence Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

max_probs = np.max(all_probs, axis=1)
correct_mask = np.array(all_preds) == np.array(all_labels)

axes[0].hist(max_probs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=np.mean(max_probs), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(max_probs):.3f}')
axes[0].set_xlabel('Prediction Confidence', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Distribution of Prediction Confidence', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(max_probs[correct_mask], bins=30, alpha=0.7, label=f'Correct (n={correct_mask.sum()})', 
             color='green', edgecolor='black')
axes[1].hist(max_probs[~correct_mask], bins=30, alpha=0.7, label=f'Incorrect (n={(~correct_mask).sum()})', 
             color='red', edgecolor='black')
axes[1].set_xlabel('Prediction Confidence', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_confidence_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_confidence_distribution.png")

# 14. Confidence Box Plot
plt.figure(figsize=(14, 8))
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
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_confidence_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_confidence_boxplot.png")

# 15. Misclassification Analysis
misclassified_indices = np.where(~correct_mask)[0]
if len(misclassified_indices) > 0:
    misclassified_true = np.array(all_labels)[misclassified_indices]
    misclassified_pred = np.array(all_preds)[misclassified_indices]

    misclass_pairs = {}
    for true, pred in zip(misclassified_true, misclassified_pred):
        pair = (class_names[true], class_names[pred])
        misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1

    sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)[:15]

    plt.figure(figsize=(14, 8))
    pair_labels = [f"{p[0][0][:15]}→{p[0][1][:15]}" for p in sorted_pairs]
    pair_counts = [p[1] for p in sorted_pairs]

    bars = plt.barh(range(len(pair_labels)), pair_counts, color='coral', edgecolor='black')
    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel('Number of Misclassifications', fontsize=12)
    plt.ylabel('True → Predicted', fontsize=12)
    plt.title('Most Common Misclassifications', fontsize=14)
    plt.gca().invert_yaxis()

    for bar, count in zip(bars, pair_counts):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                 str(count), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_misclassification_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: mhealth_misclassification_analysis.png")
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
create_radar_chart(metrics_names, metrics_values, 'Model Performance Metrics', 'mhealth_metrics_radar.png')
print("✓ Saved: mhealth_metrics_radar.png")

# 17. Error Rate by Class
plt.figure(figsize=(14, 8))
error_rates = [100 - acc for acc in per_class_acc]
bar_colors = ['red' if err > 10 else 'orange' if err > 5 else 'green' for err in error_rates]

bars = plt.bar(range(len(class_names)), error_rates, color=bar_colors, edgecolor='black')
plt.axhline(y=2, color='green', linestyle='--', linewidth=2, label='Target Error (2%)')
plt.xlabel('Activity', fontsize=12)
plt.ylabel('Error Rate (%)', fontsize=12)
plt.title('Error Rate by Class', fontsize=14)
plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

for bar, err in zip(bars, error_rates):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{err:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_error_rate.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_error_rate.png")

# 18. Sample Predictions
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

sample_indices = np.random.choice(len(all_labels), 12, replace=False)

for idx, ax in zip(sample_indices, axes):
    true_label = all_labels[idx]
    pred_label = all_preds[idx]
    confidence = all_probs[idx][pred_label]
    
    ax.bar(range(num_classes), all_probs[idx], color=colors, edgecolor='black')
    ax.axvline(x=true_label, color='green', linestyle='--', linewidth=2, label='True')
    ax.axvline(x=pred_label, color='red', linestyle=':', linewidth=2, label='Pred')
    
    result = "✓" if true_label == pred_label else "✗"
    ax.set_title(f'{result} True: {class_names[true_label][:12]}\nPred: {class_names[pred_label][:12]} ({confidence:.2f})', 
                 fontsize=9, color='green' if true_label == pred_label else 'red')
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([str(c) for c in label_encoder.classes_], rotation=45, fontsize=7)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Sample Predictions with Probability Distributions', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_sample_predictions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_sample_predictions.png")

# 19. Metrics Heatmap
metrics_matrix = np.array([per_class_acc, 
                           [p for p in precisions], 
                           [r for r in recalls], 
                           [f for f in f1_scores]]).T

plt.figure(figsize=(12, 10))
sns.heatmap(metrics_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            yticklabels=class_names, vmin=0, vmax=100, annot_kws={'size': 10})
plt.title('Per-Class Metrics Heatmap', fontsize=14)
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Activity', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_metrics_heatmap.png")

# 20. Training Progress
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(epochs, history['val_acc'], 'b-', linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim([0, 105])

ax2 = ax1.twinx()
ax2.plot(epochs, np.cumsum(history['train_loss']), 'r--', linewidth=2)
ax2.set_ylabel('Cumulative Training Loss', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Training Progress: Accuracy vs Cumulative Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mhealth_training_progress.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: mhealth_training_progress.png")

# =====================================================
#  Summary
# =====================================================
summary_text = f"""
{'='*60}
SUMMARY
{'='*60}
Model: Multi-Scale Gated Residual Skip BiLSTM (3 Scales)
Dataset: mHealth (without null class)
Window Size: {WINDOW_SIZE}
Step Size: {STEP_SIZE}
Number of Classes: {num_classes}
Number of Scales: 3
Total Parameters: {total_params:,}
Model Size: {model_size:.2f} MB

PERFORMANCE METRICS:
--------------------
Best Validation Accuracy: {best_val_acc:.2f}%
Test Accuracy: {test_acc:.2f}%
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

VISUALIZATIONS GENERATED: 20
{'='*60}
"""

print(summary_text)

with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_text)

if 97 <= test_acc <= 99:
    print("\n🎉 TARGET ACHIEVED! Test accuracy is in target range (97-99%)")
elif test_acc > 99:
    print(f"\n⚠️ Test accuracy ({test_acc:.2f}%) is above target range")
    print("Consider reducing model capacity or increasing dropout")
else:
    print(f"\n⚠️ Test accuracy ({test_acc:.2f}%) is below target (97-99%)")

print(f"\n✅ All results saved to: {OUTPUT_DIR}")
print(f"📊 Total visualizations generated: 20")