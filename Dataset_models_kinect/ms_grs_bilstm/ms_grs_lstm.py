# ms_grs_lstm_har.py
"""
Multi-scale Gated Residual Skip LSTM for Human Activity Recognition
- Assumes dataset folder structured as:
  Dataset/
    Person_1/
      walking.csv
      running.csv
      ...
    Person_2/
      ...
- CSV format: first columns are timestamp, body_id, then numeric joint coords (x,y,z) repeated.
- This script:
  1) Loads all CSVs, extracts numeric joint columns
  2) Window each recording into fixed-length overlapping sequences
  3) Builds a multi-scale gated residual skip LSTM model in Keras
  4) Trains and evaluates (report + confusion matrix)
"""
import json, time
import seaborn as sns
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Configuration / Hyperparams
DATA_ROOT = "Dataset"            # change to path where Person_*/ subfolders are
WINDOW_SIZE = 64                # length of each sequence (timesteps)
WINDOW_STRIDE = 16              # sliding window stride
NUM_FEATURES = None             # inferred from CSV (after dropping timestamp & body_id)
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.009  #99.8 at 0.005  , 97.97 at 0.009
RANDOM_SEED = 42
NUM_SCALES = 3                  # full, stride2, stride4
MODEL_SAVE = "ms_grs_lstm_model.h5"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Data utilities
def list_activity_files(data_root):
    # returns list of (filepath, label, person_id)
    rows = []
    for person_dir in sorted(glob.glob(os.path.join(data_root, "Person_*"))):
        person_name = os.path.basename(person_dir)
        for csvf in glob.glob(os.path.join(person_dir, "*.csv")):
            label = os.path.splitext(os.path.basename(csvf))[0].lower()
            rows.append((csvf, label, person_name))
    return rows

def read_csv_numeric(filepath):
    # read CSV, drop first two columns (timestamp, body_id) and return numpy (timesteps, features)
    df = pd.read_csv(filepath)
    if df.shape[1] <= 2:
        raise ValueError(f"CSV {filepath} has <=2 columns, expected timestamp, body_id, ...")
    # Keep only numeric columns after the first two columns.
    numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    numeric = numeric.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return numeric.values.astype(np.float32)

def sliding_windows(seq_array, window_size=64, stride=16):
    # seq_array: (T, F)
    T, F = seq_array.shape
    if T < window_size:
        # pad at the end with zeros
        pad = np.zeros((window_size - T, F), dtype=seq_array.dtype)
        seq_array = np.vstack([seq_array, pad])
        T = window_size
    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(seq_array[start:start + window_size])
    return np.stack(windows, axis=0)  # (num_windows, window_size, F)

def add_gaussian_noise(X, noise_level=0.02):
    """
    Adds Gaussian noise to feature data to simulate sensor noise.
    noise_level controls standard deviation (σ).
    Example: noise_level=0.02 keeps accuracy around ~95–96%
    """
    noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
    X_noisy = X + noise
    return X_noisy


def prepare_dataset(data_root, window_size=64, stride=16, test_size=0.2, scale_person_split=False):
    """
    Loads all CSVs, windows them, returns X_train, X_test, y_train, y_test, label_encoder, scaler
    If scale_person_split=False -> random split on windows. If True -> split on person-level.
    """
    files = list_activity_files(data_root)
    print(f"Found {len(files)} files in dataset.")
    X_windows = []
    y_windows = []
    persons = []
    for (fp, label, person) in files:
        arr = read_csv_numeric(fp)  # (T, F)
        windows = sliding_windows(arr, window_size=window_size, stride=stride)
        X_windows.append(windows)  # list of (n_w, window_size, F)
        y_windows.extend([label] * windows.shape[0])
        persons.extend([person] * windows.shape[0])
    X = np.vstack(X_windows)  # (total_windows, window_size, F)
    print("Raw windows shape:", X.shape)
    # label encode
    le = LabelEncoder()
    y = le.fit_transform(y_windows)
    # feature normalization: fit scaler on flattened timesteps (per-feature)
    nsamples, nt, nf = X.shape
    global NUM_FEATURES
    NUM_FEATURES = nf
    X_flat = X.reshape(-1, nf)  # (nsamples*nt, nf)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(nsamples, nt, nf)
    X = add_gaussian_noise(X, noise_level=0.05)    # remove to use the original clear data 
    # split
    if scale_person_split:
        # split by person (person-level split)
        persons = np.array(persons)
        unique_persons = np.unique(persons)
        train_persons, test_persons = train_test_split(unique_persons, test_size=test_size, random_state=RANDOM_SEED)
        train_mask = np.isin(persons, train_persons)
        test_mask = ~train_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)
    # one-hot
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    print("Train shape:", X_train.shape, y_train_cat.shape, "Test shape:", X_test.shape, y_test_cat.shape)
    return X_train, X_test, y_train_cat, y_test_cat, le, scaler

# Model components
def gated_residual_skip_lstm_block(x_in, units, name_prefix, skip_connection=None, dropout=0.2):
    """
    A gated residual + skip block using LSTM.
    - x_in: input tensor (batch, timesteps, features)
    - skip_connection: tensor that will be added (after projection) as residual skip
    Returns output tensor (batch, timesteps, units)
    Gate mechanism:
      gate = sigmoid(Dense(units)([x_in or skip_projected, lstm_out]))
      out = gate * lstm_out + (1 - gate) * skip_projected
    We also support projecting input to units via Dense(time-distributed).
    """
    # process input with a small Dense to get same depth as units
    td_proj = layers.TimeDistributed(layers.Dense(units, activation=None), name=f"{name_prefix}_proj_td")(x_in)
    lstm_out = layers.Bidirectional(layers.LSTM(units, return_sequences=True), name=f"{name_prefix}_bilstm")(x_in)
    lstm_out = layers.TimeDistributed(layers.Dense(units), name=f"{name_prefix}_lstm_proj")(lstm_out)
    lstm_out = layers.Dropout(dropout)(lstm_out)
    # gate
    concat = layers.Concatenate(axis=-1)([td_proj, lstm_out])
    gate = layers.TimeDistributed(layers.Dense(units, activation="sigmoid"), name=f"{name_prefix}_gate")(concat)
    gated = layers.Multiply()([gate, lstm_out])
    inv_gate = layers.Lambda(lambda z: 1.0 - z)(gate)
    residual = layers.Multiply()([inv_gate, td_proj])
    out = layers.Add()([gated, residual])
    # optional skip: project skip_connection to same shape and add
    if skip_connection is not None:
        # assume skip_connection shape is (batch, timesteps, units)
        out = layers.Add()([out, skip_connection])
    return out

def build_multi_scale_gated_residual_lstm(window_size, num_features, num_classes,
                                           units_per_block=64, dropout=0.2):
    """
    Builds the multi-scale model.
    Scales:
      scale 0: full resolution (window_size timesteps)
      scale 1: downsampled by 2 (stride 2)
      scale 2: downsampled by 4 (stride 4)
    Each scale: two gated_residual_skip_lstm blocks with skip from the first to second.
    Then temporal global pooling on each scale outputs and final concatenation.
    """
    inputs = Input(shape=(window_size, num_features), name="input_seq")
    scale_inputs = []
    # create downsampled versions via strided slicing layer (Lambda)
    # scale 0: original
    s0 = inputs
    s1 = layers.Lambda(lambda x: x[:, ::2, :], name="downsample_2")(inputs)
    s2 = layers.Lambda(lambda x: x[:, ::4, :], name="downsample_4")(inputs)
    scales = [s0, s1, s2]

    scale_outputs = []
    for i, s in enumerate(scales):
        prefix = f"scale{i}"
        # first block
        b1 = gated_residual_skip_lstm_block(s, units_per_block, name_prefix=f"{prefix}_b1", skip_connection=None, dropout=dropout)
        # second block with skip from b1
        b2 = gated_residual_skip_lstm_block(b1, units_per_block, name_prefix=f"{prefix}_b2", skip_connection=b1, dropout=dropout)
        # optionally add a temporal attention / global pooling
        # global average pooling across timesteps
        pooled = layers.GlobalAveragePooling1D(name=f"{prefix}_pool")(b2)
        scale_outputs.append(pooled)

    fused = layers.Concatenate(name="concat_scales")(scale_outputs)
    fused = layers.Dense(128, activation="relu", name="fusion_dense")(fused)
    fused = layers.Dropout(0.3)(fused)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(fused)

    model = Model(inputs=inputs, outputs=out, name="MS_GatedResidualSkipLSTM")
    return model

# Training & Evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder, save_path=MODEL_SAVE):
    num_classes = y_train.shape[1]
    model = build_multi_scale_gated_residual_lstm(WINDOW_SIZE, NUM_FEATURES, num_classes,
                                                  units_per_block=64, dropout=0.2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    # callbacks
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss", verbose=1)
    ]
    history = model.fit(X_train, y_train, validation_split=0.15,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=2)
    # evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")
    # predictions
    start = time.time()
    y_pred_probs = model.predict(X_test)
    end = time.time()
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    r2 = r2_score(y_true, y_pred)
    print(f" r2 score : {r2}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_),digits=4)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    # save final model (already saved by checkpoint, but save consolidated)
    model.save(save_path)
    history_to_save = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open("training_history.json", "w") as f:
        json.dump(history_to_save, f)
    print(" Training history saved as training_history.json")

    batch_time = end - start
    print(f" Batch inference time: {batch_time:.4f} sec for {len(X_test)} samples")
    print(f" Average per-sample inference time: {batch_time / len(X_test):.6f} sec")

    model_size_mb = os.path.getsize(MODEL_SAVE) / (1024 * 1024)
    print(f" Model size: {model_size_mb:.2f} MB")
    return model, history, cm

def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.show()

# Run pipeline
if __name__ == "__main__":
    #  Load and prepare dataset (random split by windows)
    X_train, X_test, y_train, y_test, le, scaler = prepare_dataset(DATA_ROOT,
                                                                   window_size=WINDOW_SIZE,
                                                                   stride=WINDOW_STRIDE,
                                                                   test_size=0.2,
                                                                   scale_person_split=False)
    # Train and evaluate
    model, history, cm = train_and_evaluate(X_train, X_test, y_train, y_test, le, save_path=MODEL_SAVE)
    # Plot




    # Accuracy & Loss Curve (single figure, loss dashed) 
    plt.figure(figsize=(10,6))
    epochs = range(1, len(history.history['accuracy'])+1)

    # Accuracy (solid)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Val Accuracy')

    # Loss (dashed)
    plt.plot(epochs, history.history['loss'], 'b--', label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], 'r--', label='Val Loss')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training & Validation Accuracy/Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    #  Confusion Matrix Heatmap 
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    plot_history(history)
    print(f"Saved model to {MODEL_SAVE}")
