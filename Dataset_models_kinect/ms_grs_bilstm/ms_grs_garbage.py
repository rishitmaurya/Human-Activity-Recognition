# evaluate_saved_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import glob, os
from ms_grs_lstm import list_activity_files, read_csv_numeric, sliding_windows

# same constants as your training script
DATA_ROOT = "Dataset"
WINDOW_SIZE = 64
WINDOW_STRIDE = 16
MODEL_PATH = "ms_grs_bilstm\\ms_grs_lstm_model.h5"

# Recreate dataset structure (only test split)
def load_test_data(data_root, window_size=64, stride=16):
    files = list_activity_files(data_root)
    X_windows, y_labels = [], []
    for fp, label, _ in files:
        arr = read_csv_numeric(fp)
        windows = sliding_windows(arr, window_size=window_size, stride=stride)
        X_windows.append(windows)
        y_labels.extend([label] * windows.shape[0])
    X = np.vstack(X_windows)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    nsamples, nt, nf = X.shape
    X_flat = X.reshape(-1, nf)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(nsamples, nt, nf)
    y_cat = to_categorical(y)
    return X, y_cat, le

# Load data, model, and evaluate 
X, y_cat, le = load_test_data(DATA_ROOT, WINDOW_SIZE, WINDOW_STRIDE)
model = load_model(MODEL_PATH)

print("\n Model loaded successfully.")

# Predictions and report 
y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_cat, axis=1)

print("\n Classification Report (4 decimal places):")
print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\n Confusion Matrix:\n", cm)
