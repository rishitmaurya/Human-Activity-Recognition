# evaluate_grs_bilstm_split.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
DATASET_PATH = "Dataset"
MODEL_PATH = "grs_bilstm\\grs_bilstm_har_final.keras"
CLASSES_PATH = "grs_bilstm\\label_classes.npy"

# --- 1️⃣ Load dataset (same as in training) ---
def load_dataset(path):
    sequences, labels = [], []
    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        if not os.path.isdir(person_path): 
            continue
        for file in os.listdir(person_path):
            if file.endswith('.csv'):
                label = file.replace('.csv','')
                df = pd.read_csv(os.path.join(person_path, file))
                sequences.append(df.iloc[:, 2:].values)  # Skip timestamp & body_id
                labels.append(label)
    return sequences, labels

MAX_LEN = 128  # same as in training
sequences, labels = load_dataset(DATASET_PATH)
NUM_FEATURES = sequences[0].shape[1]
print(f"✅ Loaded {len(sequences)} total samples")

# --- 2️⃣ Padding and Encoding ---
X = pad_sequences(sequences, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')

# Encode labels
classes = np.load(CLASSES_PATH)
encoder = LabelEncoder()
encoder.fit(classes)
y = encoder.transform(labels)
y = to_categorical(y)

# --- 3️⃣ Recreate same train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
)
print(f" Train: {X_train.shape}, Test: {X_test.shape}")

# --- 4️⃣ Load model ---
model = load_model(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# --- 5️⃣ Evaluate ---
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# --- 6️⃣ Classification report (4 decimals) ---
print("\n📊 Classification Report (4 decimal places):")
report = classification_report(y_true_classes, y_pred_classes, target_names=classes, digits=4)
print(report)



