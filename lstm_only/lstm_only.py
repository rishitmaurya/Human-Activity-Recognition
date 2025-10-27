
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix



# Define paths and parameters

root_dir = "Dataset"   # Path to your dataset root
actions = ['bending', 'jumping', 'running', 'sitting', 'squat', 'standing', 'walking']
MAX_SEQ_LEN = 100       # pad/truncate all sequences to this many frames
NUM_FEATURES = 75       # (25 joints × 3 coordinates)
print("Preparing data from:", root_dir)


# Load all CSV files

X_list, y_list = [], []

for person_folder in sorted(os.listdir(root_dir)):
    person_path = os.path.join(root_dir, person_folder)
    if not os.path.isdir(person_path):
        continue

    for action in actions:
        file_path = os.path.join(person_path, f"{action}.csv")
        if not os.path.exists(file_path):
            print("Missing file:", file_path)
            continue

        df = pd.read_csv(file_path)

        # Drop unnecessary columns
        df = df.drop(['timestamp', 'body_id'], axis=1)

        # Keep only first 100 frames (truncate)
        values = df.values[:MAX_SEQ_LEN]

        # Standardize (z-score)
        scaler = StandardScaler()
        values = scaler.fit_transform(values)

        X_list.append(values)
        y_list.append(action)

print(" Loaded sequences:", len(X_list))


# Pad sequences (if < MAX_SEQ_LEN)

X = pad_sequences(X_list, dtype='float32', padding='post', maxlen=MAX_SEQ_LEN)
print("X shape:", X.shape)  # (samples, time_steps, features)


# Encode labels

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_list)
y = to_categorical(y_encoded)
print("y shape:", y.shape)
print("Class mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(MAX_SEQ_LEN, NUM_FEATURES)),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# Save model and label encoder
model.save("lstm_har_model.h5")
np.save("label_classes.npy", label_encoder.classes_)
print("\n Model and labels saved successfully!")

import json
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

# Load saved model
model = load_model("lstm_har_model.h5")

# Measure inference time per sample
times = []
for i in range(len(X_test)):
    x_sample = np.expand_dims(X_test[i], axis=0)
    start = time.time()
    _ = model.predict(x_sample, verbose=0)
    end = time.time()
    times.append(end - start)

avg_time = np.mean(times)
std_time = np.std(times)

print(f"\n Average inference time per sample: {avg_time*1000:.2f} ms")
print(f" Std deviation: {std_time*1000:.2f} ms")

#  batch inference time
start = time.time()
_ = model.predict(X_test, verbose=0)
end = time.time()
print(f"\n Total batch inference time: {(end - start):.2f} s")
print(f" Average batch inference per sample: {(end - start)/len(X_test)*1000:.2f} ms")

# Classification Report
# Predict class probabilities
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Decode labels 
class_names = label_encoder.classes_

print("\n Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, digits=4))

# Confusion Matrix 
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Model Size 
model_path = "lstm_har_model.h5"
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n Model Size: {model_size:.2f} MB")


plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.show()

# Loss and Accuracy graph
plt.figure(figsize=(10, 6))

epochs = range(1, len(history.history['accuracy']) + 1)

# Accuracy (solid lines)
plt.plot(epochs, history.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Val Accuracy')

# Loss (dashed lines)
plt.plot(epochs, history.history['loss'], 'b--', label='Train Loss')
plt.plot(epochs, history.history['val_loss'], 'r--', label='Val Loss')

plt.title("Training & Validation Accuracy/Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

