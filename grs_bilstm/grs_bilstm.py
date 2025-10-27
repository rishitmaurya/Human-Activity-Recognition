import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Bidirectional, Add, Multiply, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
import time, json

# Load Dataset
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
                sequences.append(df.iloc[:,2:].values)  # Skip timestamp & body_id
                labels.append(label)
    return sequences, labels

dataset_path = "Dataset"
sequences, labels = load_dataset(dataset_path)
print(f"Total sequences: {len(sequences)}, Classes: {set(labels)}")

# Padding & Encoding
MAX_LEN = 128
NUM_FEATURES = sequences[0].shape[1]

X = pad_sequences(sequences, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)
classes = encoder.classes_
print(f"Classes encoded: {classes}")

# Data Augmentation
def augment_data(X, y, sigma=0.05, timeshift=2):
    X_aug, y_aug = [], []
    for xi, yi in zip(X, y):
        # Gaussian noise
        noise = np.random.normal(0, sigma, xi.shape)
        X_aug.append(xi + noise)
        y_aug.append(yi)
        # Time shift (roll)
        for shift in range(1, timeshift+1):
            X_aug.append(np.roll(xi, shift, axis=0))
            y_aug.append(yi)
            X_aug.append(np.roll(xi, -shift, axis=0))
            y_aug.append(yi)
    return np.array(X_aug), np.array(y_aug)

X_aug, y_aug = augment_data(X, y, sigma=0.05, timeshift=2)
print(f"Augmented dataset shape: {X_aug.shape}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=np.argmax(y_aug, axis=1)
)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))

# Gated Residual Skip BiLSTM Model
def gated_residual_skip_bilstm(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # First BiLSTM block
    x1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(inputs)
    
    # Second BiLSTM block
    x2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(x1)
    
    # Residual connection
    if x1.shape[-1] != x2.shape[-1]:
        x1_proj = Dense(x2.shape[-1])(x1)
    else:
        x1_proj = x1

    # Gating mechanism
    gate = Dense(x2.shape[-1], activation='sigmoid')(x2)
    gated_residual = Add()([x2, Multiply()([x1_proj, gate])])

    # Final BiLSTM + Dense
    x = Bidirectional(LSTM(128))(gated_residual)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    # Getting 93.33% accuracy at lr = 0.0006
    optimizer = Adam(learning_rate=0.0006)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = gated_residual_skip_bilstm((MAX_LEN, NUM_FEATURES), len(classes))
model.summary()

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_grs_bilstm.keras', save_best_only=True, monitor='val_accuracy')

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=classes)
print("\nClassification Report:\n", report)

# Save Model + Classes
model.save("grs_bilstm_har_final.keras")
history_to_save = {k: [float(x) for x in v] for k,v in history.history.items()}
with open("training_history.json", "w") as f:
    json.dump(history_to_save, f)
print(" Training history saved as 'training_history.json'")
np.save("label_classes.npy", classes)
print("\nModel and label classes saved.")

# Model Size 
model_path = "grs_bilstm_har_final.keras"
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f" Model size: {model_size_mb:.2f} MB")

# Inference Time
# Sample-wise
times = []
for i in range(len(X_test)):
    x_sample = np.expand_dims(X_test[i], axis=0)
    start = time.time()
    _ = model.predict(x_sample, verbose=0)
    end = time.time()
    times.append(end-start)
avg_sample_time = np.mean(times)
print(f" Avg per-sample inference time: {avg_sample_time*1000:.3f} ms")

# Batch-wise
start = time.time()
_ = model.predict(X_test, verbose=0)
end = time.time()
batch_time = end - start
print(f" Batch inference time: {batch_time:.3f} s ({len(X_test)} samples)")

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

# Confusion Matrix Heatmap 
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
