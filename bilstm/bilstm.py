# bilstm_har.py
"""
Human Activity Recognition using BiLSTM
Dataset folder structure:
Dataset/
 ┣ Person_1/
 ┃ ┣ walking.csv
 ┃ ┣ running.csv
 ┃ ┗ ...
 ┣ Person_2/
 ┃ ┣ walking.csv
 ┃ ┣ running.csv
 ┃ ┗ ...
 ...
Each CSV has columns:
timestamp, body_id, then 3D joint coordinates (x, y, z) for 25 joints
Total numeric features per frame = 75
"""



import time, json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataset

def load_dataset(path):
    sequences, labels = [], []
    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        if not os.path.isdir(person_path): continue
        for file in os.listdir(person_path):
            if file.endswith('.csv'):
                label = file.replace('.csv','')
                df = pd.read_csv(os.path.join(person_path, file))
                sequences.append(df.iloc[:,2:].values)
                labels.append(label)
    return sequences, labels

dataset_path = "Dataset"
sequences, labels = load_dataset(dataset_path)


# Padding & Encoding

MAX_LEN = 128
NUM_FEATURES = sequences[0].shape[1]
X = pad_sequences(sequences, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)
classes = encoder.classes_


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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42, stratify=np.argmax(y_aug, axis=1)
)

# compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights = dict(enumerate(class_weights))

# BiLSTM Model
def create_bilstm(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2), input_shape=input_shape),
        LayerNormalization(),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_bilstm((MAX_LEN, NUM_FEATURES), len(classes))
model.summary()

# Training
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_bilstm_aug.keras', save_best_only=True, monitor='val_accuracy')

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# Save the final trained model
model.save("bilstm_har_final.keras")  # .keras format
print("Model saved as 'bilstm_har_final.keras'")

# save the label encoder classes
np.save("label_classes.npy", classes)
print("Label classes saved as 'label_classes.npy'")


# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=classes, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Inference Time Measurement
print("\n Measuring inference time...")

# Warm-up
_ = model.predict(X_test[:5])

# Sample-wise inference time
times = []
for i in range(len(X_test)):
    x_sample = np.expand_dims(X_test[i], axis=0)
    start = time.time()
    _ = model.predict(x_sample, verbose=0)
    end = time.time()
    times.append(end - start)

avg_time = np.mean(times)
std_time = np.std(times)
print(f" Average inference time per sample: {avg_time*1000:.2f} ms")
print(f" Std deviation: {std_time*1000:.2f} ms")

# Batch inference time
start = time.time()
_ = model.predict(X_test, verbose=0)
end = time.time()
batch_time = end - start
print(f"Total batch inference time: {batch_time:.2f} s")
print(f"Average batch inference per sample: {(batch_time/len(X_test))*1000:.2f} ms")

#  Model Siz
model_path = "bilstm_har_final.keras"
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\n Model Size: {model_size:.2f} MB")

#  Save Training History (JSON) 
with open("training_history.json", "w") as f:
    json.dump(history.history, f)
print("Training history saved as 'training_history.json'")

#  Plot Accuracy & Loss Curves 
plt.figure(figsize=(10, 6))
epochs = range(1, len(history.history['accuracy']) + 1)

# Accuracy (solid)
plt.plot(epochs, history.history['accuracy'], 'b-', label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Val Accuracy')

# Loss (dashed)
plt.plot(epochs, history.history['loss'], 'b--', label='Train Loss')
plt.plot(epochs, history.history['val_loss'], 'r--', label='Val Loss')

plt.title("Training & Validation Accuracy/Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



