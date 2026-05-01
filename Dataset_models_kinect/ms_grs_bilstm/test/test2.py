# ms_grs_lstm_har.py - OPTIMIZED FOR <1 MB
"""
Multi-scale Gated Residual Skip LSTM - Ultra-compressed version
Target: <1 MB with minimal accuracy loss
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
import joblib

# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 1: Reduce model capacity
# ═══════════════════════════════════════════════════════════════════════════
DATA_ROOT       = "Dataset"
WINDOW_SIZE     = 64
WINDOW_STRIDE   = 16
NUM_FEATURES    = None
BATCH_SIZE      = 64
EPOCHS          = 20
LEARNING_RATE   = 0.009
RANDOM_SEED     = 42

# ── KEY CHANGES ────────────────────────────────────────────────────────────
UNITS_PER_BLOCK = 32       # Reduced from 64 → cuts params by ~75%
FUSION_UNITS    = 64       # Reduced from 128
NUM_SCALES      = 2        # Reduced from 3 (drop stride-4 scale)

OUTPUT_DIR = "Dataset_models_kinect\\ms_grs_bilstm\\test2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE = os.path.join(OUTPUT_DIR, "ms_grs_bilstm_model.keras")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# Custom layers (serializable, no Lambda)
# ═══════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable()
class StridedSlice2(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return x[:, ::2, :]
    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class InverseGate(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, gate):
        return 1.0 - gate
    def get_config(self):
        return super().get_config()


# ═══════════════════════════════════════════════════════════════════════════
# Data utilities (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════

def list_activity_files(data_root):
    rows = []
    for person_dir in sorted(glob.glob(os.path.join(data_root, "Person_*"))):
        person_name = os.path.basename(person_dir)
        for csvf in glob.glob(os.path.join(person_dir, "*.csv")):
            label = os.path.splitext(os.path.basename(csvf))[0].lower()
            rows.append((csvf, label, person_name))
    return rows


def read_csv_numeric(filepath):
    df = pd.read_csv(filepath)
    if df.shape[1] <= 2:
        raise ValueError(f"CSV {filepath} has <=2 columns")
    numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    numeric = numeric.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return numeric.values.astype(np.float32)


def sliding_windows(seq_array, window_size=64, stride=16):
    T, F = seq_array.shape
    if T < window_size:
        pad = np.zeros((window_size - T, F), dtype=seq_array.dtype)
        seq_array = np.vstack([seq_array, pad])
        T = window_size
    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(seq_array[start:start + window_size])
    return np.stack(windows, axis=0)


def add_gaussian_noise(X, noise_level=0.02):
    noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
    return X + noise


def prepare_dataset(data_root, window_size=64, stride=16,
                    test_size=0.2, scale_person_split=False):
    files = list_activity_files(data_root)
    print(f"Found {len(files)} files in dataset.")

    X_windows, y_windows, persons = [], [], []
    for (fp, label, person) in files:
        arr     = read_csv_numeric(fp)
        windows = sliding_windows(arr, window_size=window_size, stride=stride)
        X_windows.append(windows)
        y_windows.extend([label] * windows.shape[0])
        persons.extend([person]  * windows.shape[0])

    X = np.vstack(X_windows)
    print("Raw windows shape:", X.shape)

    le = LabelEncoder()
    y  = le.fit_transform(y_windows)

    nsamples, nt, nf = X.shape
    global NUM_FEATURES
    NUM_FEATURES = nf

    X_flat = X.reshape(-1, nf)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X      = X_flat.reshape(nsamples, nt, nf)
    X      = add_gaussian_noise(X, noise_level=0.05)

    if scale_person_split:
        persons        = np.array(persons)
        unique_persons = np.unique(persons)
        train_p, _     = train_test_split(unique_persons, test_size=test_size,
                                          random_state=RANDOM_SEED)
        mask           = np.isin(persons, train_p)
        X_train, X_test = X[mask], X[~mask]
        y_train, y_test = y[mask], y[~mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)

    y_train_cat = to_categorical(y_train)
    y_test_cat  = to_categorical(y_test)
    print("Train:", X_train.shape, y_train_cat.shape,
          "  Test:", X_test.shape, y_test_cat.shape)
    return X_train, X_test, y_train_cat, y_test_cat, le, scaler


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 2: Lighter architecture (fewer params)
# ═══════════════════════════════════════════════════════════════════════════

def lightweight_gated_block(x_in, units, name_prefix, dropout=0.3):
    """
    Simplified gated block:
    - Single LSTM (not Bidirectional) to halve params
    - Smaller projection layers
    - Higher dropout for regularization
    """
    # Project to lower dimension
    proj = layers.TimeDistributed(
               layers.Dense(units, activation=None),
               name=f"{name_prefix}_proj")(x_in)

    # Single-direction LSTM with unroll=True (no Flex ops)
    lstm_out = layers.LSTM(units, return_sequences=True, unroll=True,
                           name=f"{name_prefix}_lstm")(x_in)
    
    lstm_out = layers.Dropout(dropout)(lstm_out)

    # Gating mechanism
    concat = layers.Concatenate(axis=-1)([proj, lstm_out])
    gate   = layers.TimeDistributed(
                 layers.Dense(units, activation="sigmoid"),
                 name=f"{name_prefix}_gate")(concat)

    gated    = layers.Multiply()([gate, lstm_out])
    inv_gate = InverseGate(name=f"{name_prefix}_inv")(gate)
    residual = layers.Multiply()([inv_gate, proj])
    
    return layers.Add()([gated, residual])


def build_ultra_compact_model(window_size, num_features, num_classes,
                               units=32, fusion_units=64, num_scales=2):
    """
    Ultra-compact multi-scale model:
    - Fewer scales (2 instead of 3)
    - Smaller units (32 instead of 64)
    - Single LSTM instead of Bidirectional
    - Only ONE block per scale (not two)
    """
    inputs = Input(shape=(window_size, num_features), name="input_seq")

    # Only 2 scales: original + stride-2
    s0 = inputs
    s1 = StridedSlice2(name="downsample_2")(inputs)
    
    scales = [s0, s1][:num_scales]
    scale_outputs = []

    for i, s in enumerate(scales):
        prefix = f"scale{i}"
        
        # ── CHANGE: Only ONE block per scale (was 2) ──────────────────────
        block = lightweight_gated_block(s, units, prefix, dropout=0.3)
        
        pooled = layers.GlobalAveragePooling1D(name=f"{prefix}_pool")(block)
        scale_outputs.append(pooled)

    # Fusion with smaller Dense layer
    fused = layers.Concatenate(name="concat")(scale_outputs)
    fused = layers.Dense(fusion_units, activation="relu", 
                         name="fusion")(fused)
    fused = layers.Dropout(0.4)(fused)
    
    out = layers.Dense(num_classes, activation="softmax",
                       name="classifier")(fused)

    return Model(inputs=inputs, outputs=out, name="MS_GRS_LSTM_Compact")


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       label_encoder, scaler, save_path=MODEL_SAVE):
    num_classes = y_train.shape[1]
    
    # Use the compact model
    model = build_ultra_compact_model(
                WINDOW_SIZE, NUM_FEATURES, num_classes,
                units=UNITS_PER_BLOCK,
                fusion_units=FUSION_UNITS,
                num_scales=NUM_SCALES)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE (Optimized for size)")
    print("="*60)
    model.summary()
    
    # Count parameters
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    print(f"\nTotal trainable parameters: {trainable:,}")
    print(f"Estimated model size: ~{trainable * 4 / (1024*1024):.2f} MB (float32)")
    print("="*60 + "\n")

    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True,
            monitor="val_loss", verbose=1),
    ]

    history = model.fit(
        X_train, y_train, validation_split=0.15,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cb, verbose=2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest loss: {test_loss:.4f}  |  Test acc: {test_acc:.4f}")

    start        = time.time()
    y_pred_probs = model.predict(X_test)
    end          = time.time()

    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test,       axis=1)

    print(f"R² score: {r2_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=label_encoder.classes_, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    model.save(save_path)

    # Save artifacts
    history_to_save = {k: [float(x) for x in v]
                       for k, v in history.history.items()}
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history_to_save, f)

    joblib.dump(scaler,        os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    config = {"WINDOW_SIZE": WINDOW_SIZE,
              "NUM_FEATURES": NUM_FEATURES,
              "WINDOW_STRIDE": WINDOW_STRIDE,
              "UNITS": UNITS_PER_BLOCK,
              "NUM_SCALES": NUM_SCALES}
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
        json.dump(config, f)

    batch_time = end - start
    print(f"\nInference: {batch_time:.4f} s for {len(X_test)} samples")
    print(f"Per-sample: {batch_time/len(X_test)*1000:.3f} ms")
    
    size_mb = os.path.getsize(save_path) / (1024*1024)
    print(f"Keras model size: {size_mb:.2f} MB")
    
    return model, history, cm


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION 3: Aggressive quantization pipeline
# ═══════════════════════════════════════════════════════════════════════════

def create_representative_dataset(X_train, num_samples=200):
    """Generator for calibration data (needed for full int8 quantization)."""
    def representative_data_gen():
        for i in range(min(num_samples, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    return representative_data_gen


def quantize_ultra_compressed(model_path, X_train, output_dir):
    """
    Aggressive quantization pipeline to reach <1 MB:
    1. Dynamic-range (int8 weights, float activations)
    2. Full integer (int8 weights + activations) ← most aggressive
    3. Float16 (for comparison)
    """
    print("\n" + "="*60)
    print("ULTRA-COMPRESSION PIPELINE")
    print("="*60)

    custom_objects = {
        "StridedSlice2": StridedSlice2,
        "InverseGate":   InverseGate,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Loaded: {model_path}\n")

    results = {}
    orig_size = os.path.getsize(model_path) / (1024*1024)

    # ── 1. Dynamic-range quantization ─────────────────────────────────────
    print("[1/3] Dynamic-range quantization (int8 weights) …")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tfl = conv.convert()
    p   = os.path.join(output_dir, "model_dynamic_int8.tflite")
    open(p, "wb").write(tfl)
    results["dynamic_int8"] = p
    print(f"    Size: {len(tfl)/(1024*1024):.2f} MB")

    # ── 2. Full integer quantization (int8 weights + activations) ─────────
    print("[2/3] Full integer quantization (int8 weights + activations) …")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Provide representative dataset for activation quantization
    conv.representative_dataset = create_representative_dataset(X_train, 200)
    
    # Force int8 input/output (optional, comment out for float I/O)
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    
    tfl = conv.convert()
    p   = os.path.join(output_dir, "model_full_int8.tflite")
    open(p, "wb").write(tfl)
    results["full_int8"] = p
    print(f"    Size: {len(tfl)/(1024*1024):.2f} MB")

    # ── 3. Float16 (for comparison) ───────────────────────────────────────
    print("[3/3] Float-16 quantization …")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tfl = conv.convert()
    p   = os.path.join(output_dir, "model_float16.tflite")
    open(p, "wb").write(tfl)
    results["float16"] = p
    print(f"    Size: {len(tfl)/(1024*1024):.2f} MB")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"  Original Keras model  : {orig_size:.2f} MB")
    print("-"*60)
    
    for name, path in results.items():
        size_mb = os.path.getsize(path) / (1024*1024)
        ratio   = orig_size / size_mb
        print(f"  {name:20s} : {size_mb:.2f} MB  ({ratio:.1f}× smaller)")
    print("="*60)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TFLite evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_tflite_model(tflite_path, X_test, y_test, label_encoder,
                          model_name="TFLite", is_int8_io=False):
    """
    Evaluate TFLite model.
    is_int8_io: True if model has int8 input/output (need to quantize inputs)
    """
    print(f"\n{'─'*60}")
    print(f"Evaluating: {model_name}")
    print(f"Path      : {tflite_path}")
    size_mb = os.path.getsize(tflite_path) / (1024*1024)
    print(f"Size      : {size_mb:.2f} MB")

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()

    # Check if model expects int8 input
    input_dtype = inp_det[0]['dtype']
    print(f"Input type: {input_dtype}")

    predictions = []
    t0 = time.time()
    
    for i in range(len(X_test)):
        inp = np.expand_dims(X_test[i], 0)
        
        # Quantize input if model expects int8
        if input_dtype == np.int8:
            scale, zero_point = inp_det[0]['quantization']
            inp = (inp / scale + zero_point).astype(np.int8)
        else:
            inp = inp.astype(np.float32)
        
        interp.set_tensor(inp_det[0]['index'], inp)
        interp.invoke()
        output = interp.get_tensor(out_det[0]['index'])
        
        # Dequantize output if needed
        if out_det[0]['dtype'] == np.int8:
            scale, zero_point = out_det[0]['quantization']
            output = (output.astype(np.float32) - zero_point) * scale
        
        predictions.append(output[0])
    
    elapsed = time.time() - t0

    predictions = np.array(predictions)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc    = np.mean(y_pred == y_true)

    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Inference : {elapsed:.4f} s  |  {elapsed/len(X_test)*1000:.3f} ms/sample")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=label_encoder.classes_, digits=4))
    
    return acc, y_pred


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'],     label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'],     label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title("Accuracy")
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Data
    X_train, X_test, y_train, y_test, le, scaler = prepare_dataset(
        DATA_ROOT, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE,
        test_size=0.2, scale_person_split=False)

    # 2. Train compact model
    model, history, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test, le, scaler, save_path=MODEL_SAVE)

    # 3. Ultra-compress
    tflite_models = quantize_ultra_compressed(MODEL_SAVE, X_train, OUTPUT_DIR)

    # 4. Evaluate all compressed models
    print("\n" + "="*60)
    print("EVALUATING COMPRESSED MODELS")
    print("="*60)
    
    evaluate_tflite_model(tflite_models["dynamic_int8"], X_test, y_test, le,
                          "Dynamic Int8")
    
    evaluate_tflite_model(tflite_models["full_int8"], X_test, y_test, le,
                          "Full Int8 (most compressed)", is_int8_io=True)
    
    evaluate_tflite_model(tflite_models["float16"], X_test, y_test, le,
                          "Float16")

    # 5. Plots
    plt.figure(figsize=(10, 6))
    ep = range(1, len(history.history['accuracy']) + 1)
    plt.plot(ep, history.history['accuracy'],     'b-',  label='Train Acc')
    plt.plot(ep, history.history['val_accuracy'], 'r-',  label='Val Acc')
    plt.plot(ep, history.history['loss'],         'b--', label='Train Loss')
    plt.plot(ep, history.history['val_loss'],     'r--', label='Val Loss')
    plt.xlabel("Epochs"); plt.ylabel("Value")
    plt.title("Training History (Compact Model)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()

    plot_history(history)
    print(f"\n✓ All models saved to {OUTPUT_DIR}")