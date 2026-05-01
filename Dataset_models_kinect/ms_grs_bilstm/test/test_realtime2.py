# realtime_har_kinect_v2_quantized.py
"""
Real-Time HAR using Quantized MS-GRS-LSTM + Kinect v2
Supports TFLite quantized models (dynamic_int8, full_int8, float16)
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
import collections
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import joblib

import tensorflow as tf

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR    = r"Dataset_models_kinect\ms_grs_bilstm\test2"

# ── SELECT WHICH QUANTIZED MODEL TO USE ───────────────────────────────────────
# Options: "dynamic_int8" (recommended), "full_int8" (smallest), "float16"
QUANTIZED_MODEL_TYPE = "dynamic_int8"

MODEL_PATHS = {
    "dynamic_int8": os.path.join(MODEL_DIR, "model_dynamic_int8.tflite"),
    "full_int8":    os.path.join(MODEL_DIR, "model_full_int8.tflite"),
    "float16":      os.path.join(MODEL_DIR, "model_float16.tflite"),
}

MODEL_PATH   = MODEL_PATHS[QUANTIZED_MODEL_TYPE]
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
LE_PATH      = os.path.join(MODEL_DIR, "label_encoder.pkl")
CONFIG_PATH  = os.path.join(MODEL_DIR, "model_config.json")

DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# Optimized parameters
POSITION_SMOOTHING = 3
PREDICTION_SMOOTHING = 20
MIN_ACTIVITY_DURATION = 12

# Motion thresholds
STANDING_VELOCITY_THRESHOLD = 0.015
STANDING_VARIANCE_THRESHOLD = 0.002
LEG_VELOCITY_THRESHOLD = 0.025

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'standing': 0.30,
    'sitting': 0.35,
    'walking': 0.55,
    'running': 0.60,
    'jumping': 0.55,
    'bending': 0.45,
    'default': 0.45
}

# Quality thresholds
MIN_TRACKED_JOINTS_RATIO = 0.6

NUM_JOINTS = 25
CALIBRATION_FRAMES = 60

BONES = [
    (0,1),(1,20),(20,2),(2,3),
    (20,4),(4,5),(5,6),(6,7),(7,21),(7,22),
    (20,8),(8,9),(9,10),(10,11),(11,23),(11,24),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19)
]

JOINT_SPINE_BASE = 0
JOINT_LEFT_HIP = 12
JOINT_LEFT_KNEE = 13
JOINT_LEFT_ANKLE = 14
JOINT_RIGHT_HIP = 16
JOINT_RIGHT_KNEE = 17
JOINT_RIGHT_ANKLE = 18


# ─────────────────────────────────────────────
# Custom Keras Layers (for reference, not used in TFLite)
# ─────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class StridedSlice2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return x[:, ::2, :]
    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class InverseGate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, gate):
        return 1.0 - gate
    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────
# PATCH numpy.object in PyKinectRuntime
# ─────────────────────────────────────────────
def patch_numpy_in_pykinect():
    """Patches numpy.object → object in PyKinectRuntime.py"""
    try:
        import pykinect2
        pkg = os.path.dirname(pykinect2.__file__)
        rt_file = os.path.join(pkg, "PyKinectRuntime.py")

        with open(rt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if 'numpy.object)' not in content:
            return True

        import shutil
        backup = rt_file + ".numpy_bak"
        if not os.path.exists(backup):
            shutil.copy2(rt_file, backup)

        fixes = [
            ('numpy.object)', 'object)'),
            ('numpy.object,', 'object,'),
            ('numpy.bool)', 'bool)'),
            ('numpy.bool,', 'bool,'),
            ('numpy.int)', 'int)'),
            ('numpy.int,', 'int,'),
            ('numpy.float)', 'float)'),
            ('numpy.float,', 'float,'),
        ]
        count = 0
        for old, new in fixes:
            if old in content:
                count += content.count(old)
                content = content.replace(old, new)

        with open(rt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"  [FIX] numpy patch: {count} replacements")
        return True

    except Exception as e:
        print(f"  [WARN] numpy patch failed: {e}")
        return False


# ─────────────────────────────────────────────
# Simple Position Smoother
# ─────────────────────────────────────────────
class SimplePositionSmoother:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = collections.deque(maxlen=window_size)
    
    def smooth(self, joints_xyz):
        self.history.append(joints_xyz.copy())
        if len(self.history) < 2:
            return joints_xyz
        smoothed = np.mean(self.history, axis=0)
        return smoothed.astype(np.float32)
    
    def reset(self):
        self.history.clear()


# ─────────────────────────────────────────────
# Motion Analysis
# ─────────────────────────────────────────────
class MotionAnalyzer:
    def __init__(self):
        self.prev_positions = None
    
    def compute_velocity(self, current_positions, prev_positions=None):
        if prev_positions is None:
            prev_positions = self.prev_positions
        if prev_positions is None:
            return 0.0
        diff = current_positions - prev_positions
        velocities = np.linalg.norm(diff, axis=1)
        return np.mean(velocities)
    
    def compute_variance(self, window_positions):
        variance = np.var(window_positions, axis=0)
        return np.mean(variance)
    
    def compute_leg_velocity(self, current_positions, prev_positions=None):
        if prev_positions is None:
            prev_positions = self.prev_positions
        if prev_positions is None:
            return 0.0
        leg_joints = [JOINT_LEFT_KNEE, JOINT_LEFT_ANKLE, 
                     JOINT_RIGHT_KNEE, JOINT_RIGHT_ANKLE]
        diff = current_positions[leg_joints] - prev_positions[leg_joints]
        velocities = np.linalg.norm(diff, axis=1)
        return np.mean(velocities)
    
    def is_standing_motion(self, current_positions, window_positions=None):
        velocity = self.compute_velocity(current_positions)
        reasons = []
        standing_score = 0.0
        
        if velocity < STANDING_VELOCITY_THRESHOLD:
            standing_score += 0.4
            reasons.append(f"low_vel:{velocity:.4f}")
        
        leg_vel = self.compute_leg_velocity(current_positions)
        if leg_vel < LEG_VELOCITY_THRESHOLD:
            standing_score += 0.3
            reasons.append(f"low_leg_vel:{leg_vel:.4f}")
        
        if window_positions is not None and len(window_positions) > 10:
            variance = self.compute_variance(window_positions)
            if variance < STANDING_VARIANCE_THRESHOLD:
                standing_score += 0.3
                reasons.append(f"low_var:{variance:.4f}")
        
        is_standing = standing_score >= 0.6
        self.prev_positions = current_positions.copy()
        
        return is_standing, standing_score, " | ".join(reasons)
    
    def reset(self):
        self.prev_positions = None


# ─────────────────────────────────────────────
# Calibration Manager
# ─────────────────────────────────────────────
class CalibrationManager:
    def __init__(self, num_frames=CALIBRATION_FRAMES):
        self.num_frames = num_frames
        self.frames = []
        self.baseline_position = None
        self.calibrated = False
    
    def add_frame(self, joints_xyz, tracking_state):
        if self.calibrated:
            return True
        tracked_ratio = np.mean(tracking_state == 2)
        if tracked_ratio < 0.7:
            return False
        self.frames.append(joints_xyz.copy())
        if len(self.frames) >= self.num_frames:
            self.baseline_position = np.median(self.frames, axis=0)
            self.calibrated = True
            print(f"[CALIBRATION] Complete with {len(self.frames)} frames")
            return True
        return False
    
    def reset(self):
        self.frames = []
        self.baseline_position = None
        self.calibrated = False


# ─────────────────────────────────────────────
# Light Quality Filter
# ─────────────────────────────────────────────
class LightQualityFilter:
    @staticmethod
    def is_valid_frame(joints_xyz, tracking_state):
        tracked_ratio = np.mean(tracking_state == 2)
        inferred_ratio = np.mean(tracking_state == 1)
        total_tracked = tracked_ratio + inferred_ratio
        
        if total_tracked < MIN_TRACKED_JOINTS_RATIO:
            return False, f"poor_tracking:{total_tracked:.2f}"
        
        if np.any(~np.isfinite(joints_xyz)):
            return False, "nan_or_inf"
        
        z_coords = joints_xyz[:, 2]
        if np.any(z_coords < 0.1) or np.any(z_coords > 10.0):
            return False, f"z_range:{z_coords.min():.2f}-{z_coords.max():.2f}"
        
        if np.any(np.abs(joints_xyz[:, 0]) > 5.0) or np.any(np.abs(joints_xyz[:, 1]) > 5.0):
            return False, "xy_out_of_range"
        
        return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════
# TFLite Model Loading & Inference
# ═══════════════════════════════════════════════════════════════════════════

def load_tflite_model(model_path):
    """
    Load TFLite model and return interpreter with input/output details
    """
    print(f"  Loading TFLite model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  [ERROR] Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  [OK] TFLite model loaded")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output dtype: {output_details[0]['dtype']}")
        
        # Check if quantized
        if input_details[0]['dtype'] == np.int8:
            print(f"  [INFO] INT8 quantized model detected")
            input_scale, input_zero_point = input_details[0]['quantization']
            output_scale, output_zero_point = output_details[0]['quantization']
            print(f"  Input quantization: scale={input_scale}, zero_point={input_zero_point}")
            print(f"  Output quantization: scale={output_scale}, zero_point={output_zero_point}")
        else:
            print(f"  [INFO] Float model (dynamic quantization)")
        
        return interpreter, input_details, output_details
    
    except Exception as e:
        print(f"  [ERROR] Failed to load TFLite model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_artifacts():
    """Load model, scaler, label encoder, and config"""
    print("=" * 60)
    print("  LOADING MODEL ARTIFACTS (TFLite)")
    print("=" * 60)
    
    # Check files exist
    for p, n in [(MODEL_PATH, "TFLite Model"),
                 (SCALER_PATH, "Scaler"),
                 (LE_PATH, "Labels"),
                 (CONFIG_PATH, "Config")]:
        if not os.path.exists(p):
            print(f"  [ERROR] {n} missing: {p}")
            sys.exit(1)
    
    # Load config
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    
    ws = cfg["WINDOW_SIZE"]
    nf = cfg.get("NUM_FEATURES")
    wst = cfg.get("WINDOW_STRIDE", 16)
    
    # Handle new config structure
    if nf is None:
        units = cfg.get("UNITS", 32)
        num_scales = cfg.get("NUM_SCALES", 2)
        nf = NUM_JOINTS * 3  # 25 joints * 3 coordinates = 75
        print(f"  [INFO] NUM_FEATURES not in config, using {nf} (NUM_JOINTS*3)")
    
    print(f"  Window={ws} Features={nf} Stride={wst}")
    
    # Load label encoder
    le = joblib.load(LE_PATH)
    nc = len(le.classes_)
    print(f"  Labels({nc}): {list(le.classes_)}")
    
    # Load TFLite model
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    print(f"  [OK] Scaler loaded")
    
    # Verify with dummy input
    print(f"  Verifying model with dummy input...")
    dummy = np.zeros((1, ws, nf), dtype=np.float32)
    
    # Handle quantization
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        dummy_quantized = (dummy / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], dummy_quantized)
    else:
        interpreter.set_tensor(input_details[0]['index'], dummy)
    
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output if needed
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    print(f"  Verify OK - Output shape: {output.shape}, Sum: {output.sum():.4f}")
    print(f"  Model type: {QUANTIZED_MODEL_TYPE}")
    
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    print("=" * 60)
    
    return interpreter, input_details, output_details, scaler, le, ws, nf, wst


# ═══════════════════════════════════════════════════════════════════════════
# KINECT V2 SENSOR (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class KinectV2Sensor:
    def __init__(self):
        print("\n[KINECT] Initialising Kinect v2...")
        print("  Patching numpy deprecations...")
        patch_numpy_in_pykinect()

        try:
            from pykinect2 import PyKinectV2 as _PK
            self._PK = _PK
            print("  [OK] PyKinectV2 imported")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            sys.exit(1)

        try:
            from pykinect2 import PyKinectRuntime as _PKR
            self._PKR = _PKR
            print("  [OK] PyKinectRuntime imported")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            sys.exit(1)

        import importlib
        import pykinect2.PyKinectRuntime as _rt_mod
        importlib.reload(_rt_mod)
        self._PKR = _rt_mod
        print("  [OK] PyKinectRuntime reloaded (numpy fix active)")

        self._runtime = None
        self._bodies = None
        self._color_enabled = True
        self._frame_count = 0

        print("  Connecting to sensor...")
        try:
            self._runtime = self._PKR.PyKinectRuntime(
                self._PK.FrameSourceTypes_Color |
                self._PK.FrameSourceTypes_Body)
            print(f"  [OK] Color + Body active")
        except Exception as e1:
            print(f"  [WARN] Color+Body: {e1}")
            try:
                self._runtime = self._PKR.PyKinectRuntime(
                    self._PK.FrameSourceTypes_Body)
                self._color_enabled = False
                print("  [OK] Body-only mode")
            except Exception as e2:
                print(f"  [ERROR] Cannot connect: {e2}")
                sys.exit(1)

        print("  Warming up (3 seconds)...")
        time.sleep(3)
        print("[KINECT] Ready.\n")

    def update(self):
        color_frame = None
        bodies_out = []
        self._frame_count += 1

        if self._color_enabled:
            try:
                if self._runtime.has_new_color_frame():
                    f = self._runtime.get_last_color_frame()
                    color_frame = f.reshape((1080, 1920, 4)).astype(np.uint8)
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

        try:
            if self._runtime.has_new_body_frame():
                self._bodies = self._runtime.get_last_body_frame()
        except Exception:
            pass

        if self._bodies is not None:
            try:
                mc = self._runtime.max_body_count
                for i in range(mc):
                    try:
                        body = self._bodies.bodies[i]
                        try:
                            is_tracked = bool(body.is_tracked)
                        except Exception:
                            is_tracked = False
                        if not is_tracked:
                            continue

                        joints = body.joints
                        xyz = np.zeros((NUM_JOINTS, 3), np.float32)
                        pix = np.zeros((NUM_JOINTS, 2), np.int32)
                        trk = np.zeros(NUM_JOINTS, np.int32)

                        for j in range(NUM_JOINTS):
                            try:
                                pos = joints[j].Position
                                xyz[j, 0] = float(pos.x)
                                xyz[j, 1] = float(pos.y)
                                xyz[j, 2] = float(pos.z)
                                trk[j] = int(joints[j].TrackingState)
                            except Exception:
                                pass

                        try:
                            cp = self._runtime.body_joints_to_color_space(joints)
                            for j in range(NUM_JOINTS):
                                try:
                                    xv = float(cp[j].x)
                                    yv = float(cp[j].y)
                                    if (np.isfinite(xv) and np.isfinite(yv) and
                                            0 <= xv <= 1920 and 0 <= yv <= 1080):
                                        pix[j] = [int(xv), int(yv)]
                                    elif xyz[j, 2] > 0:
                                        pix[j, 0] = int(xyz[j,0]/xyz[j,2] * 525 + 960)
                                        pix[j, 1] = int(-xyz[j,1]/xyz[j,2] * 525 + 540)
                                except Exception:
                                    pass
                        except Exception:
                            for j in range(NUM_JOINTS):
                                if xyz[j, 2] > 0:
                                    pix[j, 0] = int(xyz[j,0]/xyz[j,2] * 525 + 960)
                                    pix[j, 1] = int(-xyz[j,1]/xyz[j,2] * 525 + 540)

                        bodies_out.append({
                            'joints_xyz': xyz,
                            'joints_pixel': pix,
                            'tracking_state': trk
                        })
                    except Exception:
                        continue
            except Exception:
                pass

        return color_frame, bodies_out

    def close(self):
        try:
            self._runtime.close()
        except Exception:
            pass
        print("[KINECT] Closed.")


class MockSensor:
    def __init__(self):
        self._t0 = time.time()
        self._acts = ["standing", "walking", "bending", "jumping", "sitting"]
        print("[MOCK] Ready\n")

    def _pose(self):
        xyz = np.zeros((NUM_JOINTS, 3), np.float32)
        y = -0.3
        xyz[0] = [0, y, 2.5]
        xyz[1] = [0, y + .25, 2.5]
        xyz[20] = [0, y + .45, 2.5]
        xyz[2] = [0, y + .52, 2.5]
        xyz[3] = [0, y + .62, 2.5]
        xyz[4] = [-.18, y + .42, 2.5]
        xyz[5] = [-.25, y + .22, 2.5]
        xyz[6] = [-.22, y + .02, 2.5]
        xyz[7] = [-.21, y - .03, 2.5]
        xyz[21] = [-.20, y - .06, 2.5]
        xyz[22] = [-.23, y - .03, 2.5]
        xyz[8] = [.18, y + .42, 2.5]
        xyz[9] = [.25, y + .22, 2.5]
        xyz[10] = [.22, y + .02, 2.5]
        xyz[11] = [.21, y - .03, 2.5]
        xyz[23] = [.20, y - .06, 2.5]
        xyz[24] = [.23, y - .03, 2.5]
        xyz[12] = [-.10, y - .02, 2.5]
        xyz[13] = [-.12, y - .32, 2.5]
        xyz[14] = [-.12, y - .62, 2.5]
        xyz[15] = [-.15, y - .65, 2.45]
        xyz[16] = [.10, y - .02, 2.5]
        xyz[17] = [.12, y - .32, 2.5]
        xyz[18] = [.12, y - .62, 2.5]
        xyz[19] = [.15, y - .65, 2.45]
        return xyz

    def update(self):
        t = time.time() - self._t0
        ai = int(t / 5.0) % len(self._acts)
        act = self._acts[ai]
        xyz = self._pose()

        if act == "walking":
            s = .08 * np.sin(t * 4)
            xyz[14, 0] += s
            xyz[18, 0] -= s
        elif act == "bending":
            b = .3 * (np.sin(t * 1.5) * .5 + .5)
            for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24]:
                xyz[j, 2] -= b * .3
                xyz[j, 1] -= b * .4
        elif act == "jumping":
            xyz[:, 1] += max(0, np.sin(t * 3)) * .2
        elif act == "sitting":
            xyz[0, 1] -= .25
            xyz[12, 1] -= .2
            xyz[16, 1] -= .2

        if act == "standing":
            xyz += np.random.normal(0, 0.005, xyz.shape)
        else:
            xyz[:, 0] += .01 * np.sin(t * 1.5)

        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
        cv2.putText(frame, f"MOCK: {act.upper()}",
                    (10, DISPLAY_HEIGHT - 40), FONT, 0.55,
                    (0, 200, 255), 1, cv2.LINE_AA)

        pix = np.zeros((NUM_JOINTS, 2), np.int32)
        for j in range(NUM_JOINTS):
            if xyz[j, 2] > 0:
                pix[j, 0] = int(xyz[j, 0] / xyz[j, 2] * 600 + DISPLAY_WIDTH // 2)
                pix[j, 1] = int(-xyz[j, 1] / xyz[j, 2] * 600 + DISPLAY_HEIGHT // 2 + 50)

        return frame, [{'joints_xyz': xyz,
                        'joints_pixel': pix,
                        'tracking_state': np.full(NUM_JOINTS, 2, np.int32)}]

    def close(self):
        print("[MOCK] Closed.")


# ═══════════════════════════════════════════════════════════════════════════
# Optimized Predictor with TFLite
# ═══════════════════════════════════════════════════════════════════════════

class OptimizedTFLitePredictor:
    """
    Optimized predictor using TFLite quantized model
    """

    def __init__(self, interpreter, input_details, output_details,
                 scaler, le, ws, nf, stride):
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.scaler = scaler
        self.le = le
        self.ws = ws
        self.nf = nf
        self.stride = stride

        # Check if model is int8 quantized
        self.is_int8_input = (input_details[0]['dtype'] == np.int8)
        self.is_int8_output = (output_details[0]['dtype'] == np.int8)

        if self.is_int8_input:
            self.input_scale, self.input_zero_point = input_details[0]['quantization']
            print(f"[PREDICTOR] INT8 input: scale={self.input_scale}, zero_point={self.input_zero_point}")

        if self.is_int8_output:
            self.output_scale, self.output_zero_point = output_details[0]['quantization']
            print(f"[PREDICTOR] INT8 output: scale={self.output_scale}, zero_point={self.output_zero_point}")

        # Buffers
        self.position_buffer = collections.deque(maxlen=ws)
        self.prediction_history = collections.deque(maxlen=PREDICTION_SMOOTHING)
        self.prob_history = collections.deque(maxlen=PREDICTION_SMOOTHING)

        # Components
        self.smoother = SimplePositionSmoother(window_size=POSITION_SMOOTHING)
        self.motion_analyzer = MotionAnalyzer()
        self.calibration = CalibrationManager()
        self.quality_filter = LightQualityFilter()

        # State
        self.label = "Initializing..."
        self.conf = 0.0
        self.probs = {}
        self.inf_ms = 0.0
        self.motion_info = ""

        # Activity tracking
        self.current_activity = None
        self.activity_frame_count = 0

        # Stats
        self.frame_count = 0
        self.skipped_frames = 0
        self.last_quality_reason = ""

        # Threading
        self._q = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._inference_thread, daemon=True)
        self._thread.start()

        print(f"[PREDICTOR] Initialized with TFLite model")
        print(f"  Window={ws}, Features={nf}, Stride={stride}")
        print(f"  Model type: {QUANTIZED_MODEL_TYPE}")

    def push(self, joints_xyz, tracking_state):
        """Push new skeleton frame"""
        self.frame_count += 1

        # Calibration phase
        if not self.calibration.calibrated:
            if self.calibration.add_frame(joints_xyz, tracking_state):
                self.label = "Calibration complete! Ready."
            else:
                progress = len(self.calibration.frames)
                self.label = f"Calibrating... {progress}/{CALIBRATION_FRAMES}"
            return

        # Quality check
        is_valid, reason = self.quality_filter.is_valid_frame(joints_xyz, tracking_state)
        if not is_valid:
            self.skipped_frames += 1
            self.last_quality_reason = reason
            return

        # Apply smoothing
        smoothed_joints = self.smoother.smooth(joints_xyz)

        # Add to buffer
        self.position_buffer.append(smoothed_joints.copy())

        # Check if ready to predict
        if len(self.position_buffer) == self.ws and self.frame_count % self.stride == 0:
            window_array = np.array(self.position_buffer)

            # Motion analysis
            is_standing_motion, standing_score, motion_reason = \
                self.motion_analyzer.is_standing_motion(smoothed_joints, window_array)

            self.motion_info = motion_reason

            # Queue for inference
            try:
                self._q.put_nowait({
                    'window': window_array.copy(),
                    'is_standing_motion': is_standing_motion,
                    'standing_score': standing_score
                })
            except queue.Full:
                pass

    def _inference_thread(self):
        """Background TFLite inference thread"""
        while True:
            data = self._q.get()

            try:
                t0 = time.time()

                window = data['window']
                is_standing_motion = data['is_standing_motion']
                standing_score = data['standing_score']

                # Extract features (same as training)
                features_window = window.reshape(self.ws, -1)

                # Ensure correct feature dimension
                if features_window.shape[1] < self.nf:
                    pad_width = ((0, 0), (0, self.nf - features_window.shape[1]))
                    features_window = np.pad(features_window, pad_width, mode='constant')
                elif features_window.shape[1] > self.nf:
                    features_window = features_window[:, :self.nf]

                # Apply scaler
                scaled = self.scaler.transform(features_window)

                # Prepare model input
                model_input = scaled.reshape(1, self.ws, self.nf).astype(np.float32)

                # ── TFLite inference with quantization handling ──────────────
                if self.is_int8_input:
                    # Quantize input: float32 -> int8
                    model_input_quantized = (model_input / self.input_scale +
                                            self.input_zero_point).astype(np.int8)
                    self.interpreter.set_tensor(self.input_details[0]['index'],
                                               model_input_quantized)
                else:
                    self.interpreter.set_tensor(self.input_details[0]['index'],
                                               model_input)

                # Run inference
                self.interpreter.invoke()

                # Get output
                output = self.interpreter.get_tensor(self.output_details[0]['index'])

                # Dequantize output if needed: int8 -> float32
                if self.is_int8_output:
                    probs = ((output.astype(np.float32) - self.output_zero_point) *
                            self.output_scale)[0]
                else:
                    probs = output[0]

                # Ensure probabilities are valid (softmax might have small errors)
                probs = np.clip(probs, 0, 1)
                probs = probs / probs.sum()  # Renormalize

                # ── Rest of prediction logic (same as before) ────────────────
                self.prob_history.append(probs)

                if len(self.prob_history) >= 3:
                    smoothed_probs = np.mean(self.prob_history, axis=0)
                else:
                    smoothed_probs = probs

                model_pred_idx = int(np.argmax(smoothed_probs))
                model_pred_label = str(self.le.classes_[model_pred_idx])
                model_confidence = float(smoothed_probs[model_pred_idx])

                # Motion-based override
                final_pred_label = model_pred_label
                final_confidence = model_confidence
                override_applied = False

                if is_standing_motion and model_pred_label.lower() in ['walking', 'running']:
                    standing_idx = None
                    for i, cls in enumerate(self.le.classes_):
                        if cls.lower() == 'standing':
                            standing_idx = i
                            break

                    if standing_idx is not None:
                        standing_prob = float(smoothed_probs[standing_idx])
                        prob_diff = model_confidence - standing_prob

                        if prob_diff < 0.25 or standing_score > 0.7:
                            final_pred_label = 'standing'
                            final_confidence = standing_prob
                            override_applied = True

                # Add to prediction history
                final_pred_idx = list(self.le.classes_).index(final_pred_label)
                self.prediction_history.append(final_pred_idx)

                # Majority vote
                if len(self.prediction_history) >= 5:
                    weights = np.linspace(0.5, 1.0, len(self.prediction_history))
                    vote_counts = collections.defaultdict(float)

                    for i, pred_idx in enumerate(self.prediction_history):
                        vote_counts[pred_idx] += weights[i]

                    voted_idx = max(vote_counts.items(), key=lambda x: x[1])[0]
                    voted_label = str(self.le.classes_[voted_idx])
                    voted_conf = float(smoothed_probs[voted_idx])
                else:
                    voted_idx = final_pred_idx
                    voted_label = final_pred_label
                    voted_conf = final_confidence

                # Activity duration check
                if self.current_activity != voted_label:
                    if self.activity_frame_count < MIN_ACTIVITY_DURATION:
                        if self.current_activity is not None:
                            voted_label = self.current_activity
                            voted_idx = list(self.le.classes_).index(self.current_activity)
                            voted_conf = float(smoothed_probs[voted_idx])
                        self.activity_frame_count += 1
                    else:
                        self.current_activity = voted_label
                        self.activity_frame_count = 0
                else:
                    self.activity_frame_count += 1

                # Confidence threshold
                threshold = CONFIDENCE_THRESHOLDS.get(
                    voted_label.lower(),
                    CONFIDENCE_THRESHOLDS['default']
                )

                if voted_conf >= threshold:
                    display_label = voted_label
                else:
                    display_label = f"Uncertain ({voted_label}?)"

                # Update state
                self.label = display_label
                self.conf = voted_conf
                self.probs = {
                    str(self.le.classes_[i]): float(smoothed_probs[i])
                    for i in range(len(smoothed_probs))
                }
                self.inf_ms = (time.time() - t0) * 1000

                if override_applied:
                    self.label += " [motion-override]"

            except Exception as e:
                print(f"[ERROR] Inference: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._q.task_done()

    def get(self):
        return self.label, self.conf, self.probs, self.inf_ms

    def get_stats(self):
        return {
            'buffer_fill': len(self.position_buffer) / self.ws * 100,
            'calibrated': self.calibration.calibrated,
            'total_frames': self.frame_count,
            'skipped_frames': self.skipped_frames,
            'skip_rate': self.skipped_frames / max(self.frame_count, 1) * 100,
            'motion_info': self.motion_info,
            'last_quality_reason': self.last_quality_reason
        }

    def reset(self):
        self.position_buffer.clear()
        self.prediction_history.clear()
        self.prob_history.clear()
        self.smoother.reset()
        self.motion_analyzer.reset()
        self.label = "Reset - Collecting..."
        self.conf = 0.0
        self.probs = {}
        self.current_activity = None
        self.activity_frame_count = 0
        self.frame_count = 0
        self.skipped_frames = 0
        print("[RESET] Predictor state cleared")

    def recalibrate(self):
        self.calibration.reset()
        self.reset()
        print("[RECALIBRATE] Starting calibration phase...")


# ─────────────────────────────────────────────
# Drawing Functions (unchanged)
# ─────────────────────────────────────────────

def draw_skel(fr, jp, ts=None, sx=1.0, sy=1.0):
    H, W = fr.shape[:2]
    for a, b in BONES:
        if a >= len(jp) or b >= len(jp):
            continue
        pa = (int(np.clip(jp[a, 0] * sx, 0, W - 1)),
              int(np.clip(jp[a, 1] * sy, 0, H - 1)))
        pb = (int(np.clip(jp[b, 0] * sx, 0, W - 1)),
              int(np.clip(jp[b, 1] * sy, 0, H - 1)))
        c = (0, 255, 0)
        if ts is not None:
            ta = ts[a] if a < len(ts) else 0
            tb = ts[b] if b < len(ts) else 0
            c = ((0, 255, 0) if ta == 2 and tb == 2
                 else (0, 255, 255) if ta >= 1 and tb >= 1
                 else (0, 0, 255))
        cv2.line(fr, pa, pb, c, 3, cv2.LINE_AA)

    for j in range(min(len(jp), NUM_JOINTS)):
        pt = (int(np.clip(jp[j, 0] * sx, 0, W - 1)),
              int(np.clip(jp[j, 1] * sy, 0, H - 1)))
        jc = (255, 255, 0)
        if ts is not None and j < len(ts):
            jc = ((255, 255, 0) if ts[j] == 2
                  else (0, 165, 255) if ts[j] == 1
                  else (128, 128, 128))
        cv2.circle(fr, pt, 5, jc, -1)
        cv2.circle(fr, pt, 5, (0, 0, 0), 1)


def draw_hud(fr, lbl, conf, prbs, fps, stats):
    H, W = fr.shape[:2]

    # Activity label
    ld = lbl.upper() if lbl else "N/A"
    lc = ((0, 255, 100) if conf >= 0.7 and "Uncertain" not in lbl
          else (0, 230, 255) if conf >= 0.5
          else (100, 100, 255))

    text_pos = (15, 45)
    cv2.putText(fr, f"Activity: {ld}", text_pos, FONT, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(fr, f"Activity: {ld}", text_pos, FONT, 1.0, lc, 2, cv2.LINE_AA)

    # Confidence
    conf_pos = (15, 80)
    cv2.putText(fr, f"Confidence: {conf * 100:.1f}%", conf_pos, FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(fr, f"Confidence: {conf * 100:.1f}%", conf_pos, FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Confidence bar
    bxs, bxe = 200, W - 260
    bw = int((bxe - bxs) * min(conf, 1.0))
    bc = ((0, 200, 0) if conf >= 0.7
          else (0, 180, 255) if conf >= 0.5
          else (0, 0, 200))
    cv2.rectangle(fr, (bxs, 65), (bxs + bw, 85), bc, -1)
    cv2.rectangle(fr, (bxs, 65), (bxe, 85), (100, 100, 100), 2)

    # Stats
    bf = stats['buffer_fill']
    cal = "✓" if stats['calibrated'] else "..."
    skip_rate = stats['skip_rate']

    stats_text = (f"FPS:{fps:.0f} | Buf:{bf:.0f}% | Inf:{stats.get('inf_ms', 0):.1f}ms | "
                  f"Skip:{skip_rate:.1f}% | Cal:{cal} | Model:{QUANTIZED_MODEL_TYPE}")
    stats_pos = (15, 120)
    cv2.putText(fr, stats_text, stats_pos, FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(fr, stats_text, stats_pos, FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Buffer bar
    bbw = int(180 * (bf / 100.0))
    cv2.rectangle(fr, (15, 130), (15 + bbw, 142), (255, 200, 0), -1)
    cv2.rectangle(fr, (15, 130), (195, 142), (100, 100, 100), 2)

    # Motion info
    motion_info = stats.get('motion_info', '')
    if motion_info:
        motion_pos = (15, 162)
        cv2.putText(fr, f"Motion: {motion_info}", motion_pos, FONT, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(fr, f"Motion: {motion_info}", motion_pos, FONT, 0.42, (150, 200, 255), 1, cv2.LINE_AA)

    # Probabilities panel
    if prbs:
        sp = sorted(prbs.items(), key=lambda x: x[1], reverse=True)
        px, py = W - 245, 150

        title_pos = (px - 5, py + 5)
        cv2.putText(fr, "Probabilities:", title_pos, FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(fr, "Probabilities:", title_pos, FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        for i, (cn, p) in enumerate(sp[:6]):
            y = py + 20 + i * 28
            bl = int(205 * p)
            c = ((0, 255, 120) if cn.lower() in lbl.lower()
                 else (150, 110, 80))

            cv2.rectangle(fr, (px, y), (px + bl, y + 18), c, -1)
            cv2.rectangle(fr, (px, y), (px + 205, y + 18), (80, 80, 80), 2)

            prob_text = f"{cn}:{p * 100:.1f}%"
            prob_pos = (px + 3, y + 14)
            cv2.putText(fr, prob_text, prob_pos, FONT, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(fr, prob_text, prob_pos, FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def draw_no_body(fr):
    H, W = fr.shape[:2]
    overlay = fr.copy()
    cv2.rectangle(overlay, (0, H - 100), (W, H), (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.8, fr, 0.2, 0, fr)

    msgs = [
        ("No skeleton detected", 0.75, (0, 0, 255), 2, H - 65),
        ("Stand facing Kinect | 1.5-3m distance | Full body visible",
         0.5, (0, 200, 255), 1, H - 38),
        ("Ensure good lighting and depth sensor can see all joints",
         0.42, (180, 180, 180), 1, H - 15),
    ]

    for txt, scale, color, thick, y in msgs:
        sz = cv2.getTextSize(txt, FONT, scale, thick)[0]
        cv2.putText(fr, txt, ((W - sz[0]) // 2, y),
                    FONT, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────

def run_realtime(mode="kinect"):
    # Load artifacts (TFLite model)
    interpreter, input_details, output_details, scaler, le, ws, nf, wst = load_artifacts()

    # Initialize sensor
    if mode == "kinect":
        sensor = KinectV2Sensor()
    else:
        sensor = MockSensor()

    # Initialize TFLite predictor
    predictor = OptimizedTFLitePredictor(
        interpreter, input_details, output_details,
        scaler, le, ws, nf, wst
    )

    # Setup display
    wn = f"Real-Time HAR - {QUANTIZED_MODEL_TYPE.upper()}"
    cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wn, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # FPS tracking
    fps_history = collections.deque(maxlen=30)
    prev_time = time.time()
    frame_count = 0
    last_color_frame = None

    print("\n" + "=" * 70)
    print(f"  REAL-TIME HAR STARTED (Model: {QUANTIZED_MODEL_TYPE})")
    print("  Press: Q=Quit | R=Reset | C=Recalibrate | S=Screenshot")
    print("=" * 70 + "\n")

    try:
        while True:
            # Get sensor data
            color_frame, bodies = sensor.update()

            if color_frame is not None:
                last_color_frame = color_frame

            # Prepare display
            if last_color_frame is not None:
                display = cv2.resize(last_color_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                oh, ow = last_color_frame.shape[:2]
                sx, sy = DISPLAY_WIDTH / ow, DISPLAY_HEIGHT / oh
            else:
                display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
                sx, sy = 1.0, 1.0

            # Process bodies
            num_bodies = len(bodies)
            if num_bodies > 0:
                body = bodies[0]
                predictor.push(body['joints_xyz'], body['tracking_state'])

                draw_skel(display, body['joints_pixel'],
                         body['tracking_state'], sx, sy)

                for extra_body in bodies[1:]:
                    draw_skel(display, extra_body['joints_pixel'],
                             extra_body['tracking_state'], sx, sy)
            else:
                draw_no_body(display)

            # FPS
            current_time = time.time()
            fps_history.append(current_time - prev_time)
            prev_time = current_time
            fps = 1.0 / max(np.mean(fps_history), 1e-6)

            # Get prediction
            label, conf, probs, inf_ms = predictor.get()
            stats = predictor.get_stats()
            stats['inf_ms'] = inf_ms

            # Draw HUD
            draw_hud(display, label, conf, probs, fps, stats)

            # Controls
            cv2.putText(display, "Q:Quit | R:Reset | C:Recalibrate | S:Save",
                        (10, display.shape[0] - 10), FONT, 0.4,
                        (120, 120, 120), 1, cv2.LINE_AA)

            # Show
            cv2.imshow(wn, display)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break
            elif key == ord('r'):
                predictor.reset()
                print("[INFO] Predictor reset")
            elif key == ord('c'):
                predictor.recalibrate()
                print("[INFO] Recalibration started")
            elif key == ord('s'):
                filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, display)
                print(f"[SAVE] {filename}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        sensor.close()
        cv2.destroyAllWindows()

        stats = predictor.get_stats()
        print("\n" + "=" * 70)
        print("  SESSION STATISTICS")
        print("=" * 70)
        print(f"  Model type: {QUANTIZED_MODEL_TYPE}")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Frames sent to predictor: {stats['total_frames']}")
        print(f"  Frames skipped (quality): {stats['skipped_frames']} ({stats['skip_rate']:.1f}%)")
        print(f"  Average FPS: {fps:.1f}")
        print("=" * 70)
        print("  Goodbye!")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time HAR with Quantized TFLite Model"
    )
    parser.add_argument(
        "--mode",
        default="kinect",
        choices=["kinect", "mock"],
        help="Sensor mode (kinect or mock)"
    )
    parser.add_argument(
        "--model",
        default="dynamic_int8",
        choices=["dynamic_int8", "full_int8", "float16"],
        help="Which quantized model to use"
    )

    args = parser.parse_args()

    # Override model selection
    QUANTIZED_MODEL_TYPE = args.model
    MODEL_PATH = MODEL_PATHS[QUANTIZED_MODEL_TYPE]

    run_realtime(mode=args.mode)