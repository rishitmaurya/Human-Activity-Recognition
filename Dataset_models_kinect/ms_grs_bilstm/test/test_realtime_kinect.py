# realtime_har_kinect_v2.py - COMPLETE FIXED VERSION
"""
Real-Time HAR using MS-GRS-BiLSTM + Kinect v2
"""

import os
import sys
import time
import json
import struct
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
try:
    tf.keras.config.enable_unsafe_deserialization()
except:
    pass
try:
    import keras
    keras.config.enable_unsafe_deserialization()
except:
    pass

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR    = r"Dataset_models_kinect\ms_grs_bilstm\test"
MODEL_PATH   = os.path.join(MODEL_DIR, "ms_grs_bilstm_model.h5")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
LE_PATH      = os.path.join(MODEL_DIR, "label_encoder.pkl")
CONFIG_PATH  = os.path.join(MODEL_DIR, "model_config.json")

DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540
FONT           = cv2.FONT_HERSHEY_SIMPLEX
SMOOTHING_WINDOW = 10
CONFIDENCE_THRESHOLD = 0.45
NUM_JOINTS = 25

BONES = [
    (0,1),(1,20),(20,2),(2,3),
    (20,4),(4,5),(5,6),(6,7),(7,21),(7,22),
    (20,8),(8,9),(9,10),(10,11),(11,23),(11,24),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19)
]


# ─────────────────────────────────────────────
# PATCH numpy.object in PyKinectRuntime
# Must be done BEFORE importing PyKinectRuntime
# ─────────────────────────────────────────────
def patch_numpy_in_pykinect():
    """
    Patches numpy.object → object in PyKinectRuntime.py
    This is critical — without it, body frame parsing
    silently fails and no bodies are ever returned.
    """
    try:
        import pykinect2
        pkg = os.path.dirname(pykinect2.__file__)
        rt_file = os.path.join(pkg, "PyKinectRuntime.py")

        with open(rt_file, 'r', encoding='utf-8',
                  errors='ignore') as f:
            content = f.read()

        if 'numpy.object)' not in content:
            return True  # Already fixed

        import shutil
        backup = rt_file + ".numpy_bak"
        if not os.path.exists(backup):
            shutil.copy2(rt_file, backup)

        # Fix all deprecated numpy types
        fixes = [
            ('numpy.object)',  'object)'),
            ('numpy.object,',  'object,'),
            ('numpy.bool)',    'bool)'),
            ('numpy.bool,',    'bool,'),
            ('numpy.int)',     'int)'),
            ('numpy.int,',     'int,'),
            ('numpy.float)',   'float)'),
            ('numpy.float,',   'float,'),
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
# Model
# ─────────────────────────────────────────────
def gated_residual_skip_lstm_block(x_in, units, name_prefix,
                                    skip_connection=None,
                                    dropout=0.2):
    from tensorflow.keras import layers
    td = layers.TimeDistributed(
        layers.Dense(units, activation=None),
        name=f"{name_prefix}_proj_td")(x_in)
    lstm = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True),
        name=f"{name_prefix}_bilstm")(x_in)
    lstm = layers.TimeDistributed(
        layers.Dense(units),
        name=f"{name_prefix}_lstm_proj")(lstm)
    lstm = layers.Dropout(dropout)(lstm)
    cat = layers.Concatenate(axis=-1)([td, lstm])
    gate = layers.TimeDistributed(
        layers.Dense(units, activation="sigmoid"),
        name=f"{name_prefix}_gate")(cat)
    gated = layers.Multiply()([gate, lstm])
    inv = layers.Lambda(lambda z: 1.0 - z)(gate)
    res = layers.Multiply()([inv, td])
    out = layers.Add()([gated, res])
    if skip_connection is not None:
        out = layers.Add()([out, skip_connection])
    return out


def build_model_from_scratch(ws, nf, nc, units=64, dropout=0.2):
    from tensorflow.keras import layers, Model, Input
    inp = Input(shape=(ws, nf), name="input_seq")
    s0 = inp
    s1 = layers.Lambda(lambda x: x[:, ::2, :],
                       name="downsample_2")(inp)
    s2 = layers.Lambda(lambda x: x[:, ::4, :],
                       name="downsample_4")(inp)
    outs = []
    for i, s in enumerate([s0, s1, s2]):
        p = f"scale{i}"
        b1 = gated_residual_skip_lstm_block(
            s, units, f"{p}_b1", None, dropout)
        b2 = gated_residual_skip_lstm_block(
            b1, units, f"{p}_b2", b1, dropout)
        pool = layers.GlobalAveragePooling1D(
            name=f"{p}_pool")(b2)
        outs.append(pool)
    fused = layers.Concatenate(name="concat_scales")(outs)
    fused = layers.Dense(128, activation="relu",
                         name="fusion_dense")(fused)
    fused = layers.Dropout(0.3)(fused)
    out = layers.Dense(nc, activation="softmax",
                       name="classifier")(fused)
    return Model(inputs=inp, outputs=out)


def load_model_safe(path, ws, nf, nc):
    print(f"  Loading: {path}")
    for i, (name, fn) in enumerate([
        ("load_model(safe_mode=False)",
         lambda: tf.keras.models.load_model(
             path, compile=False, safe_mode=False)),
        ("load_model(compile=False)",
         lambda: tf.keras.models.load_model(
             path, compile=False)),
        ("rebuild+load_weights",
         lambda: _rebuild_load(path, ws, nf, nc)),
    ], 1):
        try:
            print(f"  [Attempt {i}] {name}")
            m = fn()
            print(f"  [Attempt {i}] SUCCESS")
            return m
        except Exception as e:
            print(f"  [Attempt {i}] {e}")
    print("  FATAL: Cannot load model")
    sys.exit(1)


def _rebuild_load(path, ws, nf, nc):
    m = build_model_from_scratch(ws, nf, nc)
    m.load_weights(path)
    return m


def load_artifacts():
    print("=" * 60)
    print("  LOADING MODEL ARTIFACTS")
    print("=" * 60)
    for p, n in [(MODEL_PATH,"Model"),(SCALER_PATH,"Scaler"),
                 (LE_PATH,"Labels"),(CONFIG_PATH,"Config")]:
        if not os.path.exists(p):
            print(f"  [ERROR] {n} missing: {p}")
            sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    ws  = cfg["WINDOW_SIZE"]
    nf  = cfg["NUM_FEATURES"]
    wst = cfg.get("WINDOW_STRIDE", 16)
    print(f"  Window={ws} Features={nf} Stride={wst}")
    le = joblib.load(LE_PATH)
    nc = len(le.classes_)
    print(f"  Labels({nc}): {list(le.classes_)}")
    model  = load_model_safe(MODEL_PATH, ws, nf, nc)
    scaler = joblib.load(SCALER_PATH)
    dummy  = np.zeros((1, ws, nf), np.float32)
    pred   = model.predict(dummy, verbose=0)
    print(f"  Verify OK shape:{pred.shape} sum:{pred.sum():.4f}")
    print("=" * 60)
    return model, scaler, le, ws, nf, wst


# ═══════════════════════════════════════════════
# KINECT V2 SENSOR
# ═══════════════════════════════════════════════
class KinectV2Sensor:
    def __init__(self):
        print("\n[KINECT] Initialising Kinect v2...")

        # ── Step 1: Patch numpy BEFORE import ──
        print("  Patching numpy deprecations...")
        patch_numpy_in_pykinect()

        # ── Step 2: Import pykinect2 ──
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

        # ── Step 3: Verify numpy patch took effect ──
        # After patching the file, we need to reload the module
        import importlib
        import pykinect2.PyKinectRuntime as _rt_mod
        importlib.reload(_rt_mod)
        self._PKR = _rt_mod
        print("  [OK] PyKinectRuntime reloaded (numpy fix active)")

        # ── Step 4: SDK check ──
        sdk = os.environ.get('KINECTSDK20_DIR', '')
        if sdk:
            print(f"  [OK] SDK: {sdk}")

        # ── Step 5: Create runtime ──
        self._runtime = None
        self._bodies  = None
        self._color_enabled = True
        self._frame_count   = 0
        self._body_errors   = 0

        print("  Connecting to sensor...")
        try:
            self._runtime = self._PKR.PyKinectRuntime(
                self._PK.FrameSourceTypes_Color |
                self._PK.FrameSourceTypes_Body)
            print(f"  [OK] Color + Body active")
            print(f"  [OK] Max bodies: "
                  f"{self._runtime.max_body_count}")
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

        # ── Step 6: Warm up with detailed debug ──
        print("  Warming up (5 seconds)...")
        found = False
        t0 = time.time()
        frame_cnt   = 0
        body_cnt    = 0
        tracked_cnt = 0

        while time.time() - t0 < 5.0:
            try:
                if self._runtime.has_new_body_frame():
                    bf = self._runtime.get_last_body_frame()
                    frame_cnt += 1
                    if bf is not None:
                        body_cnt += 1
                        mc = self._runtime.max_body_count
                        for i in range(mc):
                            try:
                                b = bf.bodies[i]
                                if b.is_tracked:
                                    tracked_cnt += 1
                                    found = True
                                    break
                            except Exception as be:
                                pass
                    if found:
                        break
            except Exception as e:
                pass
            time.sleep(0.033)

        print(f"  Body frames received: {frame_cnt}")
        print(f"  Frames with data:     {body_cnt}")
        print(f"  Tracked bodies found: {tracked_cnt}")

        if found:
            print("  [OK] Body detected!")
        else:
            print("  [INFO] No body during warmup")
            print("  System will continue trying...")

        print("[KINECT] Ready.\n")

    def update(self):
        color_frame = None
        bodies_out  = []

        self._frame_count += 1

        # ── Color frame ──
        if self._color_enabled:
            try:
                if self._runtime.has_new_color_frame():
                    f = self._runtime.get_last_color_frame()
                    color_frame = f.reshape(
                        (1080, 1920, 4)).astype(np.uint8)
                    color_frame = cv2.cvtColor(
                        color_frame, cv2.COLOR_BGRA2BGR)
            except Exception:
                pass

        # ── Body frame ──
        try:
            if self._runtime.has_new_body_frame():
                self._bodies = \
                    self._runtime.get_last_body_frame()
        except Exception:
            pass

        # ── Parse body data ──
        if self._bodies is not None:
            try:
                mc = self._runtime.max_body_count
                for i in range(mc):
                    try:
                        body = self._bodies.bodies[i]
                        # Safely check is_tracked
                        try:
                            is_tracked = bool(body.is_tracked)
                        except Exception:
                            is_tracked = False

                        if not is_tracked:
                            continue

                        joints = body.joints
                        xyz = np.zeros(
                            (NUM_JOINTS, 3), np.float32)
                        pix = np.zeros(
                            (NUM_JOINTS, 2), np.int32)
                        trk = np.zeros(
                            NUM_JOINTS, np.int32)

                        for j in range(NUM_JOINTS):
                            try:
                                pos = joints[j].Position
                                xyz[j, 0] = float(pos.x)
                                xyz[j, 1] = float(pos.y)
                                xyz[j, 2] = float(pos.z)
                                trk[j] = int(
                                    joints[j].TrackingState)
                            except Exception:
                                pass

                        # Project to color pixels
                        try:
                            cp = self._runtime \
                                .body_joints_to_color_space(
                                    joints)
                            for j in range(NUM_JOINTS):
                                try:
                                    xv = float(cp[j].x)
                                    yv = float(cp[j].y)
                                    if (np.isfinite(xv) and
                                            np.isfinite(yv) and
                                            0 <= xv <= 1920 and
                                            0 <= yv <= 1080):
                                        pix[j] = [
                                            int(xv), int(yv)]
                                    elif xyz[j, 2] > 0:
                                        pix[j, 0] = int(
                                            xyz[j,0]/xyz[j,2]
                                            * 525 + 960)
                                        pix[j, 1] = int(
                                            -xyz[j,1]/xyz[j,2]
                                            * 525 + 540)
                                except Exception:
                                    pass
                        except Exception:
                            for j in range(NUM_JOINTS):
                                if xyz[j, 2] > 0:
                                    pix[j, 0] = int(
                                        xyz[j,0]/xyz[j,2]
                                        * 525 + 960)
                                    pix[j, 1] = int(
                                        -xyz[j,1]/xyz[j,2]
                                        * 525 + 540)

                        bodies_out.append({
                            'joints_xyz':    xyz,
                            'joints_pixel':  pix,
                            'tracking_state': trk
                        })

                    except Exception:
                        continue

            except Exception:
                pass

        # ── Periodic debug output ──
        if (self._frame_count % 150 == 0 and
                len(bodies_out) == 0):
            print(f"[KINECT] Frame {self._frame_count}: "
                  f"no tracked bodies. "
                  f"bodies obj={'OK' if self._bodies else 'None'}")

        return color_frame, bodies_out

    def close(self):
        try:
            self._runtime.close()
        except Exception:
            pass
        print("[KINECT] Closed.")


# ─────────────────────────────────────────────
# Mock Sensor
# ─────────────────────────────────────────────
class MockSensor:
    def __init__(self):
        self._t0   = time.time()
        self._acts = ["standing","walking","bending",
                      "jumping","sitting"]
        print("[MOCK] Ready\n")

    def _pose(self):
        xyz = np.zeros((NUM_JOINTS, 3), np.float32)
        y = -0.3
        xyz[0]=[0,y,2.5];    xyz[1]=[0,y+.25,2.5]
        xyz[20]=[0,y+.45,2.5]; xyz[2]=[0,y+.52,2.5]
        xyz[3]=[0,y+.62,2.5]
        xyz[4]=[-.18,y+.42,2.5]; xyz[5]=[-.25,y+.22,2.5]
        xyz[6]=[-.22,y+.02,2.5]; xyz[7]=[-.21,y-.03,2.5]
        xyz[21]=[-.20,y-.06,2.5]; xyz[22]=[-.23,y-.03,2.5]
        xyz[8]=[.18,y+.42,2.5];  xyz[9]=[.25,y+.22,2.5]
        xyz[10]=[.22,y+.02,2.5]; xyz[11]=[.21,y-.03,2.5]
        xyz[23]=[.20,y-.06,2.5]; xyz[24]=[.23,y-.03,2.5]
        xyz[12]=[-.10,y-.02,2.5]; xyz[13]=[-.12,y-.32,2.5]
        xyz[14]=[-.12,y-.62,2.5]; xyz[15]=[-.15,y-.65,2.45]
        xyz[16]=[.10,y-.02,2.5];  xyz[17]=[.12,y-.32,2.5]
        xyz[18]=[.12,y-.62,2.5];  xyz[19]=[.15,y-.65,2.45]
        return xyz

    def update(self):
        t   = time.time() - self._t0
        ai  = int(t / 5.0) % len(self._acts)
        act = self._acts[ai]
        xyz = self._pose()
        if act == "walking":
            s = .08 * np.sin(t * 4)
            xyz[14,0] += s; xyz[18,0] -= s
        elif act == "bending":
            b = .3 * (np.sin(t*1.5)*.5 + .5)
            for j in [1,2,3,4,5,6,7,8,9,10,11,20,21,22,23,24]:
                xyz[j,2] -= b*.3; xyz[j,1] -= b*.4
        elif act == "jumping":
            xyz[:,1] += max(0, np.sin(t*3)) * .2
        elif act == "sitting":
            xyz[0,1]-=.25; xyz[12,1]-=.2; xyz[16,1]-=.2
        xyz[:,0] += .01 * np.sin(t*1.5)

        frame = np.zeros(
            (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
        cv2.putText(frame, f"MOCK: {act.upper()}",
                    (10, DISPLAY_HEIGHT-40), FONT, 0.55,
                    (0,200,255), 1, cv2.LINE_AA)

        pix = np.zeros((NUM_JOINTS, 2), np.int32)
        for j in range(NUM_JOINTS):
            if xyz[j,2] > 0:
                pix[j,0] = int(
                    xyz[j,0]/xyz[j,2]*600 + DISPLAY_WIDTH//2)
                pix[j,1] = int(
                    -xyz[j,1]/xyz[j,2]*600 + DISPLAY_HEIGHT//2+50)

        return frame, [{'joints_xyz': xyz,
                        'joints_pixel': pix,
                        'tracking_state': np.full(
                            NUM_JOINTS, 2, np.int32)}]

    def close(self):
        print("[MOCK] Closed.")


# ─────────────────────────────────────────────
# Predictor
# ─────────────────────────────────────────────
class Predictor:
    def __init__(self, model, scaler, le, ws, nf, stride):
        self.model  = model
        self.scaler = scaler
        self.le     = le
        self.ws     = ws
        self.nf     = nf
        self.stride = stride
        self.buf    = collections.deque(maxlen=ws)
        self.hist   = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.label  = "Collecting..."
        self.conf   = 0.0
        self.probs  = {}
        self.inf_ms = 0.0
        self._fc    = 0
        self._q     = queue.Queue(maxsize=2)
        threading.Thread(target=self._run, daemon=True).start()
        print(f"[PRED] buf={ws} feat={nf} stride={stride}")

    def push(self, xyz):
        f = xyz.flatten().astype(np.float32)
        if len(f) < self.nf:
            f = np.pad(f, (0, self.nf - len(f)))
        elif len(f) > self.nf:
            f = f[:self.nf]
        self.buf.append(f)
        self._fc += 1
        if (len(self.buf) == self.ws and
                self._fc % max(1, self.stride) == 0):
            try:
                self._q.put_nowait(np.array(self.buf).copy())
            except queue.Full:
                pass

    def _run(self):
        while True:
            w = self._q.get()
            try:
                t0  = time.time()
                fl  = self.scaler.transform(
                    w.reshape(-1, self.nf))
                inp = fl.reshape(
                    1, self.ws, self.nf).astype(np.float32)
                p   = self.model.predict(inp, verbose=0)[0]
                i   = int(np.argmax(p))
                c   = float(p[i])
                ms  = (time.time() - t0) * 1000
                self.hist.append(i)
                sm  = collections.Counter(
                    self.hist).most_common(1)[0][0]
                self.label = (
                    str(self.le.classes_[sm])
                    if c >= CONFIDENCE_THRESHOLD
                    else f"Uncertain ({self.le.classes_[i]}?)")
                self.conf  = c
                self.probs = {
                    str(self.le.classes_[j]): float(p[j])
                    for j in range(len(p))}
                self.inf_ms = ms
            except Exception as e:
                print(f"[WARN] Inference: {e}")
            finally:
                self._q.task_done()

    def get(self):
        return self.label, self.conf, self.probs, self.inf_ms
    def fill(self):
        return len(self.buf) / self.ws * 100
    def reset(self):
        self.buf.clear(); self.hist.clear()
        self.label = "Collecting..."
        self.conf  = 0.0
        self.probs = {}
        self._fc   = 0


# ─────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────
def draw_skel(fr, jp, ts=None, sx=1.0, sy=1.0):
    H, W = fr.shape[:2]
    for a, b in BONES:
        if a >= len(jp) or b >= len(jp):
            continue
        pa = (int(np.clip(jp[a,0]*sx, 0, W-1)),
              int(np.clip(jp[a,1]*sy, 0, H-1)))
        pb = (int(np.clip(jp[b,0]*sx, 0, W-1)),
              int(np.clip(jp[b,1]*sy, 0, H-1)))
        c = (0, 255, 0)
        if ts is not None:
            ta = ts[a] if a < len(ts) else 0
            tb = ts[b] if b < len(ts) else 0
            c = ((0,255,0)   if ta==2 and tb==2
                 else (0,255,255) if ta>=1 and tb>=1
                 else (0,0,255))
        cv2.line(fr, pa, pb, c, 3, cv2.LINE_AA)
    for j in range(min(len(jp), NUM_JOINTS)):
        pt = (int(np.clip(jp[j,0]*sx, 0, W-1)),
              int(np.clip(jp[j,1]*sy, 0, H-1)))
        jc = (255, 255, 0)
        if ts is not None and j < len(ts):
            jc = ((255,255,0)   if ts[j]==2
                  else (0,165,255) if ts[j]==1
                  else (128,128,128))
        cv2.circle(fr, pt, 5, jc, -1)
        cv2.circle(fr, pt, 5, (0,0,0), 1)


def draw_hud(fr, lbl, conf, prbs, fps, bp, ims, nb):
    H, W = fr.shape[:2]
    ov = fr.copy()
    cv2.rectangle(ov, (0,0), (W,140), (15,15,15), -1)
    cv2.addWeighted(ov, 0.7, fr, 0.3, 0, fr)
    ld = lbl.upper() if lbl else "N/A"
    lc = ((0,255,100)  if conf >= 0.8
          else (0,230,255) if conf >= 0.5
          else (100,100,255))
    cv2.putText(fr, f"Activity: {ld}",
                (15,45), FONT, 1.1, lc, 2, cv2.LINE_AA)
    cv2.putText(fr, f"Confidence: {conf*100:.1f}%",
                (15,78), FONT, 0.6, (200,200,200), 1,
                cv2.LINE_AA)
    bxs, bxe = 220, W-250
    bw = int((bxe-bxs) * min(conf, 1.0))
    bc = ((0,200,0)   if conf >= 0.7
          else (0,180,255) if conf >= 0.5
          else (0,0,200))
    cv2.rectangle(fr, (bxs,63), (bxs+bw,83), bc, -1)
    cv2.rectangle(fr, (bxs,63), (bxe,83), (100,100,100), 1)
    cv2.putText(fr,
                f"FPS:{fps:.0f} | Buf:{bp:.0f}% | "
                f"Inf:{ims:.0f}ms | Bodies:{nb}",
                (15,115), FONT, 0.5, (150,150,150), 1,
                cv2.LINE_AA)
    bbw = int(200 * (bp/100.0))
    cv2.rectangle(fr, (15,125), (15+bbw,135),
                  (255,200,0), -1)
    cv2.rectangle(fr, (15,125), (215,135),
                  (100,100,100), 1)
    if prbs:
        sp  = sorted(prbs.items(),
                     key=lambda x: x[1], reverse=True)
        px, py = W-240, 150
        ph = len(sp)*30 + 10
        o2 = fr.copy()
        cv2.rectangle(o2, (px-10,py-10),
                      (W-5, py+ph), (15,15,15), -1)
        cv2.addWeighted(o2, 0.7, fr, 0.3, 0, fr)
        cv2.putText(fr, "Probabilities:",
                    (px-5, py+5), FONT, 0.45,
                    (200,200,200), 1, cv2.LINE_AA)
        for i, (cn, p) in enumerate(sp):
            y  = py + 20 + i*28
            bl = int(200*p)
            c  = ((0,255,100) if cn.lower()==lbl.lower()
                  else (150,100,80))
            cv2.rectangle(fr, (px,y), (px+bl,y+18), c, -1)
            cv2.rectangle(fr, (px,y), (px+200,y+18),
                          (80,80,80), 1)
            cv2.putText(fr, f"{cn}:{p*100:.1f}%",
                        (px+3,y+14), FONT, 0.4,
                        (255,255,255), 1, cv2.LINE_AA)


def draw_no_body(fr):
    H, W = fr.shape[:2]
    overlay = fr.copy()
    cv2.rectangle(overlay, (0, H-100), (W, H),
                  (0,0,80), -1)
    cv2.addWeighted(overlay, 0.8, fr, 0.2, 0, fr)
    msgs = [
        ("No skeleton detected", 0.75, (0,0,255), 2, H-65),
        ("Stand facing Kinect | 1.5-3m away | Full body visible",
         0.5, (0,200,255), 1, H-38),
        ("Make sure Kinect depth sensor can see HEAD to FEET",
         0.42, (180,180,180), 1, H-15),
    ]
    for txt, scale, color, thick, y in msgs:
        sz = cv2.getTextSize(txt, FONT, scale, thick)[0]
        cv2.putText(fr, txt, ((W-sz[0])//2, y),
                    FONT, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run_realtime(mode="kinect"):
    model, scaler, le, ws, nf, wst = load_artifacts()

    if mode == "kinect":
        sensor = KinectV2Sensor()
    else:
        sensor = MockSensor()

    pred = Predictor(model, scaler, le, ws, nf, wst)
    wn   = "Real-Time HAR"
    cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wn, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    fps_d = collections.deque(maxlen=30)
    pt    = time.time()
    fc    = 0
    lcf   = None

    print("\n" + "="*60)
    print("  REAL-TIME HAR STARTED")
    print("  q=quit  r=reset  s=screenshot")
    print("="*60 + "\n")

    try:
        while True:
            cf, bodies = sensor.update()
            if cf is not None:
                lcf = cf

            if lcf is not None:
                disp = cv2.resize(lcf,
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                oh, ow = lcf.shape[:2]
                sx, sy = DISPLAY_WIDTH/ow, DISPLAY_HEIGHT/oh
            else:
                disp = np.zeros(
                    (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
                sx, sy = 1.0, 1.0

            nb = len(bodies)
            if nb > 0:
                b0 = bodies[0]
                pred.push(b0['joints_xyz'])
                draw_skel(disp, b0['joints_pixel'],
                          b0['tracking_state'], sx, sy)
                for bx in bodies[1:]:
                    draw_skel(disp, bx['joints_pixel'],
                              bx['tracking_state'], sx, sy)
            else:
                draw_no_body(disp)

            now = time.time()
            fps_d.append(now - pt)
            pt  = now
            fps = 1.0 / max(np.mean(fps_d), 1e-6)

            lbl, conf, prbs, ims = pred.get()
            bp = pred.fill()
            draw_hud(disp, lbl, conf, prbs,
                     fps, bp, ims, nb)
            cv2.putText(disp, "Q:Quit R:Reset S:Save",
                        (10, disp.shape[0]-10), FONT, 0.4,
                        (120,120,120), 1, cv2.LINE_AA)
            cv2.imshow(wn, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                pred.reset()
                print("[INFO] Reset")
            elif k == ord('s'):
                fn = f"har_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(fn, disp)
                print(f"[SAVE] {fn}")
            fc += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        sensor.close()
        cv2.destroyAllWindows()
        print(f"[INFO] {fc} frames. Bye!")


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Real-Time HAR")
    pa.add_argument("--mode", default="kinect",
                    choices=["kinect","mock"])
    args = pa.parse_args()
    run_realtime(mode=args.mode)