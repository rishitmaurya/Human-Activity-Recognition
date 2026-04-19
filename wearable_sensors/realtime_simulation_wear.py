import pandas as pd
import numpy as np
import os
import time
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from collections import deque
import csv
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
ENABLE_LOGGING = False  # Toggle logging on/off

DATASET_PATH = r'wearable_sensors/dataset_wearable/combined_har_dataset.csv'
MODEL_DIR = r'wearable_sensors/ms_grs_bilstm_wear'
LOG_DIR = r'wearable_sensors/simulation_logs'

WINDOW_SIZE = 128
STEP_SIZE = 32
BUFFER_DISPLAY = 200  # Number of points to display in rolling plots

# Activity color mapping
ACTIVITY_COLORS = {
    'bending': '#FF6B6B',
    'cycling': '#4ECDC4',
    'lying': '#45B7D1',
    'sitting': '#96CEB4',
    'squats': '#FFEAA7',
    'standing': '#DDA0DD',
    'walking': '#98D8C8',
}
DEFAULT_COLOR = '#FFFFFF'

# =========================================================
# GPU SETUP
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# MODEL ARCHITECTURE (Must match training exactly)
# =========================================================
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.att(x), dim=1)
        return (x * weights).sum(dim=1)


class MSGRSBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv3 = nn.Conv1d(8, 64, 3, padding=1)
        self.conv5 = nn.Conv1d(8, 64, 5, padding=2)
        self.conv7 = nn.Conv1d(8, 64, 7, padding=3)

        self.bn = nn.BatchNorm1d(192)
        self.gate = nn.Conv1d(192, 192, 1)
        self.shortcut = nn.Conv1d(8, 192, 1)

        self.lstm1 = nn.LSTM(192, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        self.attention = Attention(128)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        ms = torch.cat([
            torch.relu(self.conv3(x)),
            torch.relu(self.conv5(x)),
            torch.relu(self.conv7(x))
        ], dim=1)

        ms = self.bn(ms)
        gate = torch.sigmoid(self.gate(ms))
        x = torch.relu(ms * gate + self.shortcut(x))

        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.attention(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =========================================================
# LOAD ASSETS
# =========================================================
def load_assets():
    print("Loading scaler and label encoder...")
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    num_classes = len(le.classes_)

    print(f"Classes: {le.classes_}")
    print("Loading model...")

    model = MSGRSBiLSTM(num_classes).to(device)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'final_model_98.pt'),
        map_location=device
    ))
    model.eval()
    print("Model loaded successfully!")
    return model, scaler, le


# =========================================================
# LOAD & FILTER DATASET
# =========================================================
def load_person_data():
    print("Loading dataset for Person_1...")
    df = pd.read_csv(DATASET_PATH)
    df_p1 = df[df['person_id'] == 'Person_1'].reset_index(drop=True)
    print(f"Person_1 has {len(df_p1)} data points")
    print(f"Activities: {df_p1['activity'].unique()}")
    return df_p1


# =========================================================
# LOGGING SETUP
# =========================================================
def setup_logger():
    if not ENABLE_LOGGING:
        return None, None
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    log_file = open(log_filename, 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['timestamp', 'true_activity', 'predicted_activity', 'confidence', 'all_probabilities'])
    print(f"Logging enabled: {log_filename}")
    return log_file, writer


def log_prediction(writer, timestamp, true_activity, predicted_activity, confidence, all_probs):
    if not ENABLE_LOGGING or writer is None:
        return
    writer.writerow([timestamp, true_activity, predicted_activity, f"{confidence:.4f}", str(all_probs)])


# =========================================================
# SIMULATION CLASS
# =========================================================
class RealTimeSimulation:
    def __init__(self, df, model, scaler, le):
        self.df = df
        self.model = model
        self.scaler = scaler
        self.le = le
        self.classes = le.classes_
        self.num_classes = len(self.classes)

        # Data buffers for display
        self.buf_acc_x = deque(maxlen=BUFFER_DISPLAY)
        self.buf_acc_y = deque(maxlen=BUFFER_DISPLAY)
        self.buf_acc_z = deque(maxlen=BUFFER_DISPLAY)
        self.buf_gyro_x = deque(maxlen=BUFFER_DISPLAY)
        self.buf_gyro_y = deque(maxlen=BUFFER_DISPLAY)
        self.buf_gyro_z = deque(maxlen=BUFFER_DISPLAY)
        self.buf_acc_mag = deque(maxlen=BUFFER_DISPLAY)
        self.buf_gyro_mag = deque(maxlen=BUFFER_DISPLAY)

        # Raw window buffer for prediction
        self.raw_window = deque(maxlen=WINDOW_SIZE)

        # Prediction state
        self.current_pred = "Initializing..."
        self.current_conf = 0.0
        self.current_probs = np.zeros(self.num_classes)
        self.current_true = "..."
        self.prediction_history = deque(maxlen=50)
        self.true_history = deque(maxlen=50)
        self.correct_predictions = 0
        self.total_predictions = 0

        # Data index
        self.data_idx = 0
        self.step_counter = 0

        # Timing
        self.timestamps = df['time'].values
        self.start_sim_time = time.time()
        self.start_data_time = self.timestamps[0]

        # Logging
        self.log_file, self.log_writer = setup_logger()

        # Features
        self.features = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_mag', 'gyro_mag']

        # Compute magnitudes upfront
        self.df = self.df.copy()
        self.df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        self.df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

        # Smoothed probabilities for stable display
        self.smoothed_probs = np.ones(self.num_classes) / self.num_classes

    def predict_window(self):
        if len(self.raw_window) < WINDOW_SIZE:
            return

        win = np.array(self.raw_window)
        win_scaled = self.scaler.transform(win)
        tensor = torch.tensor(win_scaled[np.newaxis, :, :], dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Exponential moving average for smooth probability display
        alpha = 0.4
        self.smoothed_probs = alpha * probs + (1 - alpha) * self.smoothed_probs

        pred_idx = np.argmax(self.smoothed_probs)
        self.current_pred = self.classes[pred_idx]
        self.current_conf = self.smoothed_probs[pred_idx]
        self.current_probs = self.smoothed_probs.copy()

        # Track accuracy
        if self.current_true != "...":
            self.total_predictions += 1
            if self.current_pred == self.current_true:
                self.correct_predictions += 1

        self.prediction_history.append(self.current_pred)
        self.true_history.append(self.current_true)

        # Log
        log_prediction(
            self.log_writer,
            self.timestamps[self.data_idx] if self.data_idx < len(self.timestamps) else "END",
            self.current_true,
            self.current_pred,
            self.current_conf,
            {c: float(f"{p:.4f}") for c, p in zip(self.classes, self.current_probs)}
        )

    def ingest_row(self):
        if self.data_idx >= len(self.df):
            return False

        row = self.df.iloc[self.data_idx]
        feature_vals = [row[f] for f in self.features]

        self.buf_acc_x.append(row['acc_x'])
        self.buf_acc_y.append(row['acc_y'])
        self.buf_acc_z.append(row['acc_z'])
        self.buf_gyro_x.append(row['gyro_x'])
        self.buf_gyro_y.append(row['gyro_y'])
        self.buf_gyro_z.append(row['gyro_z'])
        self.buf_acc_mag.append(row['acc_mag'])
        self.buf_gyro_mag.append(row['gyro_mag'])

        self.raw_window.append(feature_vals)
        self.current_true = row['activity']

        self.step_counter += 1
        if self.step_counter >= STEP_SIZE and len(self.raw_window) == WINDOW_SIZE:
            self.predict_window()
            self.step_counter = 0

        self.data_idx += 1
        return True


# =========================================================
# VISUALIZATION SETUP
# =========================================================
def setup_figure(sim):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#0D1117')
    fig.suptitle(
        'Real-Time Human Activity Recognition',
        fontsize=16, fontweight='bold', color='white', y=0.98
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.93, bottom=0.06
    )

    ax_acc = fig.add_subplot(gs[0, :2])
    ax_gyro = fig.add_subplot(gs[1, :2])
    ax_mag = fig.add_subplot(gs[2, :2])
    ax_pred_hist = fig.add_subplot(gs[3, :2])

    ax_conf_bar = fig.add_subplot(gs[0:2, 2])
    ax_info = fig.add_subplot(gs[2:4, 2])

    axes_style = [ax_acc, ax_gyro, ax_mag, ax_pred_hist, ax_conf_bar, ax_info]
    for ax in axes_style:
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')

    # Accelerometer
    ax_acc.set_title('Accelerometer (m/s²)', color='#58A6FF', fontsize=10, pad=6)
    ax_acc.set_ylabel('Acceleration', color='#8B949E', fontsize=8)
    ax_acc.grid(alpha=0.15, color='#30363D')

    # Gyroscope
    ax_gyro.set_title('Gyroscope (rad/s)', color='#58A6FF', fontsize=10, pad=6)
    ax_gyro.set_ylabel('Angular Velocity', color='#8B949E', fontsize=8)
    ax_gyro.grid(alpha=0.15, color='#30363D')

    # Magnitude
    ax_mag.set_title('Signal Magnitudes', color='#58A6FF', fontsize=10, pad=6)
    ax_mag.set_ylabel('Magnitude', color='#8B949E', fontsize=8)
    ax_mag.grid(alpha=0.15, color='#30363D')

    # Prediction history
    ax_pred_hist.set_title('Prediction Timeline', color='#58A6FF', fontsize=10, pad=6)
    ax_pred_hist.set_ylabel('Activity', color='#8B949E', fontsize=8)
    ax_pred_hist.grid(alpha=0.15, color='#30363D', axis='x')

    # Confidence bar
    ax_conf_bar.set_title('Class Probabilities (%)', color='#58A6FF', fontsize=10, pad=6)
    ax_conf_bar.set_facecolor('#161B22')
    ax_conf_bar.grid(alpha=0.15, color='#30363D', axis='x')

    # Info panel
    ax_info.axis('off')

    return fig, ax_acc, ax_gyro, ax_mag, ax_pred_hist, ax_conf_bar, ax_info


# =========================================================
# MAIN ANIMATION UPDATE
# =========================================================
def run_simulation():
    model, scaler, le = load_assets()
    df_p1 = load_person_data()
    sim = RealTimeSimulation(df_p1, model, scaler, le)

    fig, ax_acc, ax_gyro, ax_mag, ax_pred_hist, ax_conf_bar, ax_info = setup_figure(sim)

    # Pre-create line objects
    line_ax, = ax_acc.plot([], [], color='#FF6B6B', lw=1.2, label='acc_x')
    line_ay, = ax_acc.plot([], [], color='#4ECDC4', lw=1.2, label='acc_y')
    line_az, = ax_acc.plot([], [], color='#FFEAA7', lw=1.2, label='acc_z')
    ax_acc.legend(loc='upper right', fontsize=7, facecolor='#161B22',
                  edgecolor='#30363D', labelcolor='white')

    line_gx, = ax_gyro.plot([], [], color='#A29BFE', lw=1.2, label='gyro_x')
    line_gy, = ax_gyro.plot([], [], color='#FD79A8', lw=1.2, label='gyro_y')
    line_gz, = ax_gyro.plot([], [], color='#55EFC4', lw=1.2, label='gyro_z')
    ax_gyro.legend(loc='upper right', fontsize=7, facecolor='#161B22',
                   edgecolor='#30363D', labelcolor='white')

    line_amag, = ax_mag.plot([], [], color='#FDCB6E', lw=1.5, label='|acc|')
    line_gmag, = ax_mag.plot([], [], color='#6C5CE7', lw=1.5, label='|gyro|')
    ax_mag.legend(loc='upper right', fontsize=7, facecolor='#161B22',
                  edgecolor='#30363D', labelcolor='white')

    # Timing state
    last_data_time = [sim.timestamps[0]]
    last_real_time = [time.time()]
    done = [False]

    def update(frame):
        if done[0]:
            return

        now = time.time()
        elapsed_real = now - last_real_time[0]

        # Advance through data based on real time elapsed
        rows_ingested = 0
        while sim.data_idx < len(df_p1):
            next_ts = sim.timestamps[sim.data_idx]
            elapsed_data_ns = next_ts - last_data_time[0]
            elapsed_data_s = elapsed_data_ns / 1e9

            if elapsed_data_s <= elapsed_real:
                if not sim.ingest_row():
                    done[0] = True
                    break
                rows_ingested += 1
                if rows_ingested >= 50:  # Safety cap per frame
                    break
            else:
                break

        if rows_ingested > 0:
            last_data_time[0] = sim.timestamps[min(sim.data_idx, len(df_p1)-1)]
            last_real_time[0] = now

        # === Update Accelerometer ===
        if len(sim.buf_acc_x) > 1:
            t = np.arange(len(sim.buf_acc_x))
            ax_acc.set_xlim(0, BUFFER_DISPLAY)
            line_ax.set_data(t, list(sim.buf_acc_x))
            line_ay.set_data(t, list(sim.buf_acc_y))
            line_az.set_data(t, list(sim.buf_acc_z))
            all_acc = list(sim.buf_acc_x) + list(sim.buf_acc_y) + list(sim.buf_acc_z)
            mn, mx = min(all_acc), max(all_acc)
            pad = max(0.5, (mx - mn) * 0.15)
            ax_acc.set_ylim(mn - pad, mx + pad)

        # === Update Gyroscope ===
        if len(sim.buf_gyro_x) > 1:
            t = np.arange(len(sim.buf_gyro_x))
            ax_gyro.set_xlim(0, BUFFER_DISPLAY)
            line_gx.set_data(t, list(sim.buf_gyro_x))
            line_gy.set_data(t, list(sim.buf_gyro_y))
            line_gz.set_data(t, list(sim.buf_gyro_z))
            all_gyro = list(sim.buf_gyro_x) + list(sim.buf_gyro_y) + list(sim.buf_gyro_z)
            mn, mx = min(all_gyro), max(all_gyro)
            pad = max(0.3, (mx - mn) * 0.15)
            ax_gyro.set_ylim(mn - pad, mx + pad)

        # === Update Magnitude ===
        if len(sim.buf_acc_mag) > 1:
            t = np.arange(len(sim.buf_acc_mag))
            ax_mag.set_xlim(0, BUFFER_DISPLAY)
            line_amag.set_data(t, list(sim.buf_acc_mag))
            line_gmag.set_data(t, list(sim.buf_gyro_mag))
            all_mags = list(sim.buf_acc_mag) + list(sim.buf_gyro_mag)
            mn, mx = min(all_mags), max(all_mags)
            pad = max(0.3, (mx - mn) * 0.15)
            ax_mag.set_ylim(mn - pad, mx + pad)

        # === Update Confidence Bar Chart ===
        ax_conf_bar.cla()
        ax_conf_bar.set_facecolor('#161B22')
        ax_conf_bar.set_title('Class Probabilities (%)', color='#58A6FF', fontsize=10, pad=6)
        ax_conf_bar.tick_params(colors='#8B949E', labelsize=8)
        for spine in ax_conf_bar.spines.values():
            spine.set_edgecolor('#30363D')

        probs_pct = sim.current_probs * 100
        classes = sim.classes
        colors_bar = [ACTIVITY_COLORS.get(c, DEFAULT_COLOR) for c in classes]
        bars = ax_conf_bar.barh(classes, probs_pct, color=colors_bar, edgecolor='#30363D', height=0.6)

        # Highlight predicted class
        pred_idx = np.argmax(sim.current_probs)
        bars[pred_idx].set_edgecolor('white')
        bars[pred_idx].set_linewidth(2.0)

        for i, (bar, p) in enumerate(zip(bars, probs_pct)):
            ax_conf_bar.text(
                min(p + 1.0, 99), bar.get_y() + bar.get_height()/2,
                f'{p:.1f}%',
                va='center', ha='left', fontsize=8,
                color='white', fontweight='bold' if i == pred_idx else 'normal'
            )

        ax_conf_bar.set_xlim(0, 105)
        ax_conf_bar.set_xlabel('Probability (%)', color='#8B949E', fontsize=8)
        ax_conf_bar.grid(alpha=0.15, color='#30363D', axis='x')
        ax_conf_bar.tick_params(colors='#8B949E', labelsize=8)

        # === Update Prediction Timeline ===
        ax_pred_hist.cla()
        ax_pred_hist.set_facecolor('#161B22')
        ax_pred_hist.set_title('Prediction Timeline', color='#58A6FF', fontsize=10, pad=6)
        ax_pred_hist.set_ylabel('Activity', color='#8B949E', fontsize=8)
        ax_pred_hist.tick_params(colors='#8B949E', labelsize=7)
        for spine in ax_pred_hist.spines.values():
            spine.set_edgecolor('#30363D')

        if len(sim.prediction_history) > 1:
            preds = list(sim.prediction_history)
            trues = list(sim.true_history)
            unique_acts = sorted(list(set(list(sim.classes))))
            act_to_y = {a: i for i, a in enumerate(unique_acts)}

            pred_y = [act_to_y.get(p, 0) for p in preds]
            true_y = [act_to_y.get(t, 0) for t in trues]
            x_vals = np.arange(len(preds))

            ax_pred_hist.scatter(
                x_vals, pred_y,
                c=[ACTIVITY_COLORS.get(p, DEFAULT_COLOR) for p in preds],
                s=18, zorder=3, label='Predicted', alpha=0.9
            )
            ax_pred_hist.plot(
                x_vals, true_y,
                color='white', lw=1.0, alpha=0.35, label='True', zorder=2
            )

            ax_pred_hist.set_yticks(range(len(unique_acts)))
            ax_pred_hist.set_yticklabels(unique_acts, fontsize=7, color='#8B949E')
            ax_pred_hist.set_xlim(0, max(len(preds), 5))
            ax_pred_hist.set_ylim(-0.5, len(unique_acts) - 0.5)
            ax_pred_hist.grid(alpha=0.12, color='#30363D', axis='x')
            ax_pred_hist.legend(loc='upper left', fontsize=7, facecolor='#161B22',
                                edgecolor='#30363D', labelcolor='white')

        # === Update Info Panel ===
        ax_info.cla()
        ax_info.axis('off')
        ax_info.set_facecolor('#161B22')

        pred_color = ACTIVITY_COLORS.get(sim.current_pred, '#FFFFFF')
        true_color = ACTIVITY_COLORS.get(sim.current_true, '#FFFFFF')
        accuracy = (sim.correct_predictions / sim.total_predictions * 100
                    if sim.total_predictions > 0 else 0)

        progress = sim.data_idx / len(df_p1) * 100
        elapsed = time.time() - sim.start_sim_time

        # Info box background
        ax_info.add_patch(plt.Rectangle(
            (0.02, 0.02), 0.96, 0.96,
            transform=ax_info.transAxes,
            facecolor='#0D1117', edgecolor='#30363D', linewidth=1.5,
            zorder=0
        ))

        info_texts = [
            ("LIVE PREDICTION", 0.88, '#58A6FF', 11, 'bold'),
            (sim.current_pred.upper(), 0.76, pred_color, 15, 'bold'),
            (f"Confidence: {sim.current_conf*100:.1f}%", 0.66, '#E6EDF3', 9, 'normal'),
            ("─" * 28, 0.60, '#30363D', 8, 'normal'),
            ("TRUE ACTIVITY", 0.53, '#8B949E', 8, 'bold'),
            (sim.current_true.upper(), 0.44, true_color, 11, 'bold'),
            ("─" * 28, 0.37, '#30363D', 8, 'normal'),
            (f"Live Accuracy: {accuracy:.1f}%", 0.30, '#3FB950', 9, 'bold'),
            (f"Samples: {sim.data_idx:,} / {len(df_p1):,}", 0.22, '#8B949E', 8, 'normal'),
            (f"Elapsed: {elapsed:.1f}s", 0.14, '#8B949E', 8, 'normal'),
            (f"Progress: {progress:.1f}%", 0.06, '#8B949E', 8, 'normal'),
        ]

        for text, y, color, size, weight in info_texts:
            ax_info.text(
                0.5, y, text,
                transform=ax_info.transAxes,
                ha='center', va='center',
                color=color, fontsize=size, fontweight=weight
            )

        # Match indicator
        if sim.current_pred != "Initializing..." and sim.current_true != "...":
            match = sim.current_pred == sim.current_true
            match_sym = "CORRECT" if match else "INCORRECT"
            match_col = '#3FB950' if match else '#F85149'
            ax_info.text(
                0.5, -0.04, match_sym,
                transform=ax_info.transAxes,
                ha='center', va='bottom',
                color=match_col, fontsize=9, fontweight='bold'
            )

    print("\n🚀 Starting Real-Time Simulation for Person_1...")
    print(f"📊 Total data points: {len(df_p1)}")
    print(f"🔧 Window: {WINDOW_SIZE} | Step: {STEP_SIZE}")
    print(f"📝 Logging: {'ENABLED' if ENABLE_LOGGING else 'DISABLED'}")
    print("─" * 50)

    ani = FuncAnimation(
        fig,
        update,
        interval=50,      # 50ms refresh = 20 FPS
        cache_frame_data=False
    )

    plt.show()

    # Cleanup
    if ENABLE_LOGGING and sim.log_file:
        sim.log_file.close()
        print("Log file closed.")

    if sim.total_predictions > 0:
        print(f"\n📊 Final Simulation Summary")
        print(f"─" * 40)
        print(f"Total Predictions : {sim.total_predictions}")
        print(f"Correct           : {sim.correct_predictions}")
        print(f"Live Accuracy     : {sim.correct_predictions/sim.total_predictions*100:.2f}%")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == '__main__':
    run_simulation()