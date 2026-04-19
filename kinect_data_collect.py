import sys
import pandas as pd
from datetime import datetime
import pygame
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# ----------------------------
# Joint name mapping
# ----------------------------
JOINT_NAMES = {
    PyKinectV2.JointType_SpineBase: "SpineBase",
    PyKinectV2.JointType_SpineMid: "SpineMid",
    PyKinectV2.JointType_Neck: "Neck",
    PyKinectV2.JointType_Head: "Head",
    PyKinectV2.JointType_ShoulderLeft: "ShoulderLeft",
    PyKinectV2.JointType_ElbowLeft: "ElbowLeft",
    PyKinectV2.JointType_WristLeft: "WristLeft",
    PyKinectV2.JointType_HandLeft: "HandLeft",
    PyKinectV2.JointType_ShoulderRight: "ShoulderRight",
    PyKinectV2.JointType_ElbowRight: "ElbowRight",
    PyKinectV2.JointType_WristRight: "WristRight",
    PyKinectV2.JointType_HandRight: "HandRight",
    PyKinectV2.JointType_HipLeft: "HipLeft",
    PyKinectV2.JointType_KneeLeft: "KneeLeft",
    PyKinectV2.JointType_AnkleLeft: "AnkleLeft",
    PyKinectV2.JointType_FootLeft: "FootLeft",
    PyKinectV2.JointType_HipRight: "HipRight",
    PyKinectV2.JointType_KneeRight: "KneeRight",
    PyKinectV2.JointType_AnkleRight: "AnkleRight",
    PyKinectV2.JointType_FootRight: "FootRight",
    PyKinectV2.JointType_SpineShoulder: "SpineShoulder",
    PyKinectV2.JointType_HandTipLeft: "HandTipLeft",
    PyKinectV2.JointType_ThumbLeft: "ThumbLeft",
    PyKinectV2.JointType_HandTipRight: "HandTipRight",
    PyKinectV2.JointType_ThumbRight: "ThumbRight"
}

# Initialize Kinect
kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Color
)

# Pygame setup
pygame.init()
screen_width, screen_height = 960, 540
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Kinect Skeleton Viewer")

# Colors
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
DARK_GREEN = (0, 150, 0)
DARK_RED = (150, 0, 0)

font = pygame.font.SysFont("Arial", 24)

# Skeleton edges
SKELETON_EDGES = [
    (PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
    (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
    (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
    (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
    (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),
    (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
    (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
    (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
    (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
    (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
    (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
    (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),
    (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
    (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
    (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
    (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
    (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
    (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight),
]

# Data recording
records = []
recording = False
start_time = 0
elapsed_time = 0

print("Press 'Start' button to begin recording, 'Stop' to end.")

running = True
clock = pygame.time.Clock()

try:
    while running:
        # Handle events FIRST
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Button rectangles (must match drawing positions below)
                start_button = pygame.Rect(20, 20, 100, 40)
                stop_button = pygame.Rect(140, 20, 100, 40)
                
                if start_button.collidepoint(event.pos) and not recording:
                    recording = True
                    start_time = time.time()
                    elapsed_time = 0
                    records = []  # clear old data
                    print("Recording started.")
                elif stop_button.collidepoint(event.pos) and recording:
                    recording = False
                    print("Recording stopped.")
                    if records:  # ask where to save immediately
                        # use tkinter for Save As dialog
                        root = tk.Tk()
                        root.withdraw()  # hide main tkinter window
                        file_path = filedialog.asksaveasfilename(
                            defaultextension=".csv",
                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                            title="Save skeleton data as..."
                        )
                        if file_path:
                            df = pd.DataFrame(records)
                            df.to_csv(file_path, index=False)
                            print(f"Data saved to {file_path}")
                        else:
                            print("Save cancelled.")
                    records = []
                    elapsed_time = 0

        # Update stopwatch
        if recording:
            elapsed_time = time.time() - start_time

        # Draw color frame as background
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            if frame is not None:
                frame = frame.reshape((1080, 1920, 4))
                frame = np.rot90(frame)
                frame = pygame.surfarray.make_surface(frame[:, :, :3])
                frame = pygame.transform.scale(frame, (screen_width, screen_height))
                frame = pygame.transform.flip(frame, True, False)
                screen.blit(frame, (0, 0))

        # Draw skeleton + record data
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()
            if bodies is not None:
                for i in range(0, kinect.max_body_count):
                    body = bodies.bodies[i]
                    if not body.is_tracked:
                        continue

                    joints = body.joints
                    joint_points = kinect.body_joints_to_color_space(joints)

                    if recording:
                        timestamp = time.time() - start_time
                        row = {"timestamp": timestamp, "body_id": i}
                        for j in range(PyKinectV2.JointType_Count):
                            joint_name = JOINT_NAMES[j]
                            pos = joints[j].Position
                            row[f"{joint_name}_x"] = pos.x
                            row[f"{joint_name}_y"] = pos.y
                            row[f"{joint_name}_z"] = pos.z
                        records.append(row)

                    # Draw bones
                    for joint1, joint2 in SKELETON_EDGES:
                        j1 = joint_points[joint1]
                        j2 = joint_points[joint2]

                        if (np.isfinite(j1.x) and np.isfinite(j1.y) and
                            np.isfinite(j2.x) and np.isfinite(j2.y)):
                            x1, y1 = int(j1.x * screen_width / 1920), int(j1.y * screen_height / 1080)
                            x2, y2 = int(j2.x * screen_width / 1920), int(j2.y * screen_height / 1080)

                            if (0 <= x1 < screen_width and 0 <= y1 < screen_height and
                                0 <= x2 < screen_width and 0 <= y2 < screen_height):
                                pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2), 3)

        # ============================================
        # DRAW UI ELEMENTS ON TOP (AFTER EVERYTHING)
        # ============================================
        
        # Define button positions
        start_button = pygame.Rect(20, 20, 100, 40)
        stop_button = pygame.Rect(140, 20, 100, 40)

        # Draw Start button
        if recording:
            pygame.draw.rect(screen, GRAY, start_button)
            pygame.draw.rect(screen, BLACK, start_button, 2)  # border
        else:
            pygame.draw.rect(screen, DARK_GREEN, start_button)
            pygame.draw.rect(screen, GREEN, start_button, 3)  # border
        
        start_text = font.render("Start", True, WHITE)
        screen.blit(start_text, (start_button.x + 20, start_button.y + 8))

        # Draw Stop button
        if recording:
            pygame.draw.rect(screen, DARK_RED, stop_button)
            pygame.draw.rect(screen, RED, stop_button, 3)  # border
        else:
            pygame.draw.rect(screen, GRAY, stop_button)
            pygame.draw.rect(screen, BLACK, stop_button, 2)  # border
        
        stop_text = font.render("Stop", True, WHITE)
        screen.blit(stop_text, (stop_button.x + 25, stop_button.y + 8))

        # Draw timer background for better visibility
        timer_bg = pygame.Rect(260, 20, 200, 40)
        pygame.draw.rect(screen, (0, 0, 0, 180), timer_bg)
        pygame.draw.rect(screen, WHITE if recording else GRAY, timer_bg, 2)
        
        # Draw timer text
        timer_color = RED if recording else WHITE
        timer_text = font.render(f"Time: {elapsed_time:.1f}s", True, timer_color)
        screen.blit(timer_text, (270, 25))

        # Optional: Recording indicator
        if recording:
            # Blinking red dot
            if int(elapsed_time * 2) % 2 == 0:  # blinks every 0.5s
                pygame.draw.circle(screen, RED, (490, 40), 8)

        pygame.display.update()
        clock.tick(30)  # 30 FPS

except KeyboardInterrupt:
    print("Stopping recording...")

finally:
    pygame.quit()
    kinect.close()
    print("Kinect closed.")