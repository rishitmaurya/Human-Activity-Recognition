from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import time

# Initialize Kinect (body tracking only)
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)

print("Waiting for body frames... Move in front of Kinect.")

try:
    while True:
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()
            if bodies is not None:
                print("Body frame detected!")
                time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopped.")
