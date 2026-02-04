"""
config.py
Central configuration file for the Face Attendance System
"""

# ---------------- Camera Settings ----------------
CAMERA_ID = 0          # Default webcam (change to 1 if external camera is used)

# ---------------- Face Recognition Settings ----------------
FACE_MATCH_THRESHOLD = 0.5
# Lower value = stricter matching
# Typical range: 0.4 – 0.6

# ---------------- Attendance Settings ----------------
# Time intervals (in minutes) during a 50-minute lecture
CAPTURE_TIMES = [10, 25, 45]

# ---------------- General Settings ----------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
