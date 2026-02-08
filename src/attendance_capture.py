import cv2
import time
import pickle
import face_recognition
import pandas as pd
from datetime import datetime
import os
from config import CAMERA_ID, FACE_MATCH_THRESHOLD, CAPTURE_TIMES

# ---------- Load Encodings ----------
with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = list(set(data["names"]))  # unique students

# ---------- Initialize Attendance Dictionary ----------
# Presence vector for each student
attendance = {name: [0] * len(CAPTURE_TIMES) for name in known_names}

# ---------- Camera ----------
cap = cv2.VideoCapture(CAMERA_ID)
start_time = time.time()
capture_index = 0

print("🕒 Attendance capture started")

while capture_index < len(CAPTURE_TIMES):
    ret, frame = cap.read()
    if not ret:
        break

    elapsed_minutes = (time.time() - start_time) / 60

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    detected_students = set()

    for encoding in encodings:
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=FACE_MATCH_THRESHOLD
        )

        if True in matches:
            idx = matches.index(True)
            detected_students.add(data["names"][idx])

    # ---------- Capture Attendance at Interval ----------
    if elapsed_minutes >= CAPTURE_TIMES[capture_index]:
        print(f"📸 Capturing attendance at {CAPTURE_TIMES[capture_index]} minutes")

        for student in attendance:
            if student in detected_students:
                attendance[student][capture_index] = 1

        capture_index += 1

    cv2.imshow("Attendance Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================
# MODULE 7: DATA MANAGEMENT & RECORD CONSOLIDATION
# =====================================================

records = []
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
CLASS_NAME = "class_A"

for student, presence_vector in attendance.items():
    # Expected format: ID_Name
    try:
        student_id, student_name = student.split("_", 1)
    except ValueError:
        student_id = "NA"
        student_name = student

    final_status = "Present" if sum(presence_vector) >= 2 else "Absent"

    records.append({
        "Student ID": student_id,
        "Student Name": student_name,
        "Class": CLASS_NAME,
        "Presence Vector": str(presence_vector),
        "Final Attendance": final_status,
        "Timestamp": timestamp
    })

# Create DataFrame
df = pd.DataFrame(records)

# ---------- Data Validation ----------
df.drop_duplicates(subset=["Student ID"], inplace=True)
df.fillna("N/A", inplace=True)

print("✅ Attendance data consolidated")

# =====================================================
# MODULE 8: EXCEL REPORT GENERATION
# =====================================================

# Ensure output folder exists
os.makedirs("attendance", exist_ok=True)

file_name = f"attendance/Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
df.to_excel(file_name, index=False)

print(f"✅ Attendance report generated: {file_name}")
