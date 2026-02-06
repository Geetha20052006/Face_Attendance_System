import cv2
import time
import pickle
import face_recognition
import pandas as pd
from config import CAMERA_ID, FACE_MATCH_THRESHOLD, CAPTURE_TIMES

# ---------- Load Encodings ----------
with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = list(set(data["names"]))

# ---------- Initialize Attendance Dictionary ----------
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
            data["encodings"], encoding, tolerance=FACE_MATCH_THRESHOLD
        )

        if True in matches:
            idx = matches.index(True)
            detected_students.add(data["names"][idx])

    # ---------- Capture at Interval ----------
    if elapsed_minutes >= CAPTURE_TIMES[capture_index]:
        print(f"📸 Capturing attendance at {CAPTURE_TIMES[capture_index]} min")

        for student in attendance:
            if student in detected_students:
                attendance[student][capture_index] = 1

        capture_index += 1

    cv2.imshow("Attendance Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- Majority Voting ----------
final_attendance = {}

for student, records in attendance.items():
    final_attendance[student] = "Present" if sum(records) >= 2 else "Absent"

# ---------- Save to Excel ----------
df = pd.DataFrame([
    {"Student": s, "Attendance": a}
    for s, a in final_attendance.items()
])

df.to_excel("attendance/final_attendance.xlsx", index=False)

print("✅ Attendance captured and saved successfully")
