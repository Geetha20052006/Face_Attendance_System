import cv2
import pickle
import face_recognition
from config import CAMERA_ID, FACE_MATCH_THRESHOLD

# ---------- Load Encodings ----------
with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("✅ Face encodings loaded")

# ---------- Camera ----------
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🎥 Real-time face recognition started")
print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), face_encoding in zip(boxes, encodings):

        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=FACE_MATCH_THRESHOLD
        )

        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        if True in matches:
            matched_idxs = [i for i, b in enumerate(matches) if b]
            counts = {}

            for i in matched_idxs:
                counts[known_names[i]] = counts.get(known_names[i], 0) + 1

            name = max(counts, key=counts.get)
            color = (0, 255, 0)  # Green for recognized

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Recognition stopped")
