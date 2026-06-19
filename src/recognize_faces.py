import cv2
import pickle
import face_recognition
import numpy as np
import dlib
from scipy.spatial import distance
from collections import defaultdict
from config import CAMERA_ID

# ---------- Parameters ----------
DISTANCE_THRESHOLD = 0.50
FRAME_CONFIRMATION = 5
EAR_THRESHOLD = 0.21
BLINK_FRAMES = 2

# ---------- Load Encodings ----------
with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# ---------- Load Landmark Predictor ----------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------- Camera ----------
cap = cv2.VideoCapture(CAMERA_ID)

name_counter = defaultdict(int)
blink_counter = 0
blink_detected = False

print("🎥 Final Recognition System Started")
print("Blink once to activate recognition")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # ---------- Blink Detection ----------
    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES:
                blink_detected = True
                print("👁 Liveness Confirmed")
            blink_counter = 0

    # ---------- Recognition ----------
    for (top, right, bottom, left), face_encoding in zip(boxes, encodings):

        name = "Blink Required"
        color = (0, 0, 255)

        if blink_detected:
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]

                if best_distance < DISTANCE_THRESHOLD:
                    candidate_name = known_names[best_match_index]
                    name_counter[candidate_name] += 1

                    if name_counter[candidate_name] >= FRAME_CONFIRMATION:
                        name = candidate_name
                        color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Final Recognition + Blink Liveness", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 System stopped")