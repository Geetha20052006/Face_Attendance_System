import os
import pickle
import cv2
import face_recognition

DATASET_DIR = "datasets"
ENCODING_DIR = "encodings"
ENCODING_FILE = os.path.join(ENCODING_DIR, "face_encodings.pkl")

os.makedirs(ENCODING_DIR, exist_ok=True)

known_encodings = []
known_names = []

print("🔄 Encoding faces...")

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for student_folder in os.listdir(class_path):
        student_path = os.path.join(class_path, student_folder)

        if not os.path.isdir(student_path):
            continue

        student_name = student_folder

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(student_name)

print("💾 Saving encodings...")
data = {"encodings": known_encodings, "names": known_names}

with open(ENCODING_FILE, "wb") as f:
    pickle.dump(data, f)

print("✅ Face encodings generated and saved successfully")
