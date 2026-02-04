import cv2
import os

# ---------- Load Face Detector ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- Student Details ----------
STUDENT_ID = input("Enter Student ID: ")
STUDENT_NAME = input("Enter Student Name: ")
CLASS_NAME = "class_A"

# ---------- Dataset Path ----------
BASE_DIR = "datasets"
STUDENT_DIR = os.path.join(BASE_DIR, CLASS_NAME, f"{STUDENT_ID}_{STUDENT_NAME}")
os.makedirs(STUDENT_DIR, exist_ok=True)

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("📸 Capturing FACE images only")
print("Press 'c' to capture | 'q' to quit")

img_count = 0
MAX_IMAGES = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and len(faces) == 1:
        x, y, w, h = faces[0]
        face_img = frame[y:y + h, x:x + w]

        img_count += 1
        img_path = os.path.join(STUDENT_DIR, f"face_{img_count}.jpg")
        cv2.imwrite(img_path, face_img)

        print(f"✅ Face image {img_count} saved")

    if key == ord('q') or img_count >= MAX_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print("🎉 Face enrollment completed successfully")
