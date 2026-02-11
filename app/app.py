import csv
import os
import cv2
from datetime import datetime

from src.face_detector import get_faces
from src.gender_age_predictor import predict_gender_age

# =================================================
# ABSOLUTE PATH SETUP (PROJECT ROOT SAFE)
# =================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Folder to save captured images
CAPTURE_DIR = os.path.join(BASE_DIR, "captured_images")
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Report CSV
REPORT_FILE = os.path.join(BASE_DIR, "report.csv")

if not os.path.exists(REPORT_FILE):
    with open(REPORT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Gender", "Age", "Image"])

# =================================================
# MODEL PATHS
# =================================================
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")

GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

# =================================================
# LOAD AGE & GENDER MODELS (SAFE + FALLBACK)
# =================================================
age_net = None
gender_net = None

try:
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    print("✅ Age & Gender models loaded")
except Exception as e:
    print("⚠️ Using fallback age/gender logic")
    print("Reason:", e)

# =================================================
# WEBCAM OPEN (macOS FIX)
# =================================================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("Webcam opened:", cap.isOpened())

if not cap.isOpened():
    print("❌ Webcam could not be opened")
    exit(1)

# =================================================
# COUNTERS (ADVANCED FEATURE)
# =================================================
male_count = 0
female_count = 0

# =================================================
# MAIN LOOP
# =================================================
print("🔥 Gender & Age Detector running")
print("👉 Press 'c' to CAPTURE photo")
print("👉 Press 'q' to QUIT")

last_gender = "NA"
last_age = "NA"

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not read from webcam")
        break

    faces = get_faces(frame)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue

        gender, age = predict_gender_age(face_img, gender_net, age_net)
        last_gender = gender
        last_age = age

        if gender == "Male":
            male_count += 1
        elif gender == "Female":
            female_count += 1

        label = f"{gender}, Age: {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show counters on screen
    counter_text = f"Male: {male_count}  Female: {female_count}"
    cv2.putText(
        frame,
        counter_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("Gender & Age Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Capture image + CSV entry
    if key == ord('c') and last_gender != "NA":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{last_gender}_{last_age}.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)

        cv2.imwrite(filepath, frame)

        with open(REPORT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, last_gender, last_age, filename])

        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
