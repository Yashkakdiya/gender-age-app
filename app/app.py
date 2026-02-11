import csv
import os
import cv2
from datetime import datetime

from src.face_detector import get_faces
from src.gender_age_predictor import predict_gender_age

# =================================================
# PROJECT ROOT SETUP
# =================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CAPTURE_DIR = os.path.join(BASE_DIR, "captured_images")
os.makedirs(CAPTURE_DIR, exist_ok=True)

REPORT_FILE = os.path.join(BASE_DIR, "report.csv")

# =================================================
# CSV INIT
# =================================================
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
# LOAD MODELS (SAFE)
# =================================================
age_net = None
gender_net = None

try:
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    print("✅ Models loaded")
except Exception as e:
    print("⚠️ Model load failed, fallback active")
    print(e)

# =================================================
# WEBCAM (macOS SAFE)
# =================================================
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit(1)

# =================================================
# COUNTERS
# =================================================
male_count = 0
female_count = 0
last_gender = "NA"
last_age = "NA"

print("🔥 Running Gender & Age Detector")
print("👉 Press 'c' to capture")
print("👉 Press 'q' to quit")

# =================================================
# MAIN LOOP
# =================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = get_faces(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        gender, age = predict_gender_age(face, gender_net, age_net)
        last_gender, last_age = gender, age

        if gender == "Male":
            male_count += 1
        elif gender == "Female":
            female_count += 1

        label = f"{gender}, Age: {age}"
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Counter display
    cv2.putText(frame,
                f"Male: {male_count}  Female: {female_count}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,0,0),
                2)

    cv2.imshow("Gender & Age Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c') and last_gender != "NA":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{ts}_{last_gender}_{last_age}.jpg"
        path = os.path.join(CAPTURE_DIR, filename)

        cv2.imwrite(path, frame)

        with open(REPORT_FILE, "a", newline="") as f:
            csv.writer(f).writerow([ts, last_gender, last_age, filename])

        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
