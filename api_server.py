from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from PIL import Image
import sqlite3
import uuid
from datetime import datetime
import io
import os

app = FastAPI(title="AI Gender Age API")

# ==========================
# DATABASE
# ==========================
conn = sqlite3.connect("detections.db", check_same_thread=False)
c = conn.cursor()

# ==========================
# MODEL PATHS
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
FACE_MODEL = os.path.join(MODEL_DIR, "haarcascade", "haarcascade_frontalface_default.xml")

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
face_cascade = cv2.CascadeClassifier(FACE_MODEL)

AGE_LIST = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']
GENDER_LIST = ['Male', 'Female']

# ==========================
# VERIFY API KEY
# ==========================
def verify_api_key(api_key):
    c.execute("SELECT username FROM users WHERE api_key=?", (api_key,))
    return c.fetchone()

# ==========================
# MAIN API ENDPOINT
# ==========================
@app.post("/predict")
async def predict(api_key: str, file: UploadFile = File(...)):

    user = verify_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []

    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(
            face,1.0,(227,227),
            (78.42,87.76,114.89),
            swapRB=False
        )

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()[0]
        gender = GENDER_LIST[gender_preds.argmax()]
        gender_conf = float(gender_preds.max())

        age_net.setInput(blob)
        age_preds = age_net.forward()[0]
        age = AGE_LIST[age_preds.argmax()]
        age_conf = float(age_preds.max())

        results.append({
            "gender": gender,
            "gender_confidence": round(gender_conf*100,2),
            "age_group": age,
            "age_confidence": round(age_conf*100,2)
        })

        c.execute("""
        INSERT INTO detections
        (username, gender, gender_conf, age_group, age_conf, timestamp)
        VALUES (?,?,?,?,?,?)
        """, (
            user[0],
            gender,
            gender_conf,
            age,
            age_conf,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()

    return {
        "faces_detected": len(results),
        "results": results
    }
