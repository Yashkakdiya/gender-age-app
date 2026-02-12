import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import hashlib
import pandas as pd
import sqlite3
from datetime import datetime
import uuid
import plotly.express as px

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="AI Gender & Age Analytics", layout="wide")

# =============================
# DATABASE CONNECTION
# =============================
conn = sqlite3.connect("detections.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    gender TEXT,
    gender_conf REAL,
    age_group TEXT,
    age_conf REAL,
    timestamp TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT,
    api_key TEXT
)
""")
conn.commit()

# =============================
# DEFAULT USERS
# =============================
def create_default_users():
    users = [
        ("admin", hashlib.sha256("admin123".encode()).hexdigest(), "admin"),
        ("user", hashlib.sha256("user123".encode()).hexdigest(), "user")
    ]

    for u in users:
        c.execute("SELECT * FROM users WHERE username=?", (u[0],))
        if not c.fetchone():
            c.execute("INSERT INTO users VALUES (?,?,?,?)",
                      (u[0], u[1], u[2], str(uuid.uuid4())))
    conn.commit()

create_default_users()

# =============================
# LOGIN SYSTEM
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.session_state.role = ""

if not st.session_state.logged_in:
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed = hashlib.sha256(password.encode()).hexdigest()
        c.execute("SELECT role FROM users WHERE username=? AND password=?",
                  (username, hashed))
        result = c.fetchone()

        if result:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.role = result[0]
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# =============================
# SIDEBAR
# =============================
st.sidebar.success(f"👤 {st.session_state.user} ({st.session_state.role})")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.session_state.role = ""
    st.rerun()

menu = st.sidebar.selectbox("Menu", ["Detect", "History", "API Key"])

# =============================
# MODEL PATHS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
FACE_MODEL = os.path.join(MODEL_DIR, "haarcascade", "haarcascade_frontalface_default.xml")

# =============================
# LOAD MODELS SAFELY
# =============================
@st.cache_resource
def load_models():
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    face_cascade = cv2.CascadeClassifier(FACE_MODEL)
    return age_net, gender_net, face_cascade

age_net, gender_net, face_cascade = load_models()

AGE_LIST = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']
GENDER_LIST = ['Male', 'Female']

# =============================
# DETECT PAGE
# =============================
if menu == "Detect":

    st.title("👤 AI Gender & Age Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:

        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
        except:
            st.error("Invalid image file")
            st.stop()

        img = np.array(image)

        # Resize large images for stability
        if img.shape[1] > 1000:
            ratio = 1000 / img.shape[1]
            img = cv2.resize(img, (1000, int(img.shape[0]*ratio)))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            st.error("No face detected")
        else:
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
                gender_conf = round(float(gender_preds.max())*100,2)

                age_net.setInput(blob)
                age_preds = age_net.forward()[0]
                age = AGE_LIST[age_preds.argmax()]
                age_conf = round(float(age_preds.max())*100,2)

                # Insert into database
                c.execute("""
                INSERT INTO detections
                (username, gender, gender_conf, age_group, age_conf, timestamp)
                VALUES (?,?,?,?,?,?)
                """, (
                    st.session_state.user,
                    gender,
                    gender_conf,
                    age,
                    age_conf,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img,f"{gender}, {age}",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,255,0),2)

            st.image(img, width="stretch")

# =============================
# HISTORY PAGE
# =============================
elif menu == "History":

    st.title("📊 Detection History")

    if st.session_state.role == "admin":
        df = pd.read_sql_query("SELECT * FROM detections", conn)
    else:
        df = pd.read_sql_query(
            "SELECT * FROM detections WHERE username=?",
            conn, params=(st.session_state.user,)
        )

    if df.empty:
        st.info("No records found.")
    else:
        st.dataframe(df, width="stretch")

        fig = px.pie(df, names="age_group", title="Age Distribution")
        st.plotly_chart(fig, width="stretch")

        if st.session_state.role == "admin":
            if st.button("Delete All Records"):
                c.execute("DELETE FROM detections")
                conn.commit()
                st.success("All records deleted")
                st.rerun()

# =============================
# API KEY PAGE
# =============================
elif menu == "API Key":

    st.title("🔑 API Key")

    c.execute("SELECT api_key FROM users WHERE username=?",
              (st.session_state.user,))
    api_key = c.fetchone()[0]

    st.code(api_key)

    if st.button("Regenerate API Key"):
        new_key = str(uuid.uuid4())
        c.execute("UPDATE users SET api_key=? WHERE username=?",
                  (new_key, st.session_state.user))
        conn.commit()
        st.success("API Key regenerated")
        st.rerun()
