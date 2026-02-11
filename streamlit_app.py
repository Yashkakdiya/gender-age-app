import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib


from src.face_detector import get_faces
from src.gender_age_predictor import predict_gender_age


st.set_page_config(page_title="Gender & Age Detection", layout="centered")
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "user": hashlib.sha256("user123".encode()).hexdigest()
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    
if not st.session_state.logged_in:
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if username in USERS and USERS[username] == hashed:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

st.sidebar.write(f"Logged in as: {st.session_state.user}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()


st.title("👤 Gender & Age Detection System")
st.write("Upload an image to detect gender & age")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    faces = get_faces(img)

    male_count = 0
    female_count = 0

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        gender, age, g_conf, a_conf = predict_gender_age(face)

        if gender == "Male":
            male_count += 1
        else:
            female_count += 1

        label = f"{gender} ({g_conf}%), Age: {age} ({a_conf}%)"
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    st.image(img, caption="Detection Result", use_column_width=True)

    st.subheader("📊 Summary")
    st.write(f"👨 Male: {male_count}")
    st.write(f"👩 Female: {female_count}")
