import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Deepfake Cyber Intelligence", layout="wide")

# ---------------- ULTRA PRO CSS ----------------
st.markdown("""
<style>

/* 🌌 Background (soft gradient, not pure black) */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

/* 🧠 Title (bigger + glow + gradient) */
.title {
    text-align: center;
    font-size: 80px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 0px 25px rgba(56,189,248,0.4);
    margin-bottom: 20px;
}

/* 🪟 Glass Cards (clean + premium) */
.card {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 20px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(56,189,248,0.15);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
    margin-bottom: 15px;
    transition: 0.3s ease;
}

/* ✨ Hover Glow Effect */
.card:hover {
    border: 1px solid #38bdf8;
    box-shadow: 0px 0px 25px rgba(56,189,248,0.4);
}

/* 🔘 Buttons (clean neon gradient) */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0px 0px 15px #38bdf8;
    transform: scale(1.05);
}

/* 📊 Metrics Text */
.css-1xarl3l, .css-1v0mbdj {
    color: #e2e8f0 !important;
    font-size: 18px;
}

/* 🧾 Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(56,189,248,0.2);
}

/* 📁 Table */
.dataframe {
    background-color: rgba(255,255,255,0.02);
    border-radius: 10px;
}

/* 🟢 Status Glow */
.status {
    color: #22c55e;
    font-weight: bold;
    text-shadow: 0px 0px 10px #22c55e;
}

/* 🔴 Fake Alert */
.fake-box {
    background: rgba(255,0,0,0.1);
    padding: 12px;
    border-radius: 10px;
    color: #f87171;
}

/* 🟢 Real Alert */
.real-box {
    background: rgba(0,255,100,0.1);
    padding: 12px;
    border-radius: 10px;
    color: #4ade80;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🛡️ Deepfake Cyber Intelligence System</p>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_tf():
    return load_model("deepfake_model_v2.h5")

model = load_model_tf()
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- PREDICT ----------------
def predict(img):
    pred = model.predict(img, verbose=0)[0][0]
    return ("FAKE", pred) if pred > 0.5 else ("REAL", 1 - pred)

# ---------------- SIDEBAR ----------------
mode = st.sidebar.radio("Select Mode", ["Image", "Video", "Webcam"])

# ---------------- STATUS BAR ----------------
c1, c2, c3 = st.columns(3)
c1.markdown("<div class='card'>⚙️ Model: CNN (H5)</div>", unsafe_allow_html=True)
c2.markdown("<div class='card'>🟢 System Active</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'>📡 Mode: {mode}</div>", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([2,1])

# ================= LEFT PANEL =================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Detection Panel")

    # IMAGE
    if mode == "Image":
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file)
            img = np.array(image)
            st.image(image)

            if st.button("🔍 Scan"):
                faces = face_cascade.detectMultiScale(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),1.3,5)

                for (x,y,w,h) in faces:
                    face = img[y:y+h,x:x+w]
                    inp = preprocess(face)
                    label, conf = predict(inp)

                    st.session_state.history.append({
                        "Type":"Image",
                        "Result":label,
                        "Confidence":round(conf,2)
                    })

                    color = (0,0,255) if label=="FAKE" else (0,255,0)
                    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

                st.image(img, channels="BGR")

    # VIDEO (FIXED VERSION)
    elif mode == "Video":
        file = st.file_uploader("Upload Video", type=["mp4"])
        if file:
            path = "temp.mp4"
            with open(path,"wb") as f:
                f.write(file.read())

            st.video(path)

            if st.button("🔍 Scan Video"):
                cap = cv2.VideoCapture(path)

                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 30)

                preds = []
                frame_id = 0

                progress = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_id % step == 0:
                        faces = face_cascade.detectMultiScale(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),1.3,5)

                        for (x,y,w,h) in faces:
                            face = frame[y:y+h,x:x+w]
                            inp = preprocess(face)
                            _, conf = predict(inp)
                            preds.append(conf)

                    frame_id += 1
                    progress.progress(min(frame_id/total,1.0))

                cap.release()
                progress.empty()

                if len(preds) > 0:
                    avg = sum(preds)/len(preds)
                    label = "FAKE" if avg > 0.5 else "REAL"

                    st.session_state.history.append({
                        "Type":"Video",
                        "Result":label,
                        "Confidence":round(avg,2)
                    })

                    st.success(f"{label} ({avg:.2f})")
                else:
                    st.warning("No face detected")

            os.remove(path)

    # WEBCAM
    elif mode == "Webcam":
        run = st.checkbox("Start Camera")
        frame_window = st.image([])
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),1.3,5)

            for (x,y,w,h) in faces:
                face = frame[y:y+h,x:x+w]
                inp = preprocess(face)
                label, conf = predict(inp)

                st.session_state.history.append({
                    "Type":"Webcam",
                    "Result":label,
                    "Confidence":round(conf,2)
                })

                color = (0,0,255) if label=="FAKE" else (0,255,0)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            frame_window.image(frame, channels="BGR")

        cap.release()

    st.markdown("</div>", unsafe_allow_html=True)

# ================= RIGHT PANEL =================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Analytics")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        total = len(df)
        fake = len(df[df["Result"]=="FAKE"])
        real = len(df[df["Result"]=="REAL"])

        st.metric("Total", total)
        st.metric("Fake", fake)
        st.metric("Real", real)

        fig, ax = plt.subplots()
        ax.pie([fake, real], labels=["Fake","Real"], autopct='%1.1f%%')
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # LOGS
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Logs")

    for item in st.session_state.history[-5:][::-1]:
        st.write(f"{item['Type']} → {item['Result']} ({item['Confidence']})")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= HISTORY =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📁 Detection History")

if len(st.session_state.history) > 0:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)