import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile

# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="ShopLifter Detection AI",
    page_icon="🛒",
    layout="centered",
)

# =========================
# Custom CSS for Modern Look
# =========================
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #1c1c1c, #0f2027);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: rgba(0,0,0,0.4);
        padding: 20px;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #f5f5f5;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.5em 1em;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
    }
    .stFileUploader>div>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("video_finetuned_final.keras", compile=False)

model = load_model()

# =========================
# Page Title
# =========================
st.title("🛒 ShopLifter Detection AI")
st.markdown("Upload a video and let AI detect suspicious behavior instantly.")

# =========================
# Video Upload
# =========================
uploaded_video = st.file_uploader("📂 Upload your video file", type=["mp4", "avi", "mov"])

# =========================
# Video Processing
# =========================
IMG_HEIGHT = 224
IMG_WIDTH = 224
MAX_FRAMES = 20
label_map = {0: "Non-ShopLifter", 1: "ShopLifter"}

def load_video(path, max_frames=MAX_FRAMES, resize=(IMG_WIDTH, IMG_HEIGHT)):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        frames.append(frame)
        if len(frames) == max_frames:
            break
    cap.release()
    while len(frames) < max_frames:
        frames.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)))
    return np.array(frames)

# =========================
# Prediction
# =========================
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.video(tfile.name)

    if st.button("🔍 Run Detection"):
        video_data = load_video(tfile.name)
        video_data = np.expand_dims(video_data, axis=0)
        prediction = model.predict(video_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class] * 100

        st.markdown(f"""
        ### 🎯 Detection Result:
        - **Category:** {label_map[predicted_class]}
        - **Confidence:** {confidence:.2f}%
        """)

        if predicted_class == 1:
            st.error("⚠ Suspicious activity detected!")
        else:
            st.success("✅ No suspicious activity detected.")
