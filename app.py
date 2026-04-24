import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------- LOGIN ----------
def login():
    st.title("🔐 Smart Waste Classifier Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/model.h5")

model = load_model()

# ---------- CLASS NAMES ----------
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# ---------- MAIN APP ----------
def main():
    st.title("♻️ Smart Waste Classifier (AI Model)")

    file = st.file_uploader("Upload Waste Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # preprocess (IMPORTANT FIX)
        img = img.resize((224, 224))
        arr = np.array(img)
        arr = np.expand_dims(arr, axis=0)

        # prediction
        pred = model.predict(arr, verbose=0)[0]

        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        result = class_names[idx]

        # results
        st.subheader("Prediction Results")

        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {pred[i]:.2f}")

        st.success(f"Detected Waste: {result}")
        st.write("Confidence:", round(confidence, 2))
        st.progress(int(confidence * 100))

        # suggestions
        tips = {
            "plastic": "♻️ Recycle plastic properly",
            "metal": "🔩 Send metal for recycling",
            "paper": "📄 Recycle paper waste",
            "glass": "🍾 Handle glass carefully",
            "cardboard": "📦 Reuse cardboard boxes",
            "trash": "🗑️ General waste"
        }

        st.info(tips.get(result, "Follow proper waste segregation"))

        st.balloons()

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    main()