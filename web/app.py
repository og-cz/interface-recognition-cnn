import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Connector Classifier",
    page_icon="🔌",
    layout="centered"
)

st.title("🔌 Network Connector Classifier")
st.write("Identify connector types: **SATA, VGA, USB-A, HDMI, or RJ45**")

# ── Load model ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn_classifier.h5"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Place cnn_classifier.h5 in the same folder as app.py."
        )
    # compile=False avoids training-state dependency issues in deployment environments.
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
except Exception as exc:
    st.error("Failed to load model. Check that cnn_classifier.h5 is present and compatible.")
    st.exception(exc)
    st.stop()

# Must match train_data.class_indices output:
# {'hdmi': 0, 'rj45': 1, 'sata': 2, 'usba': 3, 'vga': 4}
CLASS_LABELS = ['hdmi', 'rj45', 'sata', 'usba', 'vga']

def preprocess_image(img):
    # Same preprocessing as your Colab training pipeline
    img = img.convert("RGB")          # ensure 3 channels
    img = img.resize((128, 128))      # same target_size as training
    img_array = np.array(img)
    img_array = img_array / 255.0     # same rescale=1./255 as training
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# ── Upload ───────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a connector image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
    except Exception:
        st.error("Could not read that image. Please upload a valid JPG or PNG file.")
        st.stop()

    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess
    img_array = preprocess_image(img)

    # Predict
    with st.spinner("Classifying..."):
        prediction = model.predict(img_array, verbose=0)[0]

    predicted_idx = np.argmax(prediction)
    predicted_label = CLASS_LABELS[predicted_idx]
    confidence = float(prediction[predicted_idx]) * 100

    # ── Results ──────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Connector", predicted_label.upper())
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    # Confidence breakdown for all classes
    st.markdown("#### Confidence per class")
    for label, score in zip(CLASS_LABELS, prediction):
        st.progress(
            float(score),
            text=f"{label.upper()}: {score * 100:.1f}%"
        )

    # Low confidence warning
    if confidence < 60:
        st.warning(
            "⚠️ Low confidence — the model is unsure. "
            "This is expected with limited training data, especially HDMI (only 18 images)."
        )
else:
    st.info("Upload a JPG or PNG image to classify a connector.")