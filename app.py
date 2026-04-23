import os
import gdown
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from io import BytesIO
from tensorflow.keras.applications.efficientnet import preprocess_input
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Brain MRI AI Assistant",
    page_icon="🧠",
    layout="centered"
)

# ======================
# SETTINGS
# ======================
MODEL_PATH = "efficientnet_dataset_final_best.keras"
MODEL_URL = "https://drive.google.com/uc?id=1rz4ERlK5OL1bkf4-v0YCiqEtXgAIrFbH"

IMG_SIZE = 224
class_names = ["Glioma", "Meningioma", "Notumor", "Pituitary"]
LAST_CONV_LAYER_NAME = "top_conv"

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... Please wait."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======================
# PREPROCESS
# ======================
def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image).astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ======================
# GRAD-CAM
# ======================
def make_gradcam_heatmap(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return heatmap.numpy()

    heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_heatmap(img, heatmap):
    img = np.array(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ======================
# PDF REPORT
# ======================
def create_pdf_report(original_img_pil, heatmap_img_array, pred_class, confidence, probs):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, height - 45, "Brain MRI AI Analysis Report")

    c.setFont("Helvetica", 11)
    c.drawString(40, height - 65, "Developed by Read Aloush")
    c.drawString(40, height - 82, "Project: AI-based Brain MRI Tumor Classification")
    c.line(40, height - 95, width - 40, height - 95)

    result_text = "No tumor detected" if pred_class == "Notumor" else f"Tumor detected ({pred_class})"

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 125, "1. Prediction Summary")

    c.setFont("Helvetica", 12)
    c.drawString(55, height - 150, f"Final Model Prediction: {result_text}")
    c.drawString(55, height - 170, f"Confidence Score: {confidence:.2f}%")

    if pred_class == "Notumor":
        c.drawString(55, height - 190, "Predicted Tumor Type: Not applicable")
    else:
        c.drawString(55, height - 190, f"Predicted Tumor Type: {pred_class}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 225, "2. Class Probabilities")

    y = height - 250
    c.setFont("Helvetica", 11)

    for name, p in zip(class_names, probs):
        c.drawString(55, y, f"{name}: {p * 100:.2f}%")
        y -= 18

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y - 15, "3. Visual Explanation")

    y -= 45

    orig_buffer = BytesIO()
    original_img_pil.convert("RGB").resize((220, 220)).save(orig_buffer, format="PNG")
    orig_buffer.seek(0)

    heat_buffer = BytesIO()
    Image.fromarray(heatmap_img_array).save(heat_buffer, format="PNG")
    heat_buffer.seek(0)

    c.drawImage(ImageReader(orig_buffer), 40, y - 220, width=220, height=220)
    c.drawImage(ImageReader(heat_buffer), 310, y - 220, width=220, height=220)

    c.setFont("Helvetica", 10)
    c.drawString(90, y - 238, "Original MRI Image")
    c.drawString(355, y - 238, "Grad-CAM Focus Area")

    y = y - 275

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "4. Grad-CAM Explanation")

    y -= 22
    c.setFont("Helvetica", 10)

    explanation_lines = [
        "The Grad-CAM heatmap highlights the image regions that influenced the AI model's prediction.",
        "Warmer colors such as red and yellow indicate areas with higher model attention.",
        "This visualization helps explain where the model focused during the analysis."
    ]

    for line in explanation_lines:
        c.drawString(55, y, line)
        y -= 15

    y -= 15
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "5. Important Medical Disclaimer")

    y -= 22
    c.setFont("Helvetica", 10)

    disclaimer_lines = [
        "This report is generated by an AI model for research and educational support only.",
        "It is not a substitute for professional medical diagnosis or clinical judgment.",
        "The final diagnosis must always be made by a qualified medical specialist.",
        "Model results may be affected by image quality, dataset limitations, and model uncertainty."
    ]

    for line in disclaimer_lines:
        c.drawString(55, y, line)
        y -= 15

    c.line(40, 45, width - 40, 45)
    c.setFont("Helvetica", 9)
    c.drawString(40, 30, "Generated by Brain MRI AI Assistant")
    c.drawRightString(width - 40, 30, "Read Aloush | 2026")

    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# ======================
# HEADER
# ======================
st.title("🧠 Brain MRI AI Assistant")
st.caption("Medical + AI powered brain tumor classification")
st.markdown("### 👨‍💻 Developed by **Read Aloush**")

st.info(
    "Upload a brain MRI image and the model will predict tumor presence, type, "
    "confidence, and highlight the focus area."
)

# ======================
# UPLOAD
# ======================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ======================
# MAIN
# ======================
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(img, use_container_width=True)

    if st.button("Analyze Image", use_container_width=True):
        with st.spinner("Analyzing image..."):
            arr = prepare_image(img)
            preds = model.predict(arr, verbose=0)[0]

        idx = np.argmax(preds)
        pred_class = class_names[idx]
        confidence = preds[idx] * 100

        st.subheader("Diagnosis Result")

        if pred_class == "Notumor":
            st.success(f"No tumor detected — Confidence: {confidence:.2f}%")
        else:
            st.warning(f"Tumor detected: {pred_class} — Confidence: {confidence:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_class)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("All Probabilities")
        for name, p in zip(class_names, preds):
            st.write(f"**{name}: {p*100:.2f}%**")
            st.progress(float(p))

        heatmap_result = None

        try:
            heatmap = make_gradcam_heatmap(arr, model, LAST_CONV_LAYER_NAME)
            heatmap_result = overlay_heatmap(img, heatmap)

            st.subheader("Image Comparison")
            col_img1, col_img2 = st.columns(2)

            with col_img1:
                st.image(
                    img.resize((IMG_SIZE, IMG_SIZE)),
                    caption="Original MRI Image",
                    use_container_width=True
                )

            with col_img2:
                st.image(
                    heatmap_result,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            heatmap_buffer = BytesIO()
            Image.fromarray(heatmap_result).save(heatmap_buffer, format="PNG")

            st.download_button(
                label="⬇️ Download Heatmap",
                data=heatmap_buffer.getvalue(),
                file_name="heatmap_result.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Grad-CAM could not be generated: {e}")

        if heatmap_result is not None:
            pdf_file = create_pdf_report(img, heatmap_result, pred_class, confidence, preds)

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file,
                file_name="brain_mri_ai_report.pdf",
                mime="application/pdf"
            )

        st.divider()
        st.warning(
            "This tool is for research and educational support only. "
            "It is not a substitute for medical diagnosis."
        )

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("©️ 2026 Read Aloush | AI Brain MRI Project")