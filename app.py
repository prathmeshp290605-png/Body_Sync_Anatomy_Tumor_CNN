import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Body Sync Anatomy",
    page_icon="🧬",
    layout="wide",
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = Path("models/tumor_cnn.keras")
CLASS_PATH = Path("models/class_names.json")
SEGMENTATION_MODEL_PATH = Path(
    "models/tumor_segmentation.keras"
)

IMG_SIZE = (224, 224)
SEG_SIZE = (128, 128)

HISTORY_FILE = Path("analysis_history.json")


# =========================================================
# LOAD CLASSIFICATION MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# LOAD CLASS NAMES
# =========================================================

@st.cache_data
def load_classes():
    with open(
        CLASS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# =========================================================
# LOAD SEGMENTATION MODEL
# =========================================================

@st.cache_resource
def load_segmentation_model():
    return tf.keras.models.load_model(
        SEGMENTATION_MODEL_PATH
    )


# =========================================================
# IMAGE QUALITY
# =========================================================

def calculate_image_quality(image):

    gray = image.convert("L")

    arr = np.array(
        gray,
        dtype=np.float32
    )

    brightness = float(
        np.mean(arr)
    )

    contrast = float(
        np.std(arr)
    )

    width, height = image.size

    return {
        "brightness": brightness,
        "contrast": contrast,
        "width": width,
        "height": height,
    }


def check_mri_quality(image):

    quality = calculate_image_quality(image)

    warnings = []

    if (
        quality["width"] < 150
        or quality["height"] < 150
    ):
        warnings.append(
            "Image resolution is very low."
        )

    if quality["brightness"] < 25:
        warnings.append(
            "Image appears extremely dark."
        )

    if quality["brightness"] > 235:
        warnings.append(
            "Image appears extremely bright."
        )

    if quality["contrast"] < 15:
        warnings.append(
            "Image contrast is very low."
        )

    return quality, warnings


# =========================================================
# GRAD-CAM
# =========================================================

def get_backbone(model):

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            if "mobilenet" in layer.name.lower():
                return layer

    raise ValueError(
        "MobileNetV2 backbone not found."
    )


def get_last_conv_layer(backbone):

    for layer in reversed(
        backbone.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            return layer

    for layer in reversed(
        backbone.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.DepthwiseConv2D
        ):
            return layer

    raise ValueError(
        "No convolutional layer found."
    )


def make_gradcam_heatmap(
    image_array,
    model,
    class_index
):

    backbone = get_backbone(model)

    target_layer = get_last_conv_layer(
        backbone
    )

    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = (
            backbone_grad_model(
                image_array,
                training=False
            )
        )

        backbone_index = None

        for i, layer in enumerate(
            model.layers
        ):

            if layer.name == backbone.name:

                backbone_index = i
                break

        if backbone_index is None:

            raise ValueError(
                "MobileNetV2 layer not found."
            )

        x = backbone_output

        for layer in model.layers[
            backbone_index + 1:
        ]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        class_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients[0]
    )

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value
        + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def create_heatmap_image(
    heatmap,
    size
):

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    r = np.clip(
        2 * heatmap,
        0,
        1
    )

    g = np.clip(
        2 * (
            1
            - np.abs(
                heatmap - 0.5
            ) * 2
        ),
        0,
        1
    )

    b = np.clip(
        2 * (
            1 - heatmap
        ),
        0,
        1
    )

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    rgb = (
        rgb * 255
    ).astype(
        np.uint8
    )

    heatmap_image = Image.fromarray(
        rgb
    )

    heatmap_image = heatmap_image.resize(
        size,
        Image.Resampling.BILINEAR
    )

    return heatmap_image


def overlay_heatmap(
    original_image,
    heatmap_image,
    alpha=0.45
):

    original_image = (
        original_image.convert("RGB")
    )

    heatmap_image = (
        heatmap_image.convert("RGB")
    )

    return Image.blend(
        original_image,
        heatmap_image,
        alpha
    )


# =========================================================
# SEGMENTATION
# =========================================================

def predict_tumor_mask(
    image,
    segmentation_model
):

    seg_image = image.resize(
        SEG_SIZE
    )

    arr = np.array(
        seg_image,
        dtype=np.float32
    )

    arr = arr / 255.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    prediction = (
        segmentation_model.predict(
            arr,
            verbose=0
        )[0]
    )

    mask_probability = prediction[
        :, :, 0
    ]

    # Threshold kept at 0.2
    # because your current model detected
    # approximately 1.48% segmented area.

    binary_mask = (
        mask_probability > 0.2
    ).astype(
        np.uint8
    )

    return mask_probability, binary_mask


# =========================================================
# ANALYSIS HISTORY
# =========================================================

def load_history():

    if not HISTORY_FILE.exists():
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return []


def save_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


def add_analysis_to_history(
    prediction,
    confidence,
    tumor_area_percentage
):

    history = load_history()

    new_record = {
        "date": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "prediction": prediction,
        "confidence": round(
            confidence,
            2
        ),
        "tumor_area": round(
            tumor_area_percentage,
            2
        )
    }

    history.append(
        new_record
    )

    save_history(history)


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf_report(
    image,
    prediction,
    confidence,
    class_names,
    probabilities,
    quality,
    tumor_area_percentage,
    gradcam_success
):

    buffer = BytesIO()

    pdf = canvas.Canvas(
        buffer,
        pagesize=A4
    )

    width, height = A4

    y = height - 50

    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        y,
        "BODY SYNC ANATOMY"
    )

    y -= 30

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        y,
        "AI MRI ANALYSIS REPORT"
    )

    y -= 40

    # -----------------------------------------------------
    # CNN CLASSIFICATION
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "CNN CLASSIFICATION"
    )

    y -= 22

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawString(
        60,
        y,
        f"Prediction: {prediction}"
    )

    y -= 18

    pdf.drawString(
        60,
        y,
        f"Confidence: {confidence:.2f}%"
    )

    y -= 30

    # -----------------------------------------------------
    # CLASS PROBABILITIES
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "CLASS PROBABILITIES"
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        10
    )

    for name, probability in zip(
        class_names,
        probabilities
    ):

        pdf.drawString(
            60,
            y,
            f"{name}: "
            f"{float(probability) * 100:.2f}%"
        )

        y -= 16

    y -= 20

    # -----------------------------------------------------
    # IMAGE QUALITY
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "MRI IMAGE QUALITY"
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawString(
        60,
        y,
        f"Resolution: "
        f"{quality['width']} x "
        f"{quality['height']}"
    )

    y -= 16

    pdf.drawString(
        60,
        y,
        f"Brightness: "
        f"{quality['brightness']:.2f}"
    )

    y -= 16

    pdf.drawString(
        60,
        y,
        f"Contrast: "
        f"{quality['contrast']:.2f}"
    )

    y -= 30

    # -----------------------------------------------------
    # SEGMENTATION
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "TUMOR SEGMENTATION"
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawString(
        60,
        y,
        "Approximate Tumor Area: "
        f"{tumor_area_percentage:.2f}%"
    )

    y -= 30

    # -----------------------------------------------------
    # GRAD-CAM
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "EXPLAINABLE AI"
    )

    y -= 20

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawString(
        60,
        y,
        "Grad-CAM Generated: "
        f"{'Yes' if gradcam_success else 'No'}"
    )

    y -= 30

    # -----------------------------------------------------
    # UPLOADED MRI
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        12
    )

    pdf.drawString(
        50,
        y,
        "UPLOADED MRI"
    )

    y -= 15

    image_buffer = BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    pdf.drawImage(
        ImageReader(image_buffer),
        50,
        y - 200,
        width=250,
        height=180,
        preserveAspectRatio=True
    )

    y -= 230

    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        10
    )

    pdf.drawString(
        50,
        y,
        "DISCLAIMER"
    )

    y -= 18

    pdf.setFont(
        "Helvetica",
        8
    )

    pdf.drawString(
        50,
        y,
        "This is an AI research/educational prototype."
    )

    y -= 12

    pdf.drawString(
        50,
        y,
        "The results are not a medical diagnosis."
    )

    y -= 12

    pdf.drawString(
        50,
        y,
        "Do not use this output for clinical decisions."
    )

    pdf.save()

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# HEADER
# =========================================================

st.title(
    "🧬 Body Sync Anatomy"
)

st.subheader(
    "AI Tumor Prediction & Explainable AI Web Application"
)

st.write(
    "Upload an MRI image to obtain CNN-based "
    "classification, Grad-CAM visualization, "
    "tumor segmentation and an AI analysis report."
)


# =========================================================
# CHECK MODELS
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        "Classification model not found. "
        "Please run train.py first."
    )

    st.stop()


if not CLASS_PATH.exists():

    st.error(
        "class_names.json not found."
    )

    st.stop()


if not SEGMENTATION_MODEL_PATH.exists():

    st.error(
        "Segmentation model not found. "
        "Please run train_segmentation.py first."
    )

    st.stop()


# =========================================================
# LOAD MODELS
# =========================================================

model = load_model()

class_names = load_classes()

segmentation_model = (
    load_segmentation_model()
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# MAIN PROCESS
# =========================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =====================================================
    # IMAGE QUALITY
    # =====================================================

    quality, quality_warnings = (
        check_mri_quality(image)
    )

    st.markdown("---")

    st.markdown(
        "### 🔍 MRI Image Quality Check"
    )

    q1, q2, q3 = st.columns(3)

    with q1:

        st.metric(
            "Resolution",
            f"{quality['width']} × "
            f"{quality['height']}"
        )

    with q2:

        st.metric(
            "Brightness",
            f"{quality['brightness']:.1f}"
        )

    with q3:

        st.metric(
            "Contrast",
            f"{quality['contrast']:.1f}"
        )


    if quality_warnings:

        st.warning(
            "⚠️ Image quality warnings detected."
        )

        for warning in quality_warnings:

            st.write(
                f"• {warning}"
            )

    else:

        st.success(
            "✅ Basic image quality checks passed."
        )


    # =====================================================
    # PREPROCESSING
    # =====================================================

    resized_image = image.resize(
        IMG_SIZE
    )

    st.markdown("---")

    st.markdown(
        "### 🧹 Preprocessed MRI"
    )

    st.image(
        resized_image,
        caption="224 × 224 image used by CNN",
        width=300
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # =====================================================
    # CNN PREDICTION
    # =====================================================

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    best_index = int(
        np.argmax(probabilities)
    )

    prediction = class_names[
        best_index
    ]

    confidence = (
        float(
            probabilities[
                best_index
            ]
        ) * 100
    )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    gradcam_success = False

    try:

        heatmap = make_gradcam_heatmap(
            image_array,
            model,
            best_index
        )

        heatmap_image = (
            create_heatmap_image(
                heatmap,
                image.size
            )
        )

        overlay_image = (
            overlay_heatmap(
                image,
                heatmap_image
            )
        )

        gradcam_success = True

    except Exception as error:

        st.warning(
            "Grad-CAM could not be generated."
        )

        st.caption(
            f"Technical information: {error}"
        )


    # =====================================================
    # CNN RESULT
    # =====================================================

    st.markdown("---")

    result_col1, result_col2 = (
        st.columns(2)
    )

    with result_col1:

        st.markdown(
            "### 🧠 AI Prediction"
        )

        st.success(
            f"Prediction: {prediction}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with result_col2:

        st.markdown(
            "### 📊 Class Probabilities"
        )

        for name, probability in zip(
            class_names,
            probabilities
        ):

            percentage = (
                float(probability) * 100
            )

            st.write(
                f"**{name}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )


    # =====================================================
    # GRAD-CAM DISPLAY
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### 🔥 Grad-CAM Explanation"
    )

    if gradcam_success:

        st.image(
            overlay_image,
            caption=(
                "Highlighted regions show areas "
                "that influenced the CNN prediction."
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Grad-CAM visualization is unavailable "
            "for this model."
        )


    # =====================================================
    # TUMOR SEGMENTATION
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### 🎯 Tumor Segmentation"
    )

    mask_probability, tumor_mask = (
        predict_tumor_mask(
            image,
            segmentation_model
        )
    )


    # =====================================================
    # TUMOR AREA
    # =====================================================

    tumor_pixels = np.sum(
        tumor_mask > 0
    )

    total_pixels = (
        tumor_mask.shape[0]
        * tumor_mask.shape[1]
    )

    tumor_area_percentage = (
        tumor_pixels
        / total_pixels
    ) * 100


    # =====================================================
    # SAVE ANALYSIS HISTORY
    # =====================================================

    current_analysis = (
        prediction,
        round(confidence, 2),
        round(tumor_area_percentage, 2)
    )

    if (
        "last_saved_prediction"
        not in st.session_state
        or st.session_state.last_saved_prediction
        != current_analysis
    ):

        add_analysis_to_history(
            prediction,
            confidence,
            tumor_area_percentage
        )

        st.session_state.last_saved_prediction = (
            current_analysis
        )


    # =====================================================
    # MASK DISPLAY
    # =====================================================

    seg_col1, seg_col2 = (
        st.columns(2)
    )

    with seg_col1:

        st.image(
            tumor_mask * 255,
            caption="Predicted Tumor Mask",
            use_container_width=True
        )

    with seg_col2:

        st.metric(
            "Approximate Tumor Area",
            f"{tumor_area_percentage:.2f}%"
        )

        if tumor_area_percentage > 1:

            st.warning(
                "⚠️ A segmented region was detected "
                "by the U-Net model."
            )

        else:

            st.info(
                "ℹ️ No significant segmented region detected."
            )


    # =====================================================
    # SEGMENTATION OVERLAY
    # =====================================================

    mask_resized = Image.fromarray(
        (
            tumor_mask * 255
        ).astype(
            np.uint8
        )
    ).resize(
        image.size,
        Image.Resampling.NEAREST
    )

    mask_array = np.array(
        mask_resized
    )

    original_array = np.array(
        image
    )

    segmentation_overlay = (
        original_array.copy()
    )

    segmentation_overlay[
        mask_array > 0
    ] = [
        255,
        0,
        0
    ]

    segmentation_overlay = (
        Image.fromarray(
            segmentation_overlay
        )
    )


    st.markdown(
        "### 🖼️ Tumor Segmentation Overlay"
    )

    st.image(
        segmentation_overlay,
        caption=(
            "The highlighted red region represents "
            "the area identified by the U-Net "
            "segmentation model."
        ),
        use_container_width=True
    )


    # =====================================================
    # AI REPORT
    # =====================================================

    st.markdown("---")

    st.markdown(
        "### 📄 AI Analysis Report"
    )

    report = f"""
BODY SYNC ANATOMY
AI MRI ANALYSIS REPORT
================================

CNN CLASSIFICATION

Prediction: {prediction}
Confidence: {confidence:.2f}%

CLASS PROBABILITIES
"""

    for name, probability in zip(
        class_names,
        probabilities
    ):

        report += (
            f"{name}: "
            f"{float(probability) * 100:.2f}%\n"
        )

    report += f"""

MRI IMAGE QUALITY

Resolution: {quality['width']} × {quality['height']}
Brightness: {quality['brightness']:.2f}
Contrast: {quality['contrast']:.2f}

TUMOR SEGMENTATION

Approximate Tumor Area:
{tumor_area_percentage:.2f}%

EXPLAINABLE AI

Grad-CAM Generated:
{"Yes" if gradcam_success else "No"}

================================

DISCLAIMER

This is an AI research/educational prototype.
The results are not a medical diagnosis and
must not be used for clinical decision-making.
"""


    # TXT DOWNLOAD

    st.download_button(
        label="📥 Download AI Report",
        data=report,
        file_name=(
            "Body_Sync_Anatomy_AI_Report.txt"
        ),
        mime="text/plain"
    )


    # PDF REPORT

    pdf_data = create_pdf_report(
        image=image,
        prediction=prediction,
        confidence=confidence,
        class_names=class_names,
        probabilities=probabilities,
        quality=quality,
        tumor_area_percentage=tumor_area_percentage,
        gradcam_success=gradcam_success
    )


    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_data,
        file_name=(
            "Body_Sync_Anatomy_AI_Report.pdf"
        ),
        mime="application/pdf"
    )


    # =====================================================
    # CONFIDENCE WARNING
    # =====================================================

    st.markdown("---")

    if confidence < 70:

        st.warning(
            "⚠️ Low-confidence prediction. "
            "The model is uncertain about this image. "
            "This result is for research/educational "
            "use only."
        )

    else:

        st.info(
            "ℹ️ The prediction and visualizations "
            "are AI model outputs and are not "
            "confirmed medical findings."
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.info(
    "⚠️ Body Sync Anatomy is a B.Tech/research "
    "prototype. AI predictions, Grad-CAM and "
    "segmentation results are not medical diagnoses "
    "and should not be used for clinical decision-making."
)


# =========================================================
# ANALYSIS HISTORY DASHBOARD
# =========================================================

st.markdown("---")

st.markdown(
    "## 📊 Analysis History Dashboard"
)

history = load_history()


if history:

    total_scans = len(history)

    average_confidence = (
        sum(
            item["confidence"]
            for item in history
        )
        / total_scans
    )

    average_area = (
        sum(
            item["tumor_area"]
            for item in history
        )
        / total_scans
    )

    most_common_prediction = max(
        set(
            item["prediction"]
            for item in history
        ),
        key=lambda x: sum(
            1
            for item in history
            if item["prediction"] == x
        )
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    h1, h2, h3, h4 = (
        st.columns(4)
    )

    with h1:

        st.metric(
            "🔢 Total Scans",
            total_scans
        )

    with h2:

        st.metric(
            "📊 Avg Confidence",
            f"{average_confidence:.2f}%"
        )

    with h3:

        st.metric(
            "🎯 Avg Tumor Area",
            f"{average_area:.2f}%"
        )

    with h4:

        st.metric(
            "🧠 Most Predicted",
            most_common_prediction
        )


    # -----------------------------------------------------
    # HISTORY TABLE
    # -----------------------------------------------------

    st.markdown(
        "### 📋 Previous Analyses"
    )

    table_data = []

    for item in reversed(history):

        table_data.append(
            {
                "Date & Time": item["date"],
                "Prediction": item["prediction"],
                "Confidence": (
                    f"{item['confidence']:.2f}%"
                ),
                "Tumor Area": (
                    f"{item['tumor_area']:.2f}%"
                )
            }
        )

    st.dataframe(
        table_data,
        use_container_width=True
    )


    # -----------------------------------------------------
    # CLEAR HISTORY
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Analysis History"
    ):

        save_history([])

        st.session_state.pop(
            "last_saved_prediction",
            None
        )

        st.success(
            "Analysis history cleared."
        )

        st.rerun()


else:

    st.info(
        "No analysis history available yet. "
        "Upload an MRI image to create the first record."
    )