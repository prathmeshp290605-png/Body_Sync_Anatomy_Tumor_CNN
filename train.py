import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Body Sync Anatomy",
    page_icon="🧬",
    layout="wide",
)

MODEL_PATH = Path("models/tumor_cnn.keras")
CLASS_PATH = Path("models/class_names.json")
IMG_SIZE = (224, 224)


# =========================================================
# LOAD MODEL AND CLASSES
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# FIND MOBILENETV2 BACKBONE
# =========================================================

def get_backbone(model):

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "mobilenet" in layer.name.lower():
                return layer

    raise ValueError("MobileNetV2 backbone not found.")


# =========================================================
# FIND LAST CONVOLUTIONAL LAYER
# =========================================================

def get_last_conv_layer(backbone):

    for layer in reversed(backbone.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    # Some MobileNetV2 versions use DepthwiseConv2D
    for layer in reversed(backbone.layers):

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return layer

    raise ValueError("No convolutional layer found.")


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam_heatmap(image_array, model, class_index):

    backbone = get_backbone(model)

    target_layer = get_last_conv_layer(backbone)

    # Create a model that returns:
    # target layer output + final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Average gradients
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weight feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # Remove negative values
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# =========================================================
# CREATE HEATMAP
# =========================================================

def create_heatmap_image(heatmap, size):

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    # Create RGB heatmap
    r = heatmap
    g = np.sqrt(heatmap)
    b = 1.0 - heatmap

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    rgb = (
        rgb * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        rgb
    )

    heatmap_image = heatmap_image.resize(
        size,
        Image.Resampling.BILINEAR
    )

    return heatmap_image


# =========================================================
# OVERLAY
# =========================================================

def overlay_heatmap(
    original_image,
    heatmap_image,
    alpha=0.45
):

    original_image = original_image.convert(
        "RGB"
    )

    heatmap_image = heatmap_image.convert(
        "RGB"
    )

    return Image.blend(
        original_image,
        heatmap_image,
        alpha
    )


# =========================================================
# HEADER
# =========================================================

st.title("🧬 Body Sync Anatomy")

st.subheader(
    "AI Tumor Prediction & Explainable AI"
)

st.write(
    "Upload an MRI image to obtain a CNN-based "
    "tumor classification and a Grad-CAM visualization "
    "of image regions that influenced the prediction."
)


# =========================================================
# CHECK FILES
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        "tumor_cnn.keras not found inside models folder."
    )

    st.stop()


if not CLASS_PATH.exists():

    st.error(
        "class_names.json not found inside models folder."
    )

    st.stop()


# =========================================================
# LOAD
# =========================================================

model = load_model()

class_names = load_classes()


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # =====================================================
    # PREPROCESS
    # =====================================================

    resized_image = image.resize(
        IMG_SIZE
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
    # PREDICTION
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

        heatmap_image = create_heatmap_image(
            heatmap,
            image.size
        )

        overlay_image = overlay_heatmap(
            image,
            heatmap_image
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
    # IMAGE SECTION
    # =====================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🩻 Original MRI"
        )

        st.image(
            image,
            caption="Uploaded MRI",
            use_container_width=True
        )


    with col2:

        st.markdown(
            "### 🔥 Grad-CAM Heatmap"
        )

        if gradcam_success:

            st.image(
                overlay_image,
                caption=(
                    "Highlighted regions show areas "
                    "that contributed more strongly "
                    "to the CNN prediction."
                ),
                use_container_width=True
            )

        else:

            st.info(
                "Grad-CAM visualization unavailable."
            )


    # =====================================================
    # PREDICTION
    # =====================================================

    st.divider()

    result_col1, result_col2 = st.columns(2)

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
    # WARNING
    # =====================================================

    if confidence < 70:

        st.warning(
            "⚠️ Low-confidence prediction. "
            "The model is uncertain about this image. "
            "This result is intended for research/educational "
            "use only and is not a medical diagnosis."
        )

    else:

        st.info(
            "ℹ️ Grad-CAM highlights image regions that "
            "influenced the model prediction. It should "
            "not be interpreted as a confirmed tumor "
            "location."
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.info(
    "⚠️ Body Sync Anatomy is a B.Tech/research prototype. "
    "AI predictions and Grad-CAM visualizations are not "
    "medical diagnoses and should not be used for clinical "
    "decision-making."
)