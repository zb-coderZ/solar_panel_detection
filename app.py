"""
Solar Panel Defect Classifier + Image Processing Lab
Run with:  streamlit run app.py
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import model_utils as mu
import processing as P

MODEL_PATH = Path("model/solar_model.pt")
CLASSES_PATH = Path("model/class_names.json")
METRICS_PATH = Path("model/metrics.json")
MAX_SIDE = 1600          # very large photos are scaled down so the app stays fast
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Solar Panel Defect Classifier", page_icon="☀️", layout="wide")

# Short explanation + suggested action per class. Keys are normalised class names.
CLASS_INFO = {
    "clean": ("The panel surface looks clean.", "No action needed. Keep monitoring regularly."),
    "dusty": ("Dust has built up on the panel surface.", "Schedule cleaning. Dust reduces the light reaching the cells."),
    "birddrop": ("Bird droppings are covering part of the panel.", "Clean soon. Droppings can create hot spots."),
    "electricaldamage": ("The panel shows signs of electrical damage (burn marks, discolouration).", "Have a technician inspect it. Do not touch the panel."),
    "physicaldamage": ("The panel has visible physical damage (cracks, broken glass).", "Inspect and replace if the glass or cells are broken."),
    "snowcovered": ("The panel is covered with snow.", "Remove the snow safely. Output is blocked until it clears."),
}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    model = mu.load_trained_model(MODEL_PATH, num_classes=len(classes))
    return model, classes


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def read_uploaded_image(uploaded) -> np.ndarray:
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(img, dtype=np.uint8)


def predict(model, classes, img: np.ndarray):
    """Preprocess exactly like in training and return (probabilities, top index)."""
    probs = mu.predict_probs(model, img)
    return probs, int(np.argmax(probs))


def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# UI pieces
# --------------------------------------------------------------------------
def show_prediction(title: str, img: np.ndarray, model, classes):
    probs, top = predict(model, classes, img)
    name, conf = classes[top], float(probs[top])
    meaning, action = CLASS_INFO.get(normalise(name), ("", ""))

    st.markdown(f"#### {title}")
    c1, c2 = st.columns([1, 1])
    c1.metric("Predicted class", name)
    c2.metric("Confidence", f"{conf:.1%}")
    if conf < LOW_CONFIDENCE:
        st.warning("Low confidence. The image may be unclear or different from the training data.")
    if meaning:
        st.write(meaning)
        st.caption(f"Suggested action: {action}")

    chart_df = pd.DataFrame({"Class": classes, "Probability": probs}).set_index("Class")
    st.bar_chart(chart_df, height=260)


def sidebar_processing_controls():
    """Build the sidebar controls and return the list of (technique, params) to apply."""
    st.sidebar.header("2. Processing")
    st.sidebar.caption("Pick one or more techniques. They are applied in the order you select them.")

    groups = {}
    for name, spec in P.TECHNIQUES.items():
        groups.setdefault(spec["group"], []).append(name)

    options = [n for g in groups.values() for n in g]
    selected = st.sidebar.multiselect(
        "Techniques", options, placeholder="Choose techniques (optional)",
        help="Groups: " + ", ".join(groups.keys()),
    )

    steps = []
    for name in selected:
        spec = P.TECHNIQUES[name]
        params = {}
        with st.sidebar.expander(f"{name} settings", expanded=True):
            if not spec["params"]:
                st.caption("No settings for this technique.")
            for p in spec["params"]:
                key = f"{name}_{p[0]}"
                if p[2] == "slider":
                    _, label, _, lo, hi, default, step = p
                    params[p[0]] = st.slider(label, lo, hi, default, step, key=key)
                else:
                    _, label, _, options_list, _, default = p
                    params[p[0]] = st.selectbox(label, options_list, index=options_list.index(default), key=key)
        steps.append((name, params))
    return steps


def run_pipeline(img: np.ndarray, steps):
    out = img.copy()
    for name, params in steps:
        out = P.apply_technique(out, name, **params)
    return out


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main():
    st.title("☀️ Solar Panel Defect Classifier")
    st.caption("Upload a solar panel photo, apply image processing, and classify its condition with a trained CNN (MobileNetV2, PyTorch).")

    model, classes = load_model_and_classes()
    if model is None:
        st.error(
            "Trained model not found. Run all cells of **train.ipynb** first. "
            "It creates `model/solar_model.pt` and `model/class_names.json`. "
            "Image processing still works without the model."
        )

    st.sidebar.header("1. Upload")
    uploaded = st.sidebar.file_uploader("Solar panel image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    steps = sidebar_processing_controls()

    if uploaded is None:
        st.info("Upload an image from the sidebar to begin.")
        if model is not None:
            st.write("**The model can recognise:** " + ", ".join(classes))
        return

    try:
        original = read_uploaded_image(uploaded)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        return

    processed = run_pipeline(original, steps)

    # ---- original vs processed -------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Original")
        st.image(original, width="stretch")
        st.caption(f"{original.shape[1]} x {original.shape[0]} px")
    with col_b:
        st.subheader("Processed")
        st.image(processed, width="stretch")
        st.caption(f"{processed.shape[1]} x {processed.shape[0]} px" +
                   (" | steps: " + " → ".join(n for n, _ in steps) if steps else " | no processing selected"))
        st.download_button("Download processed image", to_png_bytes(processed),
                           file_name="processed.png", mime="image/png")

    # ---- tabs ---------------------------------------------------------------
    tab_pred, tab_hist, tab_info = st.tabs(["🔍 Prediction", "📊 Histograms", "ℹ️ Model info"])

    with tab_pred:
        if model is None:
            st.info("Train the model with train.ipynb to enable predictions.")
        else:
            target = st.radio(
                "Run the prediction on", ["Original", "Processed", "Both"],
                horizontal=True, index=0 if not steps else 2,
            )
            if target == "Both":
                c1, c2 = st.columns(2)
                with c1:
                    show_prediction("Original image", original, model, classes)
                with c2:
                    show_prediction("Processed image", processed, model, classes)
            elif target == "Original":
                show_prediction("Original image", original, model, classes)
            else:
                show_prediction("Processed image", processed, model, classes)
            st.caption("Tip: compare original and processed results to see how blur, edges or "
                       "contrast changes affect the model's decision.")

    with tab_hist:
        h1, h2 = st.columns(2)
        with h1:
            st.pyplot(P.plot_histograms(original, "Original"))
        with h2:
            st.pyplot(P.plot_histograms(processed, "Processed"))
        st.caption("Dust and snow usually shift the histogram toward brighter values and reduce contrast. "
                   "Try Histogram Equalization or CLAHE and watch the histogram spread out.")

    with tab_info:
        metrics = load_metrics()
        if metrics is None:
            st.write("No metrics file found. Run train.ipynb to generate it.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
            m2.metric("Images used", metrics["num_images"])
            m3.metric("Test images", metrics["test_images"])
            st.write(f"Backbone: {metrics['backbone']}, input size {metrics['image_size'][0]}x{metrics['image_size'][1]}")
            st.dataframe(pd.DataFrame(metrics["per_class"]).T.round(3), width="stretch")
            cm_path = Path("model/confusion_matrix.png")
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion matrix on the test set")


if __name__ == "__main__":
    main()
