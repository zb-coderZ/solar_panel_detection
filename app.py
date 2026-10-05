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


def inject_theme():
    """Apply the visual system for the dashboard without changing app behavior."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #10233f;
            --muted: #64748b;
            --navy: #102a43;
            --teal: #0f766e;
            --teal-soft: #e6f5f2;
            --gold: #e3a008;
            --canvas: #f6f9fc;
            --line: #dce6ef;
            --white: #ffffff;
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            color: var(--ink);
        }
        .stApp {
            background: linear-gradient(180deg, #f8fbfd 0%, var(--canvas) 52%, #f3f7fa 100%);
        }
        .block-container {
            max-width: 1440px;
            padding: 1.25rem 3.25rem 4rem;
        }
        [data-testid="stHeader"] {
            background: rgba(248, 251, 253, .88);
        }
        [data-testid="stSidebar"] {
            background: var(--navy);
            border-right: 0;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1.35rem 1.1rem;
        }
        [data-testid="stSidebar"] * {
            color: #edf7f7;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,.09);
            border-color: rgba(255,255,255,.22);
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #b7d3d4;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #a9c4c6;
        }
        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            letter-spacing: -.025em;
        }
        h1 { font-size: clamp(2.15rem, 4vw, 3.55rem) !important; line-height: 1.05 !important; }
        h2 { margin-top: 1.6rem; }
        [data-testid="stMetric"] {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(16,42,67,.05);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--navy); font-family: 'Space Grotesk', sans-serif; }
        [data-testid="stImage"] img { border-radius: 16px; border: 1px solid var(--line); }
        [data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #b9d7d3;
            color: var(--teal);
            font-weight: 700;
            background: var(--white);
            transition: all .2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--teal);
            color: var(--white);
            background: var(--teal);
        }
        .premium-nav {
            align-items: center;
            background: rgba(255,255,255,.88);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(16,42,67,.07);
            display: flex;
            gap: 1.4rem;
            justify-content: space-between;
            margin-bottom: 2.6rem;
            padding: .8rem 1.15rem;
            position: sticky;
            top: .5rem;
            z-index: 100;
            backdrop-filter: blur(14px);
        }
        .brand-lockup { align-items: center; display: flex; gap: .7rem; min-width: 240px; }
        .brand-mark {
            align-items: center; background: var(--gold); border-radius: 11px;
            color: var(--navy); display: flex; font-size: 1.2rem; height: 38px; justify-content: center; width: 38px;
        }
        .brand-name { color: var(--navy); font-family: 'Space Grotesk'; font-size: 1rem; font-weight: 700; }
        .brand-sub { color: var(--muted); font-size: .72rem; }
        .nav-links { display: flex; gap: 1.25rem; }
        .nav-links span { color: var(--muted); font-size: .86rem; font-weight: 600; }
        .nav-links span:hover { color: var(--teal); }
        .nav-status {
            background: var(--teal-soft); border-radius: 999px; color: var(--teal);
            font-size: .76rem; font-weight: 700; padding: .4rem .75rem; white-space: nowrap;
        }
        .hero-kicker { color: var(--teal); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .hero-copy { color: var(--muted); font-size: 1.04rem; line-height: 1.7; max-width: 760px; }
        .section-label {
            color: var(--teal); font-size: .75rem; font-weight: 700; letter-spacing: .13em;
            margin: 2.1rem 0 .7rem; text-transform: uppercase;
        }
        .empty-state {
            background: linear-gradient(135deg, #ffffff, #edf8f6);
            border: 1px dashed #9ccbc4; border-radius: 18px; padding: 2rem; text-align: center;
        }
        .empty-state strong { color: var(--navy); font-family: 'Space Grotesk'; font-size: 1.2rem; }
        @media (max-width: 760px) {
            .block-container { padding: 1rem 1rem 3rem; }
            .premium-nav { gap: .5rem; margin-bottom: 1.7rem; }
            .nav-links { gap: .55rem; }
            .nav-links span { font-size: .7rem; }
            .brand-lockup { min-width: 0; }
            .brand-sub { display: none; }
            .nav-status { font-size: .65rem; padding: .32rem .5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    st.sidebar.markdown("### Image lab")
    st.sidebar.caption("Refine the image before classification.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 2 · Processing")
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
    inject_theme()
    model, classes = load_model_and_classes()
    st.markdown(
        f"""
        <nav class="premium-nav" aria-label="Primary navigation">
            <div class="brand-lockup">
                <div class="brand-mark">☀</div>
                <div>
                    <div class="brand-name">SolarScope</div>
                    <div class="brand-sub">Defect intelligence workspace</div>
                </div>
            </div>
            <div class="nav-links" aria-label="Workspace areas">
                <span>Workspace</span>
                <span>Results</span>
                <span>Insights</span>
            </div>
            <div class="nav-status">● {"Model ready" if model is not None else "Model offline"}</div>
        </nav>
        <div id="workspace" class="hero-kicker">Solar asset diagnostics</div>
        """,
        unsafe_allow_html=True,
    )
    st.title("See the signal in every panel.")
    st.markdown(
        '<p class="hero-copy">Upload a solar panel image, enhance the visual signal, and get an explainable condition assessment from your trained CNN.</p>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "Trained model not found. Run all cells of **train.ipynb** first. "
            "It creates `model/solar_model.pt` and `model/class_names.json`. "
            "Image processing still works without the model."
        )

    st.sidebar.markdown("#### 1 · Upload")
    uploaded = st.sidebar.file_uploader("Solar panel image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    steps = sidebar_processing_controls()

    if uploaded is None:
        st.markdown(
            '<div class="empty-state"><strong>Your workspace is ready</strong><br>'
            '<span>Upload a panel image from the left to begin a visual inspection.</span></div>',
            unsafe_allow_html=True,
        )
        if model is not None:
            st.caption("Recognised conditions · " + " · ".join(classes))
        return

    try:
        original = read_uploaded_image(uploaded)
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        return

    processed = run_pipeline(original, steps)

    st.markdown('<div id="results" class="section-label">01 · Visual comparison</div>', unsafe_allow_html=True)
    st.subheader("Original and enhanced signal")
    st.caption("Review the source image alongside the version sent to the classifier.")

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

    st.markdown('<div id="insights" class="section-label">02 · Intelligence</div>', unsafe_allow_html=True)
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
