"""
processing.py
-------------
Image enhancement / processing functions used by the Streamlit app.

Convention: every function takes an RGB uint8 numpy array (H x W x 3) and
returns an RGB uint8 numpy array of the same kind, so the output of one step
can always be fed into the next step (chaining) or into the model.
"""

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _odd(k: int) -> int:
    """Kernel sizes for OpenCV filters must be odd and >= 1."""
    k = int(k)
    if k < 1:
        k = 1
    return k if k % 2 == 1 else k + 1


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert a single channel image to 3 channel RGB."""
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# --------------------------------------------------------------------------
# 1. Zoom / Resize
# --------------------------------------------------------------------------
def resize_image(img: np.ndarray, scale_percent: int = 100) -> np.ndarray:
    """Resize the whole image by a percentage (50 = half size, 200 = double)."""
    scale = max(scale_percent, 5) / 100.0
    h, w = img.shape[:2]
    new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def zoom_image(img: np.ndarray, zoom: float = 1.0,
               center_x: int = 50, center_y: int = 50) -> np.ndarray:
    """
    Zoom into the image. The crop window is centred on (center_x%, center_y%)
    and the result is resized back to the original size, so the output keeps
    the same dimensions as the input.
    """
    zoom = max(float(zoom), 1.0)
    if zoom == 1.0:
        return img.copy()
    h, w = img.shape[:2]
    crop_w, crop_h = int(w / zoom), int(h / zoom)
    cx, cy = int(w * center_x / 100), int(h * center_y / 100)
    x1 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
    y1 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
    crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def rotate_image(img: np.ndarray, angle: int = 0) -> np.ndarray:
    """Rotate around the centre, keeping the original canvas size."""
    if angle % 360 == 0:
        return img.copy()
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def flip_image(img: np.ndarray, mode: str = "Horizontal") -> np.ndarray:
    code = {"Horizontal": 1, "Vertical": 0, "Both": -1}[mode]
    return cv2.flip(img, code)


# --------------------------------------------------------------------------
# 2. Blurring / smoothing
# --------------------------------------------------------------------------
def average_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = _odd(ksize)
    return cv2.blur(img, (k, k))


def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    k = _odd(ksize)
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))


def median_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = max(_odd(ksize), 3)
    return cv2.medianBlur(img, k)


def bilateral_filter(img: np.ndarray, diameter: int = 9,
                     sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, int(diameter), float(sigma_color), float(sigma_space))


# --------------------------------------------------------------------------
# 3. Histogram based enhancement + histogram plot
# --------------------------------------------------------------------------
def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Equalise only the brightness channel so colours are not distorted."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe_enhancement(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def plot_histograms(img: np.ndarray, title: str = "Histogram"):
    """Return a matplotlib figure with a grayscale and an RGB histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    axes[0].hist(_gray(img).ravel(), bins=256, range=(0, 255), color="gray")
    axes[0].set_title(f"{title} - Grayscale")
    axes[0].set_xlabel("Pixel intensity")
    axes[0].set_ylabel("Pixel count")

    for i, (name, colour) in enumerate(zip(["Red", "Green", "Blue"], ["red", "green", "blue"])):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
        axes[1].plot(hist, color=colour, label=name, linewidth=1.2)
    axes[1].set_title(f"{title} - RGB channels")
    axes[1].set_xlabel("Pixel intensity")
    axes[1].set_xlim(0, 255)
    axes[1].legend()

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 4. Edge detection
# --------------------------------------------------------------------------
def canny_edges(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    blurred = cv2.GaussianBlur(_gray(img), (5, 5), 0)
    return _to_rgb(cv2.Canny(blurred, int(low), int(high)))


def sobel_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = min(_odd(ksize), 7)
    g = cv2.GaussianBlur(_gray(img), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return _to_rgb(_to_uint8(mag))


def laplacian_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = min(_odd(ksize), 7)
    g = cv2.GaussianBlur(_gray(img), (3, 3), 0)
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=k)
    return _to_rgb(_to_uint8(cv2.convertScaleAbs(lap)))


def prewitt_edges(img: np.ndarray) -> np.ndarray:
    g = _gray(img).astype(np.float32)
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(g, -1, kx)
    gy = cv2.filter2D(g, -1, ky)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX)
    return _to_rgb(_to_uint8(mag))


def scharr_edges(img: np.ndarray) -> np.ndarray:
    g = _gray(img)
    gx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX)
    return _to_rgb(_to_uint8(mag))


# --------------------------------------------------------------------------
# 5. Advanced filters
# --------------------------------------------------------------------------
def sharpen(img: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """Unsharp masking: original + amount * (original - blurred)."""
    blurred = cv2.GaussianBlur(img, (0, 0), float(sigma))
    return cv2.addWeighted(img, 1.0 + float(amount), blurred, -float(amount), 0)


def brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=float(contrast), beta=int(brightness))


def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    gamma = max(float(gamma), 0.05)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return _to_rgb(_gray(img))


def negative(img: np.ndarray) -> np.ndarray:
    return 255 - img


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(_gray(img), (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _to_rgb(th)


def adaptive_threshold(img: np.ndarray, block: int = 11, c: int = 2) -> np.ndarray:
    block = max(_odd(block), 3)
    th = cv2.adaptiveThreshold(_gray(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, int(c))
    return _to_rgb(th)


def emboss(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    out = cv2.filter2D(_gray(img), -1, kernel) + 128
    return _to_rgb(_to_uint8(out))


def morphology(img: np.ndarray, operation: str = "Erosion", ksize: int = 3, iterations: int = 1) -> np.ndarray:
    ops = {
        "Erosion": cv2.MORPH_ERODE,
        "Dilation": cv2.MORPH_DILATE,
        "Opening": cv2.MORPH_OPEN,
        "Closing": cv2.MORPH_CLOSE,
        "Gradient": cv2.MORPH_GRADIENT,
    }
    k = _odd(ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(img, ops[operation], kernel, iterations=int(iterations))


# --------------------------------------------------------------------------
# Registry used by the app: name -> (function, parameter spec)
# Parameter spec: (param_name, label, widget, min, max, default[, step])
#   widget "slider" -> st.slider ; "select" -> st.selectbox (options in min)
# --------------------------------------------------------------------------
TECHNIQUES = {
    # Zoom / resize
    "Resize": {
        "group": "Zoom / Resize", "func": resize_image,
        "params": [("scale_percent", "Scale (%)", "slider", 10, 300, 100, 5)],
    },
    "Zoom": {
        "group": "Zoom / Resize", "func": zoom_image,
        "params": [("zoom", "Zoom factor", "slider", 1.0, 5.0, 2.0, 0.1),
                   ("center_x", "Centre X (%)", "slider", 0, 100, 50, 1),
                   ("center_y", "Centre Y (%)", "slider", 0, 100, 50, 1)],
    },
    "Rotate": {
        "group": "Zoom / Resize", "func": rotate_image,
        "params": [("angle", "Angle (degrees)", "slider", -180, 180, 0, 5)],
    },
    "Flip": {
        "group": "Zoom / Resize", "func": flip_image,
        "params": [("mode", "Direction", "select", ["Horizontal", "Vertical", "Both"], None, "Horizontal")],
    },
    # Blur
    "Average Blur": {
        "group": "Blur / Smoothing", "func": average_blur,
        "params": [("ksize", "Kernel size", "slider", 1, 51, 5, 2)],
    },
    "Gaussian Blur": {
        "group": "Blur / Smoothing", "func": gaussian_blur,
        "params": [("ksize", "Kernel size", "slider", 1, 51, 5, 2),
                   ("sigma", "Sigma (0 = auto)", "slider", 0.0, 20.0, 0.0, 0.5)],
    },
    "Median Blur": {
        "group": "Blur / Smoothing", "func": median_blur,
        "params": [("ksize", "Kernel size", "slider", 3, 31, 5, 2)],
    },
    "Bilateral Filter": {
        "group": "Blur / Smoothing", "func": bilateral_filter,
        "params": [("diameter", "Diameter", "slider", 3, 25, 9, 2),
                   ("sigma_color", "Sigma colour", "slider", 10, 200, 75, 5),
                   ("sigma_space", "Sigma space", "slider", 10, 200, 75, 5)],
    },
    # Histogram based
    "Histogram Equalization": {
        "group": "Histogram", "func": histogram_equalization, "params": [],
    },
    "CLAHE": {
        "group": "Histogram", "func": clahe_enhancement,
        "params": [("clip_limit", "Clip limit", "slider", 1.0, 10.0, 2.0, 0.5),
                   ("tile", "Tile grid size", "slider", 2, 16, 8, 1)],
    },
    # Edges
    "Canny Edge": {
        "group": "Edge Detection", "func": canny_edges,
        "params": [("low", "Low threshold", "slider", 0, 255, 100, 5),
                   ("high", "High threshold", "slider", 0, 255, 200, 5)],
    },
    "Sobel Edge": {
        "group": "Edge Detection", "func": sobel_edges,
        "params": [("ksize", "Kernel size", "slider", 1, 7, 3, 2)],
    },
    "Laplacian Edge": {
        "group": "Edge Detection", "func": laplacian_edges,
        "params": [("ksize", "Kernel size", "slider", 1, 7, 3, 2)],
    },
    "Prewitt Edge": {
        "group": "Edge Detection", "func": prewitt_edges, "params": [],
    },
    "Scharr Edge": {
        "group": "Edge Detection", "func": scharr_edges, "params": [],
    },
    # Advanced
    "Sharpen (Unsharp Mask)": {
        "group": "Advanced Filters", "func": sharpen,
        "params": [("amount", "Amount", "slider", 0.0, 5.0, 1.5, 0.1),
                   ("sigma", "Blur sigma", "slider", 0.5, 10.0, 1.0, 0.5)],
    },
    "Brightness / Contrast": {
        "group": "Advanced Filters", "func": brightness_contrast,
        "params": [("brightness", "Brightness", "slider", -100, 100, 0, 5),
                   ("contrast", "Contrast", "slider", 0.5, 3.0, 1.0, 0.1)],
    },
    "Gamma Correction": {
        "group": "Advanced Filters", "func": gamma_correction,
        "params": [("gamma", "Gamma", "slider", 0.2, 3.0, 1.0, 0.1)],
    },
    "Grayscale": {"group": "Advanced Filters", "func": to_grayscale, "params": []},
    "Negative": {"group": "Advanced Filters", "func": negative, "params": []},
    "Otsu Threshold": {"group": "Advanced Filters", "func": otsu_threshold, "params": []},
    "Adaptive Threshold": {
        "group": "Advanced Filters", "func": adaptive_threshold,
        "params": [("block", "Block size", "slider", 3, 51, 11, 2),
                   ("c", "Constant C", "slider", 0, 20, 2, 1)],
    },
    "Emboss": {"group": "Advanced Filters", "func": emboss, "params": []},
    "Morphology": {
        "group": "Advanced Filters", "func": morphology,
        "params": [("operation", "Operation", "select",
                    ["Erosion", "Dilation", "Opening", "Closing", "Gradient"], None, "Erosion"),
                   ("ksize", "Kernel size", "slider", 1, 15, 3, 2),
                   ("iterations", "Iterations", "slider", 1, 5, 1, 1)],
    },
}


def apply_technique(img: np.ndarray, name: str, **params) -> np.ndarray:
    """Apply a technique from the registry by name."""
    return TECHNIQUES[name]["func"](img, **params)
