# Solar Panel Defect Classifier + Image Processing Lab

A mini Computer Vision application built for **Python 3.14** (tested on 3.14.2) with **PyTorch**. A MobileNetV2 CNN classifies a solar panel photo into one of **6 conditions**, and a Streamlit web app lets the user upload an image, apply OpenCV processing and enhancement techniques, and compare the original and processed results.

| | |
|---|---|
| **Task** | Image classification (6 classes) |
| **Model** | MobileNetV2 (ImageNet pre-trained), transfer learning + fine-tuning, PyTorch / torchvision |
| **Dataset** | [Solar panel clean and faulty images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-clean-and-faulty-images) (Kaggle, by Afroz) |
| **Classes** | Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered |
| **Frontend** | Streamlit |
| **Processing** | OpenCV |

---

## 1. Project structure

```
solar_defect_app/
├── train.ipynb          # Step 1: download data, preprocess, train, evaluate, save model
├── app.py               # Step 2: Streamlit web app
├── processing.py        # OpenCV image processing functions (used by app.py)
├── model_utils.py       # Shared model + preprocessing code (used by notebook AND app)
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── model/               # Created/filled by train.ipynb
    ├── solar_model.pt           (trained model weights)
    ├── class_names.json         (class labels)
    ├── metrics.json             (test accuracy and per-class scores)
    └── *.png                    (class distribution, curves, confusion matrix, ...)
```

The `model/` folder is empty when you receive the project. It is filled when you run `train.ipynb`.

---

## 2. Requirements

* **Python 3.14.2** (tested). The project also works on Python 3.10 to 3.13. **Avoid Python 3.14.1**: torchvision excludes that exact version. If you are on 3.14.1, update to 3.14.2 or newer.
* The project uses **PyTorch** instead of TensorFlow because TensorFlow does not support Python 3.14.
* **VS Code** with the **Python** and **Jupyter** extensions
* **Internet connection** for the first run (installs packages, downloads the pre-trained weights and the ~330 MB dataset)
* About 2 GB of free disk space
* A GPU is **not** required. Training on a normal laptop CPU takes roughly 10 to 30 minutes for this dataset size.

---

## 3. Setup (VS Code, Windows / macOS / Linux)

### 3.1 Open the project
1. Unzip the project.
2. In VS Code: **File > Open Folder** and select the `solar_defect_app` folder.
3. Open a terminal: **Terminal > New Terminal**.

### 3.2 Create a virtual environment

**Windows (PowerShell or CMD)**
```
python -m venv venv
venv\Scripts\activate
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install the dependencies
```
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```
This can take a few minutes because PyTorch is a large package.

### 3.4 Select the Python interpreter and kernel in VS Code
1. Press `Ctrl+Shift+P`, type **Python: Select Interpreter** and choose the one inside `venv`.
2. When you open `train.ipynb`, click **Select Kernel** (top right), choose **Python Environments**, then pick the `venv` environment.

---

## 4. Step 1: Train the model (`train.ipynb`)

1. Open `train.ipynb` in VS Code.
2. Click **Run All** (or run the cells one by one from top to bottom).

What the notebook does:

| Section | Action |
|---|---|
| 2 | Downloads the dataset with `kagglehub` and finds the class folders automatically |
| 3 | Counts images per class, shows sample images, removes unreadable files |
| 4 | Stratified **70 / 15 / 15** split into train, validation and test sets |
| 5 | Preprocessing: RGB decode, **resize to 224x224**, ImageNet normalisation, class weights for imbalance, data augmentation (flip, rotation, zoom, contrast) |
| 6 to 8 | MobileNetV2 transfer learning: Phase 1 trains the new head, Phase 2 fine-tunes the last blocks (14 to 18) with a smaller learning rate |
| 9 | Evaluation on the **unseen test set**: accuracy, precision/recall/F1 per class, confusion matrix, wrong predictions |
| 10 | Saves the model, class names, metrics and plots into `model/` |

When it finishes, check that `model/solar_model.pt` and `model/class_names.json` exist.

The first run downloads the MobileNetV2 ImageNet weights (about 14 MB) automatically. After training, the app does not need internet.

### If the dataset download fails
Use the manual route instead:
1. Download the dataset from the Kaggle link above and extract it.
2. In the configuration cell of `train.ipynb`, set the path of the extracted folder:
   ```python
   MANUAL_DATA_DIR = r"C:\Users\you\Downloads\solar-panel-clean-and-faulty-images"
   ```
3. Run the notebook again. The notebook searches inside that folder for the class folders automatically.

If `kagglehub` asks for credentials: on Kaggle go to **Settings > API > Create New Token**, then place the downloaded `kaggle.json` in `C:\Users\<you>\.kaggle\` (Windows) or `~/.kaggle/` (macOS/Linux).

### Optional: train on Google Colab (faster)
Upload `train.ipynb` **and `model_utils.py`** to Colab, set the runtime to **GPU**, run all cells, then download the generated `model/` folder and place it in the project folder on your computer.

---

## 5. Step 2: Run the web app

In the VS Code terminal (with the virtual environment active):
```
streamlit run app.py
```
Your browser opens at `http://localhost:8501`. If it does not, open that address manually. To stop the app, press `Ctrl+C` in the terminal.

### How to use the app
1. **Upload** a solar panel image in the sidebar (JPG, PNG, BMP, WEBP).
2. The **Original** image appears on the left.
3. In the sidebar, choose one or more **processing techniques**. They are applied in the order you select them. Adjust their settings with the sliders.
4. The **Processed** image appears on the right. You can download it as PNG.
5. In the **Prediction** tab, choose to classify the original image, the processed image, or both. You get the predicted class, confidence, a suggested action and a probability bar chart for all 6 classes.
6. In the **Histograms** tab, compare the grayscale and RGB histograms of the original and processed images.
7. In the **Model info** tab, see the test accuracy, per-class scores and the confusion matrix.

### Available processing techniques

| Group | Techniques |
|---|---|
| **Zoom / Resize** | Resize (%), Zoom (factor and centre point), Rotate, Flip |
| **Blur / Smoothing** | Average blur, Gaussian blur, Median blur, Bilateral filter |
| **Histogram** | Histogram plots (grayscale + RGB), Histogram equalization, CLAHE |
| **Edge detection** | Canny, Sobel, Laplacian, Prewitt, Scharr |
| **Advanced filters** | Sharpen (unsharp mask), Brightness/Contrast, Gamma correction, Grayscale, Negative, Otsu threshold, Adaptive threshold, Emboss, Morphology (erosion, dilation, opening, closing, gradient) |

Ideas to try with this dataset:
* **Canny / Sobel** on a *Physical-Damage* panel to highlight cracks.
* **Histograms** of a *Snow-Covered* vs a *Clean* panel: snow shifts the histogram toward bright values.
* **CLAHE** on a *Dusty* panel to recover contrast, then predict on the processed image and compare with the original prediction.
* **Gaussian blur** with a large kernel to see how the model's confidence drops when details disappear.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install torch` fails or finds no matching version | Check `python --version`. It must not be 3.14.1. Upgrade pip (`python -m pip install --upgrade pip`) and make sure you use a 64-bit Python. |
| App shows "Trained model not found" | Run all cells in `train.ipynb` first. Make sure you start `streamlit run app.py` from inside the project folder, so `model/` is found. |
| Notebook says no class folders found | Set `MANUAL_DATA_DIR` (see section 4). |
| Several class roots are printed in the notebook | The notebook picks the 6-class folder with the fewest images (the original data, not an augmented copy). To force a folder, set `DATA_DIR` in the cell below the list. |
| `ModuleNotFoundError` | The virtual environment is not active or the kernel is wrong. Activate `venv` and reselect the kernel. |
| Out of memory while training | Lower `BATCH_SIZE` from 16 to 8 in the configuration cell. |
| Weights download fails (no internet) | The first run needs internet for the pre-trained MobileNetV2 weights. Connect and run again. |
| Want GPU training on your NVIDIA card | Install the CUDA build from pytorch.org (choose your CUDA version there) instead of the default `torch`. The default pip install works on CPU. |
| Training is slow | Use Google Colab with a GPU, or reduce `EPOCHS_PHASE1` and `EPOCHS_PHASE2`. |
| Slow processing on huge images | The app automatically scales photos down to a maximum side of 1600 px. |

---

## 7. Technical notes (useful for the report)

* **Data split:** stratified 70/15/15, fixed random seed (42), so results are reproducible. The test set is used only once, after training.
* **Preprocessing:** all images are converted to RGB, resized to 224x224 and normalised with the ImageNet mean and standard deviation. The same code (`model_utils.py`) is used in training and in the app.
* **Augmentation (training set only):** random horizontal flip, rotation (up to 10 degrees), zoom (90% to 110%) and contrast (10%). Brightness changes are avoided on purpose because strong brightness changes can blur the difference between *Clean* and *Dusty*.
* **Imbalance:** class weights are computed from the training set.
* **Training strategy:** Phase 1 trains only the new classifier head (learning rate 1e-3). Phase 2 unfreezes blocks 14 to 18 with learning rate 1e-4. BatchNorm layers of the frozen blocks stay in evaluation mode. Early stopping (patience 4) and learning rate reduction are used, and the best epoch is restored.
* **Evaluation:** accuracy, per-class precision/recall/F1, confusion matrix and a review of misclassified images. Check the confusion matrix for classes that are visually similar (for example Clean vs Dusty, or Electrical-damage vs Physical-Damage) and discuss them in your report.
* **Your own results will vary** with the dataset version and the random seed. Report the numbers that your run prints, not numbers from elsewhere.
* **Limitations:** the dataset is small and was collected from the internet, so the model may not generalise perfectly to photos taken with a different camera, angle or lighting.

---

## 8. Credits

* Dataset: *Solar panel clean and faulty images* by Afroz (Kaggle, `pythonafroz/solar-panel-clean-and-faulty-images`). Check the dataset page for its licence and terms of use.
* Model backbone: MobileNetV2 (Sandler et al., 2018), pre-trained on ImageNet, via torchvision.
* Libraries: PyTorch, torchvision, OpenCV, Streamlit, scikit-learn, Matplotlib, Seaborn, Pillow, kagglehub.
