# SolarScope: Solar Panel Defect Classifier

SolarScope is a computer-vision application for inspecting solar-panel images. It combines a PyTorch image-classification model with an interactive Streamlit image-processing laboratory.

The application can:

- classify a panel image into one of six conditions;
- apply one or more OpenCV processing operations in a user-defined order;
- compare the original and processed images;
- run predictions on the original image, the processed image, or both;
- display class probabilities, confidence warnings, and a suggested inspection action;
- compare grayscale and RGB histograms; and
- display the trained model's test metrics and confusion matrix.

> **Project status:** The repository includes a trained model and its evaluation artifacts in [`model/`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/model). The Streamlit application can therefore be started immediately after installing the dependencies.

## Project summary

| Item | Details |
|---|---|
| Task | Six-class solar-panel image classification |
| Model | MobileNetV2 with ImageNet transfer learning and fine-tuning |
| Framework | PyTorch and torchvision |
| Dataset | [Solar panel clean and faulty images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-clean-and-faulty-images) |
| Web application | Streamlit |
| Image processing | OpenCV |
| Evaluation | Accuracy, loss, precision, recall, F1-score, confusion matrix, and wrong-prediction review |
| Tested environment | Python 3.14.2, PyTorch 2.14.1+cpu, CPU inference/training |

## Recognised classes

The model predicts one of the following labels:

1. `Bird-drop`
2. `Clean`
3. `Dusty`
4. `Electrical-damage`
5. `Physical-Damage`
6. `Snow-Covered`

The application also maps each class to a short explanation and a suggested next action. These suggestions are for visual-assistance purposes only and are not a substitute for a qualified electrical or solar-panel inspection.

## Repository structure

```text
solar_defect_app/
├── app.py                  # Streamlit dashboard and prediction workflow
├── model_utils.py          # Shared model architecture and preprocessing
├── processing.py           # OpenCV processing functions and technique registry
├── train.ipynb             # Dataset preparation, training, and evaluation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── model/
    ├── solar_model.pt     # Trained MobileNetV2 weights
    ├── class_names.json   # Class order used by the model
    ├── metrics.json       # Saved test metrics
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── class_distribution.png
    ├── sample_images.png
    └── wrong_predictions.png
```

`model/` is generated and populated by the notebook. The current repository already contains the generated artifacts, including [`model/solar_model.pt`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/model/solar_model.pt).

## Model and data pipeline

### Dataset

The training notebook downloads the Kaggle dataset with `kagglehub`. It searches the downloaded directory for the class-folder layout and selects the six-class source directory with the fewest images when multiple candidate roots are found. This avoids accidentally selecting an augmented duplicate.

The current saved model was evaluated on:

- **907 total images**
- **634 training images**
- **136 validation images**
- **137 test images**
- a fixed, stratified **70% / 15% / 15%** split
- random seed **42**

Unreadable files are removed from the file list before splitting. If automatic download is unavailable, the notebook supports a manually extracted dataset directory.

### Preprocessing and augmentation

The same inference preprocessing is used during training and in the web application through [`model_utils.py`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/model_utils.py):

- convert images to RGB;
- resize to **224 × 224**;
- convert to tensors; and
- normalise with ImageNet mean and standard deviation.

Training images additionally use:

- random horizontal flipping;
- random affine rotation up to 10 degrees;
- random scale between 90% and 110%; and
- a small contrast variation.

Class weights are calculated from the training split to reduce the effect of class imbalance.

### Training strategy

The notebook trains MobileNetV2 in two phases:

1. **Head training:** the pretrained feature extractor is frozen and the new six-class classifier head is trained.
2. **Fine-tuning:** the final MobileNetV2 feature blocks (14–18) are unfrozen and trained with a smaller learning rate.

The classifier head uses dropout (`p=0.3`) followed by a linear layer. Training uses early stopping, learning-rate reduction, and restoration of the best validation epoch. The default notebook configuration is:

```text
Batch size:             16
Phase 1 maximum epochs: 12
Phase 2 maximum epochs: 10
Phase 1 learning rate:  1e-3
Phase 2 learning rate:  1e-4
Early-stopping patience: 4 epochs
```

The notebook automatically uses CUDA when a CUDA-capable PyTorch installation and GPU are available; otherwise it uses the CPU.

## Current evaluation results

The values below are read from [`model/metrics.json`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/model/metrics.json) and describe the model currently stored in the repository. Results can change if the dataset, split, seed, or training configuration changes.

| Metric | Value |
|---|---:|
| Test accuracy | **81.75%** |
| Test loss | 0.6585 |
| Test images | 137 |
| Input size | 224 × 224 |

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bird-drop | 82.14% | 76.67% | 79.31% |
| Clean | 82.35% | 90.32% | 86.15% |
| Dusty | 75.86% | 66.67% | 70.97% |
| Electrical-damage | 72.22% | 92.86% | 81.25% |
| Physical-Damage | 80.00% | 80.00% | 80.00% |
| Snow-Covered | 100.00% | 94.74% | 97.30% |

The generated plots in `model/` provide additional information about class distribution, training history, confusion patterns, and incorrect predictions. The metrics shown in the Streamlit **Model info** tab come from the same JSON file.

## Requirements

- Python **3.10–3.14**
- Python **3.14.2** is the tested version.
- Avoid Python **3.14.1**, because the compatible torchvision build is not available for that exact version.
- VS Code with the Python and Jupyter extensions is recommended for running the notebook.
- Internet access is required for first-time dependency installation, dataset download, and MobileNetV2 ImageNet weights.
- Approximately 2 GB of free disk space is recommended for dependencies, the dataset cache, and generated artifacts.
- A GPU is optional. CPU training is supported but slower.

## Installation

From the project directory, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, run this once for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, select the interpreter inside `.venv`. When opening [`train.ipynb`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/train.ipynb), select the same environment as the notebook kernel.

## Run the application

With the virtual environment active, run:

```bash
streamlit run app.py
```

Open <http://localhost:8501> if Streamlit does not open a browser automatically.

### Application workflow

1. Upload a JPG, JPEG, PNG, BMP, or WEBP image from the sidebar.
2. Choose zero or more processing techniques. Techniques are applied in the order selected.
3. Review the original and processed images side by side.
4. Download the processed image as a PNG.
5. In **Prediction**, classify the original image, processed image, or both.
6. Review the predicted class, confidence, probability chart, explanation, and suggested action.
7. In **Histograms**, compare grayscale and RGB distributions.
8. In **Model info**, review the saved test accuracy, per-class metrics, input size, image counts, and confusion matrix.

Uploaded images are automatically EXIF-rotated, converted to RGB, and scaled down when their longest side exceeds **1600 pixels** so that the application remains responsive. The model itself always receives its standard 224 × 224 input through the shared preprocessing pipeline.

## Available image-processing techniques

All operations in [`processing.py`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/processing.py) accept and return RGB `uint8` NumPy images, so operations can be chained safely.

| Category | Techniques |
|---|---|
| Zoom / resize | Resize, Zoom, Rotate, Flip |
| Blur / smoothing | Average Blur, Gaussian Blur, Median Blur, Bilateral Filter |
| Histogram enhancement | Histogram Equalization, CLAHE |
| Edge detection | Canny Edge, Sobel Edge, Laplacian Edge, Prewitt Edge, Scharr Edge |
| Advanced filters | Sharpen (Unsharp Mask), Brightness / Contrast, Gamma Correction, Grayscale, Negative, Otsu Threshold, Adaptive Threshold, Emboss, Morphology |

Morphology supports erosion, dilation, opening, closing, and gradient operations. Most techniques expose their parameters as Streamlit sliders or select boxes.

## Retrain the model

Retraining is optional because the repository already includes model artifacts.

1. Open [`train.ipynb`](D:/05.%20UNI%20Works/01.%20Computer%20Vision%20Lab/solar_defect_app/train.ipynb).
2. Select the project virtual environment as the notebook kernel.
3. Run the cells from top to bottom.
4. Allow `kagglehub` to download the dataset, or set `MANUAL_DATA_DIR` in the configuration cell to an extracted local dataset directory.
5. Confirm that the notebook regenerates the files in `model/`.
6. Restart Streamlit if it was already running so the cached model is reloaded.

For Kaggle authentication, create an API token in Kaggle settings and place `kaggle.json` in the standard Kaggle credentials directory (`%USERPROFILE%\.kaggle\` on Windows or `~/.kaggle/` on macOS/Linux).

### Optional Google Colab workflow

Upload `train.ipynb` and `model_utils.py` to Colab, select a GPU runtime, run the notebook, download the generated `model/` directory, and copy it into the project directory.

## Troubleshooting

| Problem | Resolution |
|---|---|
| `pip install torch` reports no matching distribution | Confirm that Python is 64-bit and is not 3.14.1. Upgrade pip and use Python 3.14.2 or another supported version. |
| The app reports that the trained model is missing | Run the notebook or restore `model/solar_model.pt` and `model/class_names.json`. Start Streamlit from the project directory. |
| Notebook cannot find class folders | Set `MANUAL_DATA_DIR` to the extracted dataset directory and run the notebook again. |
| Kaggle download fails | Configure Kaggle API credentials or download and extract the dataset manually. |
| `ModuleNotFoundError` appears | Activate the project virtual environment and select it as the VS Code interpreter and notebook kernel. |
| Training runs out of memory | Reduce `BATCH_SIZE` from 16 to 8 in the notebook configuration. |
| Pretrained weights cannot be downloaded | Connect to the internet for the first training run. The saved model can be used offline afterward. |
| Training is slow | Use a CUDA-enabled PyTorch installation or train in Google Colab with a GPU. |
| Very large uploads are slow | The app automatically limits the longest image side to 1600 pixels. |
| Prediction confidence is low | Review the warning and compare the original and processed predictions. The model may be seeing an unclear image or an image outside the training distribution. |

## Limitations

- The dataset is relatively small and collected from online sources.
- Performance may change for different cameras, viewpoints, panel types, lighting conditions, and geographic locations.
- Image processing can make an image easier to inspect, but it does not guarantee a more accurate prediction.
- A classification result should not be used as a safety-critical electrical diagnosis.
- The reported metrics are for the saved test split and should not be interpreted as a guarantee of real-world accuracy.

## Credits

- Dataset: *Solar panel clean and faulty images* by Afroz on Kaggle.
- Model backbone: MobileNetV2 by Sandler et al., pretrained on ImageNet through torchvision.
- Libraries: PyTorch, torchvision, OpenCV, Streamlit, scikit-learn, NumPy, pandas, Matplotlib, Seaborn, Pillow, and kagglehub.

Check the Kaggle dataset page and the individual library licences before redistributing the dataset or deploying the application commercially.
