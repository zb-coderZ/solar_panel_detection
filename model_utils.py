"""
model_utils.py
--------------
Shared model code used by BOTH train.ipynb and app.py, so training and
prediction always use exactly the same architecture and preprocessing.

Framework: PyTorch + torchvision (supports Python 3.14).
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 with a new classification head for `num_classes` outputs."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# Evaluation / inference preprocessing: resize, tensor, ImageNet normalisation
eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Training preprocessing: same as above plus augmentation
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
    transforms.ColorJitter(contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_trained_model(path, num_classes: int, device=None) -> nn.Module:
    """Rebuild the architecture (no internet needed) and load the saved weights."""
    device = device or get_device()
    model = build_model(num_classes, pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def predict_probs(model: nn.Module, img_rgb: np.ndarray, device=None) -> np.ndarray:
    """Take an RGB uint8 array (H x W x 3) and return class probabilities."""
    device = device or next(model.parameters()).device
    tensor = eval_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()
