#!/usr/bin/env python3
"""
GenderAge MobileNetV3-Small Model Definition
---------------------------------------------
Architecture from the Kaggle training notebook.
Backbone: MobileNetV3-Small (~2.5M params)
Inputs: 224x224 RGB
Outputs: 8-class logits combining Gender + Age

Classes:
0: Female_Child
1: Female_YoungAdult
2: Female_MiddleAged
3: Female_OldAged
4: Male_Child
5: Male_YoungAdult
6: Male_MiddleAged
7: Male_OldAged
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required.  Install it with:  pip install timm")

GENDER_LABELS = ["Female", "Male"]
AGE_SHORT_LABELS = ["Child", "YoungAdult", "MiddleAged", "OldAged"]
AGE_LABELS = [
    "Child (0-16)",
    "Young Adults (17-30)",
    "Middle-aged Adults (31-45)",
    "Old-aged Adults (45+)",
]

ALL_8_CLASSES = [
    "Female_Child", "Female_YoungAdult", "Female_MiddleAged", "Female_OldAged",
    "Male_Child", "Male_YoungAdult", "Male_MiddleAged", "Male_OldAged"
]

def load_gender_age_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a GenderAgeModel from a .pth checkpoint produced by the training pipeline.
    """
    print(f"\nLoading GenderAge checkpoint: {checkpoint_path}")

    # Create base MobileNetV3-Small
    model = timm.create_model('mobilenetv3_small_100', pretrained=False)
    
    # Replace classifier correctly for 8 classes
    # timm's mobilenetv3 uses model.classifier as the linear layer
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, len(ALL_8_CLASSES))

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Support both raw state_dict and wrapped checkpoint dicts
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if "val_loss" in ckpt:
            print(f"  Checkpoint val_loss : {ckpt['val_loss']:.4f}")
        if "epoch" in ckpt:
            print(f"  Saved at epoch      : {ckpt['epoch'] + 1}")
    else:
        state_dict = ckpt

    # Load the state dictionary
    model.load_state_dict(state_dict)
    
    # Move to requested device
    model = model.to(device)
    model.eval()

    print(f"  Model loaded ✓  (device={device})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters        : {total_params:,}")
    print(f"  Approx size (fp32): {total_params * 4 / 1024**2:.1f} MB")

    return model

if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "best_checkpoint.pth"
    m = load_gender_age_model(ckpt_path, device="cpu")

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = m(dummy)
    print(f"\nForward pass OK → logits {logits.shape}")
