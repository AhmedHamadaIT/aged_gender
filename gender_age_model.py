#!/usr/bin/env python3
"""
GenderAge MobileNetV3-Small Model Definition
---------------------------------------------
Architecture from the Kaggle training notebook.
Backbone  : MobileNetV3-Small (~2.5M params, ~7 MB float32)
Input     : 224×224 RGB images (ImageNet normalisation)
Outputs   : gender logits (2)  |  age-class logits (4)

Age classes:
  0 → Child          (0–16)
  1 → Young Adults   (17–30)
  2 → Middle-aged    (31–45)
  3 → Old-aged       (46+)
"""

import torch
import torch.nn as nn

# ── Try importing timm (required) ─────────────────────────────────────────────
try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required.  Install it with:  pip install timm"
    )

# ── Label / class constants ───────────────────────────────────────────────────
GENDER_LABELS = ["Female", "Male"]          # index 0 = Female, 1 = Male

AGE_BINS = [(0, 16), (17, 30), (31, 45), (46, 120)]
AGE_LABELS = [
    "Child (0-16)",
    "Young Adults (17-30)",
    "Middle-aged Adults (31-45)",
    "Old-aged Adults (45+)",
]

# Short labels used as keys in YOLO-style class maps
AGE_SHORT_LABELS = ["Child", "YoungAdult", "MiddleAged", "OldAged"]

# ── Model ─────────────────────────────────────────────────────────────────────

class GenderAgeModel(nn.Module):
    """
    Lightweight multi-task model for edge devices.

    Backbone  : MobileNetV3-Small
    Inputs    : 224×224 RGB
    Outputs   : gender logits (2)  |  age class logits (4)
    """

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        num_age_classes: int = 4,
        pretrained: bool = True,
        img_size: int = 224,
    ):
        super().__init__()

        print(f"  Building GenderAgeModel ({model_name})…")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Probe real output dim (timm's num_features can be wrong for MobileNetV3)
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.backbone(dummy)
            if out.dim() > 2:
                out = out.flatten(1)
            in_features = out.shape[1]

        print(f"  Backbone output features : {in_features}")

        NECK_DIM = 256

        self.neck = nn.Sequential(
            nn.Linear(in_features, NECK_DIM),
            nn.BatchNorm1d(NECK_DIM),
            nn.Hardswish(),
            nn.Dropout(0.25),
        )

        self.gender_head = nn.Sequential(
            nn.Linear(NECK_DIM, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

        self.age_head = nn.Sequential(
            nn.Linear(NECK_DIM, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_age_classes),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        embeddings = self.neck(features)
        return self.gender_head(embeddings), self.age_head(embeddings)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_gender_age_model(
    checkpoint_path: str,
    device: str = "cpu",
    model_name: str = "mobilenetv3_small_100",
    num_age_classes: int = 4,
    img_size: int = 224,
) -> GenderAgeModel:
    """
    Load a GenderAgeModel from a .pth checkpoint produced by the training pipeline.

    Parameters
    ----------
    checkpoint_path : path to best_checkpoint.pth
    device          : 'cpu' or 'cuda'
    model_name      : timm backbone identifier (must match how it was trained)
    num_age_classes : number of age bins (default 4)
    img_size        : input resolution (default 224)

    Returns
    -------
    GenderAgeModel in eval() mode on the requested device
    """
    print(f"\nLoading GenderAge checkpoint: {checkpoint_path}")

    model = GenderAgeModel(
        model_name=model_name,
        num_age_classes=num_age_classes,
        pretrained=False,       # weights come from the checkpoint
        img_size=img_size,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Support both raw state_dict and wrapped checkpoint dicts
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # Print saved validation metrics if present
        if "val_gender_acc" in ckpt:
            print(f"  Checkpoint val_gender_acc : {ckpt['val_gender_acc']:.2f}%")
        if "val_age_acc" in ckpt:
            print(f"  Checkpoint val_age_acc    : {ckpt['val_age_acc']:.2f}%")
        if "epoch" in ckpt:
            print(f"  Saved at epoch            : {ckpt['epoch'] + 1}")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"  Model loaded ✓  (device={device})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters        : {total_params:,}")
    print(f"  Approx size (fp32): {total_params * 4 / 1024**2:.1f} MB")

    return model


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "best_checkpoint.pth"
    m = load_gender_age_model(ckpt_path, device="cpu")

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        g, a = m(dummy)
    print(f"\nForward pass OK → gender {g.shape}, age {a.shape}")
    print("Gender probs:", torch.softmax(g, dim=1).numpy())
    print("Age probs   :", torch.softmax(a, dim=1).numpy())
