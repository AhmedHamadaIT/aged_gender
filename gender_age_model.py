#!/usr/bin/env python3
"""
gender_age_model.py
-------------------
GenderAge MobileNetV3-Small model definition.

Supports TWO checkpoint layouts:

  Layout A – "flat" 8-class classifier (old):
      conv_stem.weight, blocks.*, classifier.weight, …

  Layout B – "backbone + neck + heads" (current best_checkpoint.pth):
      backbone.*                            MobileNetV3-Small (no classifier)
      neck.0.weight  neck.0.bias           Linear(1024 → 256)
      neck.1.*                             BatchNorm(256)
      gender_head.0.weight/bias            Linear(256 → 128)
      gender_head.1.*                      BatchNorm(128)
      gender_head.4.weight/bias            Linear(128 → 2)
      age_head.0.weight/bias               Linear(256 → 128)
      age_head.1.*                         BatchNorm(128)
      age_head.4.weight/bias               Linear(128 → 4)

Layout is detected automatically from the first checkpoint key.

Classes (Layout A – 8-class):
  0: Female_Child        4: Male_Child
  1: Female_YoungAdult   5: Male_YoungAdult
  2: Female_MiddleAged   6: Male_MiddleAged
  3: Female_OldAged      7: Male_OldAged

Layout B outputs:  (gender_logits [B,2], age_logits [B,4])
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required.  Install it with:  pip install timm")

# ── Labels ────────────────────────────────────────────────────────────────────
GENDER_LABELS    = ["Female", "Male"]
AGE_SHORT_LABELS = ["Child", "YoungAdult", "MiddleAged", "OldAged"]
AGE_LABELS = [
    "Child (0-16)",
    "Young Adults (17-30)",
    "Middle-aged Adults (31-45)",
    "Old-aged Adults (45+)",
]
ALL_8_CLASSES = [
    "Female_Child", "Female_YoungAdult", "Female_MiddleAged", "Female_OldAged",
    "Male_Child",   "Male_YoungAdult",   "Male_MiddleAged",   "Male_OldAged",
]


# ── Layout B: backbone + neck + separate gender / age heads ───────────────────
class GenderAgeModel(nn.Module):
    """
    Exact architecture that matches best_checkpoint.pth:

        backbone  = MobileNetV3-Small (timm, num_classes=0, keeps global avg-pool)
        neck      = Linear(1024→256) + BatchNorm(256)
        gender_head = Linear(256→128) + BatchNorm(128) + Dropout + ReLU + Linear(128→2)
        age_head    = Linear(256→128) + BatchNorm(128) + Dropout + ReLU + Linear(128→4)

    The indices that appear in the checkpoint state_dict are:
        neck:        [0]=Linear  [1]=BatchNorm
        gender_head: [0]=Linear  [1]=BatchNorm  [4]=Linear   (2=Dropout, 3=ReLU)
        age_head:    [0]=Linear  [1]=BatchNorm  [4]=Linear
    """

    def __init__(self, neck_in: int = 1024, neck_out: int = 256,
                 head_hidden: int = 128,
                 num_gender: int = 2, num_age: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        import sys
        print("    [DEBUG] init backbone...", flush=True)

        # Backbone – remove the built-in classifier; keep global avg-pool
        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=False,
            num_classes=0,   # returns (B, 1024) after global avg-pool + flatten
        )

        print("    [DEBUG] init neck...", flush=True)
        self.neck = nn.Sequential(
            nn.Linear(neck_in, neck_out),   # [0]
            nn.BatchNorm1d(neck_out),        # [1]
        )

        # Gender head  (indices 0,1,2,3,4 in Sequential)
        # NOTE: inplace=False to avoid heap corruption on Jetson Orin unified memory
        self.gender_head = nn.Sequential(
            nn.Linear(neck_out, head_hidden),   # [0]
            nn.BatchNorm1d(head_hidden),         # [1]
            nn.Dropout(dropout),                 # [2]
            nn.ReLU(inplace=False),              # [3]
            nn.Linear(head_hidden, num_gender),  # [4]
        )

        # Age head
        self.age_head = nn.Sequential(
            nn.Linear(neck_out, head_hidden),   # [0]
            nn.BatchNorm1d(head_hidden),         # [1]
            nn.Dropout(dropout),                 # [2]
            nn.ReLU(inplace=False),              # [3]
            nn.Linear(head_hidden, num_age),     # [4]
        )

    def forward(self, x):
        feats  = self.backbone(x)       # (B, 1024)
        neck   = self.neck(feats)       # (B, 256)
        gender = self.gender_head(neck) # (B, 2)
        age    = self.age_head(neck)    # (B, 4)
        return gender, age


# ── Layout A: flat 8-class vanilla timm model ─────────────────────────────────
def _build_flat_model(num_classes: int = 8) -> nn.Module:
    model = timm.create_model('mobilenetv3_small_100', pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# ── Helper: strip a prefix from all keys in a state dict ──────────────────────
def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


# ── Public loader ─────────────────────────────────────────────────────────────
def load_gender_age_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load the GenderAge model from *checkpoint_path*.

    Auto-detects Layout A (flat 8-class) vs Layout B (backbone+neck+heads)
    by inspecting the first key in model_state_dict.

    Returns an nn.Module in eval() mode on *device*.
    """
    print(f"\nLoading GenderAge checkpoint: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── Unwrap checkpoint dict ─────────────────────────────────────────────
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        if "val_loss" in raw:
            print(f"  val_loss  : {raw['val_loss']:.4f}")
        if "epoch" in raw:
            print(f"  epoch     : {raw['epoch'] + 1}")
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state_dict = raw
    else:
        state_dict = raw

    first_key = next(iter(state_dict))
    print(f"  First key : {first_key}")

    # ── Layout detection ───────────────────────────────────────────────────
    if first_key.startswith("backbone."):
        # ── Layout B ──────────────────────────────────────────────────────
        print("  Detected  : Layout B  (backbone + neck + gender_head + age_head)")

        # Read dimensions from the checkpoint itself
        neck_in     = state_dict["neck.0.weight"].shape[1]   # e.g. 1024
        neck_out    = state_dict["neck.0.weight"].shape[0]   # e.g. 256
        head_hidden = state_dict["gender_head.0.weight"].shape[0]  # e.g. 128
        num_gender  = state_dict["gender_head.4.weight"].shape[0]  # e.g. 2
        num_age     = state_dict["age_head.4.weight"].shape[0]     # e.g. 4

        print(f"  neck      : {neck_in} → {neck_out}")
        print(f"  heads     : {neck_out} → {head_hidden}  "
              f"| gender_out={num_gender}  age_out={num_age}")

        import sys
        print("  [DEBUG] Creating GenderAgeModel instance...", flush=True)
        model = GenderAgeModel(
            neck_in=neck_in, neck_out=neck_out,
            head_hidden=head_hidden,
            num_gender=num_gender, num_age=num_age,
        )
        print("  [DEBUG] GenderAgeModel instance created.", flush=True)

        # Load each sub-module separately for clear error reporting
        backbone_sd    = _strip_prefix(state_dict, "backbone.")
        neck_sd        = _strip_prefix(state_dict, "neck.")
        gender_head_sd = _strip_prefix(state_dict, "gender_head.")
        age_head_sd    = _strip_prefix(state_dict, "age_head.")

        print("  [DEBUG] Loading backbone_sd...", flush=True)
        miss_b, unex_b = model.backbone.load_state_dict(backbone_sd, strict=False)
        print("  [DEBUG] Loading neck_sd...", flush=True)
        miss_n, unex_n = model.neck.load_state_dict(neck_sd, strict=True)
        print("  [DEBUG] Loading gender_head_sd...", flush=True)
        miss_g, unex_g = model.gender_head.load_state_dict(gender_head_sd, strict=True)
        print("  [DEBUG] Loading age_head_sd...", flush=True)
        miss_a, unex_a = model.age_head.load_state_dict(age_head_sd, strict=True)

        print("  [DEBUG] All state_dicts loaded successfully.", flush=True)

        # (Removed manual del to prevent Jetson glibc memory corruption prior to CUDA init)

        for name, miss, unex in [
            ("backbone",    miss_b, unex_b),
            ("neck",        miss_n, unex_n),
            ("gender_head", miss_g, unex_g),
            ("age_head",    miss_a, unex_a),
        ]:
            if miss:
                print(f"  [WARN] {name} missing keys ({len(miss)}): {miss[:3]}")
            if unex:
                print(f"  [WARN] {name} unexpected keys ({len(unex)}): {unex[:3]}")

    else:
        # ── Layout A ──────────────────────────────────────────────────────
        print("  Detected  : Layout A  (flat 8-class classifier)")

        num_classes = (state_dict["classifier.weight"].shape[0]
                       if "classifier.weight" in state_dict else 8)

        model = _build_flat_model(num_classes)
        miss, unex = model.load_state_dict(state_dict, strict=False)
        if miss:
            print(f"  [WARN] missing keys ({len(miss)}): {miss[:3]}")
        if unex:
            print(f"  [WARN] unexpected keys ({len(unex)}): {unex[:3]}")

    model.eval()  # eval on CPU first — ALWAYS before CUDA transfer
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # ── Safe CUDA transfer (Jetson / embedded GPU friendly) ──────────────────
    if device == "cuda":
        if not torch.cuda.is_available():
            print("  [WARN] CUDA not available — falling back to CPU")
            device = "cpu"
        else:
            try:
                print("  [DEBUG] 1. CUDA init...", flush=True)
                # 1. Explicit CUDA init before any CUDA operation
                torch.cuda.init()
                torch.cuda.synchronize()

                print("  [DEBUG] 2. Disabling cuDNN auto-tuner...", flush=True)
                # 2. Disable cuDNN auto-tuner (causes crash on first run on Jetson)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

                print("  [DEBUG] 3. Emptying CUDA cache...", flush=True)
                # 3. Clear any stale allocations
                torch.cuda.empty_cache()

                print("  [DEBUG] 4. Moving sub-modules...", flush=True)
                # 4. Move model sub-modules one by one (avoids OOM spike)
                if hasattr(model, 'backbone'):
                    print("  [DEBUG]   -> backbone", flush=True)
                    model.backbone = model.backbone.cuda()
                    torch.cuda.synchronize()
                if hasattr(model, 'neck'):
                    print("  [DEBUG]   -> neck", flush=True)
                    model.neck = model.neck.cuda()
                    torch.cuda.synchronize()
                if hasattr(model, 'gender_head'):
                    print("  [DEBUG]   -> gender_head", flush=True)
                    model.gender_head = model.gender_head.cuda()
                    torch.cuda.synchronize()
                if hasattr(model, 'age_head'):
                    print("  [DEBUG]   -> age_head", flush=True)
                    model.age_head = model.age_head.cuda()
                    torch.cuda.synchronize()
                if hasattr(model, 'classifier'):
                    print("  [DEBUG]   -> classifier", flush=True)
                    # Layout A flat model
                    model = model.cuda()
                    torch.cuda.synchronize()

                print(f"  CUDA transfer ✓  ({torch.cuda.get_device_name(0)})")
            except Exception as e:
                print(f"  [WARN] CUDA transfer failed ({e}) — falling back to CPU")
                device = "cpu"
                model = model.cpu()
    else:
        model = model.cpu()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Size fp32 : {total_params * 4 / 1024**2:.1f} MB")
    print(f"  Device    : {device}")
    print("  Model loaded ✓")

    return model


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "best_checkpoint.pth"
    model = load_gender_age_model(ckpt_path, device="cpu")

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    if isinstance(out, (tuple, list)):
        g, a = out
        print(f"\nForward pass OK (Layout B)")
        print(f"  gender logits : {g.shape}  → classes: {GENDER_LABELS}")
        print(f"  age logits    : {a.shape}  → classes: {AGE_SHORT_LABELS}")
    else:
        print(f"\nForward pass OK (Layout A) → logits {out.shape}")
