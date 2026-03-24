import torch
import torch.nn as nn
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    
try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")

# =========================================================================
# 1. Loading YOLO Age/Gender Model (best.pt)
# =========================================================================
def load_yolo_age_gender(model_path="best.pt", device=None):
    print("\n" + "="*60)
    print(f"STEP 1: Loading YOLO Age/Gender ({model_path})")
    print("="*60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    start = time.time()
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"Success! Loaded on {device} in {time.time()-start:.2f}s")
        
        classes_dict = model.names
        print(f"\nModel Classes ({len(classes_dict)} total):")
        for class_id, class_name in classes_dict.items():
            print(f"  - ID {class_id}: {class_name}")
            
        return model
    except Exception as e:
        print(f"Failed to load YOLO Age/Gender: {e}")
        return None

# =========================================================================
# 2. Loading Custom MobileNetV3 Age/Gender Model (best_checkpoint.pth)
# =========================================================================
class StandaloneGenderAge(nn.Module):
    def __init__(self, neck_in=1024, neck_out=256, head_hidden=128, num_gender=2, num_age=4):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0)
        self.neck = nn.Sequential(nn.Linear(neck_in, neck_out), nn.BatchNorm1d(neck_out))
        self.gender_head = nn.Sequential(
            nn.Linear(neck_out, head_hidden), nn.BatchNorm1d(head_hidden),
            nn.Dropout(0.3), nn.ReLU(inplace=False), nn.Linear(head_hidden, num_gender)
        )
        self.age_head = nn.Sequential(
            nn.Linear(neck_out, head_hidden), nn.BatchNorm1d(head_hidden),
            nn.Dropout(0.3), nn.ReLU(inplace=False), nn.Linear(head_hidden, num_age)
        )

    def forward(self, x):
        feats = self.backbone(x)
        neck = self.neck(feats)
        return self.gender_head(neck), self.age_head(neck)

def load_mobilenet_age_gender(checkpoint_path="best_checkpoint.pth", device=None):
    print("\n" + "="*60)
    print(f"STEP 2: Loading MobileNetV3 Age/Gender ({checkpoint_path})")
    print("="*60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    start = time.time()
    try:
        model = StandaloneGenderAge()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        def get_subdict(prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        
        model.backbone.load_state_dict(get_subdict("backbone."), strict=False)
        model.neck.load_state_dict(get_subdict("neck."), strict=True)
        model.gender_head.load_state_dict(get_subdict("gender_head."), strict=True)
        model.age_head.load_state_dict(get_subdict("age_head."), strict=True)
        
        model.to(device)
        model.eval()
        print(f"Success! Loaded on {device} in {time.time()-start:.2f}s")
        
        gender_classes = {0: "Female", 1: "Male"}
        age_classes = {0: "Child", 1: "YoungAdult", 2: "MiddleAged", 3: "OldAged"}
        
        print(f"\nModel Classes:")
        print(f"Gender Head:")
        for idx, name in gender_classes.items():
            print(f"  - ID {idx}: {name}")
            
        print(f"Age Head:")
        for idx, name in age_classes.items():
            print(f"  - ID {idx}: {name}")
            
        return model
    except Exception as e:
        print(f"Failed to load MobileNet: {e}")
        return None

# =========================================================================
# 3. Loading YOLO Mood Model (best mood.pt)
# =========================================================================
def load_yolo_mood(model_path="best mood.pt", device=None):
    print("\n" + "="*60)
    print(f"STEP 3: Loading YOLO Mood ({model_path})")
    print("="*60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    start = time.time()
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"Success! Loaded on {device} in {time.time()-start:.2f}s")
        
        classes_dict = model.names
        print(f"\nModel Classes ({len(classes_dict)} total):")
        for class_id, class_name in classes_dict.items():
            print(f"  - ID {class_id}: {class_name}")
            
        return model
    except Exception as e:
        print(f"Failed to load YOLO Mood: {e}")
        return None

# =========================================================================
# Main Execution
# =========================================================================
if __name__ == "__main__":
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {current_device.upper()}")
    
    yolo_age_model = load_yolo_age_gender("best.pt", current_device)
    mobilenet_age_model = load_mobilenet_age_gender("best_checkpoint.pth", current_device)
    yolo_mood_model = load_yolo_mood("best mood.pt", current_device)
    
    print("\nDone! All available models have been loaded and their classes extracted.")