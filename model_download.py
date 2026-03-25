import torch
import torchreid

# 1. Build the original OSNet architecture using torchreid
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=751, # Default for Market1501
    loss='softmax',
    pretrained=False
)

# 2. Load your existing weights
weight_path = './models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
torchreid.utils.load_pretrained_weights(model, weight_path)

# 3. Prepare for inference
model.eval()

# 4. Create a dummy input matching the expected shape (B, C, H, W)
dummy_input = torch.randn(1, 3, 256, 128)

# 5. Trace the model into a standalone TorchScript graph
print("Tracing model...")
traced_model = torch.jit.trace(model, dummy_input)

# 6. Save the new format
output_path = './models/osnet_x1_0.pt'
traced_model.save(output_path)
print(f"Success! Model exported to {output_path}")