# Setting Up and Running on NVIDIA Orin

This guide provides step-by-step instructions for setting up the environment, exporting the YOLO model to ONNX, and running inference on an NVIDIA Orin device.

## Prerequisites

It is highly recommended to run this project inside an Nvidia L4T (Linux for Tegra) Machine Learning container or ensure that PyTorch and Torchvision are installed via Nvidia's provided Jetson wheels, rather than from standard PyPI, as standard PyPI versions do not support Jetson's hardware acceleration properly.

If you don't use a container, ensure you have JetPack installed on your Orin.

## Step 1: Clone the Repository

On your NVIDIA Orin device, open a terminal and clone the repository:

```bash
git clone https://github.com/AhmedHamadaIT/aged_gender.git
cd aged_gender
```

## Step 2: Run the Setup Script

The setup script creates a virtual environment that specifically allows access to `system-site-packages`. This is crucial on Jetson devices to utilize the pre-installed optimized versions of PyTorch, Torchvision, and TensorRT.

Make the script executable and run it:

```bash
chmod +x setup_orin.sh
./setup_orin.sh
```

## Step 3: Activate the Virtual Environment

Activate the virtual environment created in the previous step:

```bash
source .venv/bin/activate
```

*(You will need to run this command every time you open a new terminal to work on this project).*

## Step 4: Export the Model to ONNX

We have provided a script to export your PyTorch model (`best.pt`) to the ONNX format or TensorRT Engine. The script uses settings (`dynamic=False` and `simplify=True`) which are highly recommended for the best performance when subsequently generating TensorRT engines on the NVIDIA Orin.

Make sure your `best.pt` file is in the project directory, then run:

```bash
python export_model.py --model best.pt --format onnx
```

This will generate a `best.onnx` file in your directory.

## Step 5: (Optional but Recommended) Generate TensorRT Engine

For maximum performance on an Orin device, you should convert the ONNX model into a TensorRT engine. Ultralytics YOLO supports this natively if TensorRT is installed in your Jetson environment.

You can trigger this by simply attempting to export to `engine` format. The system will use the ONNX file implicitly:

```bash
python export_model.py --model best.pt --format engine --half
```
*(This step might take several minutes to compile the engine optimized specifically for your Orin).*

This will produce a `best.engine` file.

## Step 6: Run Inference Scripts

You can now use any of the provided inference scripts.

> **Important Note for Jetson/Orin**: When running inference, if you generated a `.engine` file in Step 5, you should pass `best.engine` instead of `best.pt` to the scripts for maximumFPS. If you skipped Step 5, use `best.onnx`.

### Real-time Monitoring (Camera/Video)

To run the real-time webcam monitor (press 'q' to quit):

```bash
python realtime_monitor.py --model best.engine --source 0
```
*(Replace `0` with the path to a video file if you are testing a video, e.g., `--source video.mp4`)*

### Single Image or Folder Inference

To run inference on a single image and generate a report:

```bash
python model_inference.py --model best.engine --source images/test.jpg --save True
```

