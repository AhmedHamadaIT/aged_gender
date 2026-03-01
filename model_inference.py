#!/usr/bin/env python3
"""
Model Inference and Performance Analysis Script
For use with fine-tuned YOLO classification model
"""

import os
import sys
import time
import json
import argparse
import platform
import psutil
import pandas as pd
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime
import humanize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import atexit
import gc

@atexit.register
def cleanup():
    """Cleanup CUDA memory on exit to prevent Jetson Orin glibc corruption."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Try importing ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# Try importing psutil for system monitoring
try:
    import psutil
except ImportError:
    os.system('pip install psutil')
    import psutil

# Constants
ALL_8_CLASSES = [
    "Female_Child", "Female_YoungAdult", "Female_MiddleAged", "Female_OldAged",
    "Male_Child", "Male_YoungAdult", "Male_MiddleAged", "Male_OldAged"
]

class ModelPerformanceAnalyzer:
    """Main class for model inference and performance analysis"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.model_size_mb = 0
        self.results = {
            'model_name': os.path.basename(model_path),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device': device,
            'system_info': {},
            'performance': {},
            'inference_stats': {},
            'classification_report': {},
            'errors': []
        }
        
    def load_model(self):
        """Load the YOLO model and measure its size"""
        print(f"\n{'='*60}")
        print(f"Loading model: {self.model_path}")
        print(f"{'='*60}")
        
        try:
            # Load model
            start_time = time.time()
            self.model = YOLO(self.model_path)
            load_time = time.time() - start_time
            
            # Get model size
            self.model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            
            print(f"✓ Model loaded successfully in {load_time:.2f}s")
            print(f"✓ Model size: {self.model_size_mb:.2f} MB")
            print(f"✓ Device: {self.device}")
            
            # Move model to device if needed
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
                print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
                
                try:
                    debug_device = next(self.model.model.parameters()).device
                    print(f"✓ Model Device inside PyTorch: {debug_device}")
                except Exception:
                    pass
                
                # Warmup / Test CUDA architecture support
                try:
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    self.model.predict(dummy_img, verbose=False, device=self.device, half=True)
                except Exception as e:
                    print(f"\n[!] WARNING: CUDA error detected (likely unsupported old GPU architecture).")
                    print(f"[!] Falling back to CPU...\n")
                    self.device = 'cpu'
                    self.model.to('cpu')
                    self.results['device'] = 'cpu'
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {str(e)}")
            self.results['errors'].append(f"Model loading error: {str(e)}")
            return False
    
    def get_system_info(self):
        """Gather system information"""
        print("\n" + "-"*40)
        print("System Information")
        print("-"*40)
        
        # CPU info
        cpu_info = {
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_physical': psutil.cpu_count(logical=False),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A',
            'cpu_percent': psutil.cpu_percent(interval=1),
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            'total_ram_gb': memory.total / (1024**3),
            'available_ram_gb': memory.available / (1024**3),
            'used_percent': memory.percent
        }
        
        # GPU info if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) / (1024**3),
                'gpu_memory_cached': torch.cuda.memory_reserved(0) / (1024**3),
                'cuda_version': torch.version.cuda
            }
        
        # Python/OS info
        system_info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cpu': cpu_info,
            'memory': memory_info,
            'gpu': gpu_info
        }
        
        self.results['system_info'] = system_info
        
        # Print summary
        print(f"OS: {system_info['os']} {system_info['os_version']}")
        print(f"CPU: {cpu_info['cpu_count']} cores ({cpu_info['cpu_physical']} physical)")
        print(f"RAM: {memory_info['total_ram_gb']:.2f} GB (Available: {memory_info['available_ram_gb']:.2f} GB)")
        if gpu_info:
            print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_total']:.2f} GB)")
        
        return system_info
    
    def benchmark_single_image(self, image_path, warmup=False):
        """Benchmark inference on a single image"""
        if not self.model:
            return None
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Measure inference time and memory
        torch.cuda.reset_peak_memory_stats() if self.device == 'cuda' else None
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
        
        # Run inference
        is_half = (self.device == 'cuda')
        results = self.model.predict(img, verbose=False, augment=False, device=self.device, half=is_half)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
        
        # Get predictions
        if results and len(results) > 0:
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = float(probs.data[top1_idx])
            top1_class = results[0].names[top1_idx]
            
            # Get top-5 predictions
            top5_indices = probs.data.topk(5).indices.tolist()
            top5_confidences = probs.data.topk(5).values.tolist()
            top5_classes = [results[0].names[idx] for idx in top5_indices]
        else:
            top1_class = "Unknown"
            top1_conf = 0
            top5_classes = []
            top5_confidences = []
        
        # Calculate metrics
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        if self.device == 'cuda':
            memory_used = (end_memory - start_memory) / (1024**2)  # MB
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            memory_used = 0
            peak_memory = 0
        
        return {
            'inference_time_ms': inference_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': peak_memory,
            'predicted_class': top1_class,
            'confidence': top1_conf,
            'top5_classes': top5_classes,
            'top5_confidences': top5_confidences
        }
    
    def benchmark_batch(self, image_paths, batch_size=32):
        """Benchmark batch inference"""
        if not self.model:
            return None
        
        print(f"\nBatch inference with size={batch_size}...")
        
        # Prepare batches
        batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
        
        total_time = 0
        total_images = 0
        all_times = []
        
        torch.cuda.reset_peak_memory_stats() if self.device == 'cuda' else None
        start_memory = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
        
        for batch in tqdm(batches, desc="Processing batches"):
            batch_start = time.time()
            
            # Run batch inference
            is_half = (self.device == 'cuda')
            results = self.model.predict(batch, batch=len(batch), verbose=False, augment=False, device=self.device, half=is_half)
            
            batch_end = time.time()
            batch_time = (batch_end - batch_start) * 1000  # ms
            
            total_time += batch_time
            total_images += len(batch)
            all_times.append(batch_time / len(batch))  # per-image time
        
        end_memory = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if self.device == 'cuda' else 0
        
        avg_time_per_image = total_time / total_images
        avg_batch_time = total_time / len(batches)
        fps = 1000 / avg_time_per_image
        
        memory_used = (end_memory - start_memory) / (1024**2) if self.device == 'cuda' else 0
        
        return {
            'total_images': total_images,
            'total_batches': len(batches),
            'avg_time_per_image_ms': avg_time_per_image,
            'avg_batch_time_ms': avg_batch_time,
            'fps': fps,
            'memory_used_mb': memory_used,
            'peak_memory_mb': peak_memory,
            'all_times_ms': all_times
        }
    
    def run_inference_on_folder(self, folder_path, recursive=True, save_vis=False, output_dir=None):
        """Run inference on all images in a folder"""
        print(f"\n{'='*60}")
        print(f"Running inference on folder: {folder_path}")
        print(f"{'='*60}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        if recursive:
            for ext in image_extensions:
                image_paths.extend(Path(folder_path).rglob(f'*{ext}'))
                image_paths.extend(Path(folder_path).rglob(f'*{ext.upper()}'))
        else:
            for ext in image_extensions:
                image_paths.extend(Path(folder_path).glob(f'*{ext}'))
                image_paths.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in {folder_path}")
            return None
        
        print(f"Found {len(image_paths)} images")
        
        # Run inference on all images
        results = []
        class_counts = defaultdict(int)
        confidence_scores = []
        
        # Single image benchmark (first image)
        if image_paths:
            print("\n" + "-"*40)
            print("Single Image Benchmark")
            print("-"*40)
            
            single_result = self.benchmark_single_image(image_paths[0])
            if single_result:
                print(f"Inference time: {single_result['inference_time_ms']:.2f} ms")
                print(f"Memory used: {single_result['memory_used_mb']:.2f} MB")
                print(f"Peak memory: {single_result['peak_memory_mb']:.2f} MB")
                print(f"Prediction: {single_result['predicted_class']} (conf: {single_result['confidence']:.4f})")
                
                # Top-5
                print("\nTop-5 predictions:")
                for i, (cls, conf) in enumerate(zip(single_result['top5_classes'], 
                                                     single_result['top5_confidences'])):
                    print(f"  {i+1}. {cls}: {conf:.4f}")
        
        # Batch benchmark
        if len(image_paths) >= 4:
            print("\n" + "-"*40)
            print("Batch Benchmark")
            print("-"*40)
            
            batch_result = self.benchmark_batch(image_paths[:min(100, len(image_paths))], batch_size=32)
            if batch_result:
                print(f"Average time per image: {batch_result['avg_time_per_image_ms']:.2f} ms")
                print(f"FPS: {batch_result['fps']:.2f}")
                print(f"Peak memory: {batch_result['peak_memory_mb']:.2f} MB")
                
                self.results['performance']['batch_benchmark'] = batch_result
        
        # Full inference on all images
        print("\n" + "-"*40)
        print("Full Dataset Inference")
        print("-"*40)
        
        vis_dir = None
        if save_vis and output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = os.path.join(output_dir, f'processed_images_{timestamp}')
            os.makedirs(vis_dir, exist_ok=True)
            print(f"Saving visualized images to: {vis_dir}")

        total_start = time.time()
        total_memory_start = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
        
        cpu_usages = []
        ram_usages = []
        gpu_usages = []
        
        for img_path in tqdm(image_paths, desc="Inferencing"):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Record system stats
            cpu_usages.append(psutil.cpu_percent(interval=None))
            ram_usages.append(psutil.virtual_memory().percent)
            if self.device == 'cuda' and torch.cuda.is_available():
                try:
                    gpu_usages.append(torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0)
                except:
                    gpu_usages.append(0)
            
            # Run inference
            is_half = (self.device == 'cuda')
            results_obj = self.model.predict(img, verbose=False, augment=False, device=self.device, half=is_half)
            
            if results_obj and len(results_obj) > 0:
                probs = results_obj[0].probs
                top1_idx = probs.top1
                top1_conf = float(probs.data[top1_idx])
                top1_class = results_obj[0].names[top1_idx]
                
                # Get top-5
                top5_indices = probs.data.topk(5).indices.tolist()
                top5_confidences = probs.data.topk(5).values.tolist()
                top5_classes = [results_obj[0].names[idx] for idx in top5_indices]
                
                class_counts[top1_class] += 1
                confidence_scores.append(top1_conf)
                
                results.append({
                    'image': os.path.basename(img_path),
                    'path': img_path,
                    'predicted_class': top1_class,
                    'confidence': top1_conf,
                    'top5': top5_classes,
                    'top5_confidences': top5_confidences
                })
                
                # Save visualization
                if vis_dir:
                    vis_img = img.copy()
                    
                    # Create class-specific subfolder if it doesn't exist
                    class_dir = os.path.join(vis_dir, top1_class)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Add only top prediction
                    text = f"{top1_class}: {top1_conf:.3f}"
                    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imwrite(os.path.join(class_dir, os.path.basename(img_path)), vis_img)
        
        total_end = time.time()
        total_memory_end = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
        
        total_time = total_end - total_start
        avg_time_per_image = total_time * 1000 / len(results) if results else 0
        fps = len(results) / total_time if total_time > 0 else 0
        
        memory_used = (total_memory_end - total_memory_start) / (1024**2) if self.device == 'cuda' else 0
        
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0
        avg_ram = np.mean(ram_usages) if ram_usages else 0
        avg_gpu = np.mean(gpu_usages) if gpu_usages else 0
        
        print(f"\n✓ Processed {len(results)} images")
        print(f"✓ Total time: {total_time:.2f} s")
        print(f"✓ Average time: {avg_time_per_image:.2f} ms per image")
        print(f"✓ FPS: {fps:.2f}")
        print(f"✓ Memory used: {memory_used:.2f} MB")
        print(f"✓ Avg CPU Usage: {avg_cpu:.1f}%")
        print(f"✓ Avg RAM Usage: {avg_ram:.1f}%")
        if self.device == 'cuda':
            print(f"✓ Avg GPU Util: {avg_gpu:.1f}%")
        
        # Store results
        inference_stats = {
            'total_images': len(results),
            'total_time_seconds': total_time,
            'avg_time_ms': avg_time_per_image,
            'fps': fps,
            'memory_used_mb': memory_used,
            'avg_cpu_usage_percent': avg_cpu,
            'avg_ram_usage_percent': avg_ram,
            'avg_gpu_utilization_percent': avg_gpu
        }
        
        self.results['inference_stats'] = inference_stats
        self.results['classification_report'] = {
            'class_distribution': dict(class_counts),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': min(confidence_scores) if confidence_scores else 0,
            'max_confidence': max(confidence_scores) if confidence_scores else 0,
            'std_confidence': np.std(confidence_scores) if confidence_scores else 0
        }
        
        return results
    
    def generate_report(self, output_dir='./reports'):
        """Generate comprehensive performance report"""
        print(f"\n{'='*60}")
        print("Generating Performance Report")
        print(f"{'='*60}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Add performance summary
        self.results['performance']['model_size_mb'] = self.model_size_mb
        self.results['performance']['num_classes'] = len(ALL_8_CLASSES)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = os.path.join(output_dir, f'report_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate Markdown report
        md_path = os.path.join(output_dir, f'report_{timestamp}.md')
        self._generate_markdown_report(md_path)
        
        # Generate HTML report
        html_path = os.path.join(output_dir, f'report_{timestamp}.html')
        self._generate_html_report(html_path)
        
        # Generate visualizations
        viz_path = os.path.join(output_dir, f'visualizations_{timestamp}.png')
        self._generate_visualizations(viz_path)
        
        print(f"\n✓ Reports saved to: {output_dir}")
        print(f"  - JSON: {os.path.basename(json_path)}")
        print(f"  - Markdown: {os.path.basename(md_path)}")
        print(f"  - HTML: {os.path.basename(html_path)}")
        print(f"  - Visualizations: {os.path.basename(viz_path)}")
        
        return json_path
    
    def _generate_markdown_report(self, output_path):
        """Generate markdown format report"""
        with open(output_path, 'w') as f:
            f.write(f"# Model Performance Report\n\n")
            f.write(f"**Model:** {self.results['model_name']}\n")
            f.write(f"**Date:** {self.results['timestamp']}\n")
            f.write(f"**Device:** {self.results['device']}\n\n")
            
            f.write("## System Information\n\n")
            sys_info = self.results['system_info']
            f.write(f"- **OS:** {sys_info.get('os', 'N/A')} {sys_info.get('os_version', '')}\n")
            f.write(f"- **CPU:** {sys_info.get('cpu', {}).get('cpu_count', 'N/A')} cores\n")
            f.write(f"- **RAM:** {sys_info.get('memory', {}).get('total_ram_gb', 0):.2f} GB\n")
            if sys_info.get('gpu'):
                f.write(f"- **GPU:** {sys_info['gpu'].get('gpu_name', 'N/A')}\n")
                f.write(f"- **GPU Memory:** {sys_info['gpu'].get('gpu_memory_total', 0):.2f} GB\n\n")
            
            f.write("## Model Information\n\n")
            f.write(f"- **Model Size:** {self.results['performance'].get('model_size_mb', 0):.2f} MB\n")
            f.write(f"- **Number of Classes:** {self.results['performance'].get('num_classes', 0)}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("### Single Image Inference\n")
            perf = self.results.get('performance', {})
            batch = perf.get('batch_benchmark', {})
            if batch:
                f.write(f"- **Average Time:** {batch.get('avg_time_per_image_ms', 0):.2f} ms\n")
                f.write(f"- **FPS:** {batch.get('fps', 0):.2f}\n")
                f.write(f"- **Peak Memory:** {batch.get('peak_memory_mb', 0):.2f} MB\n\n")
            
            f.write("### Batch Inference\n")
            stats = self.results.get('inference_stats', {})
            f.write(f"- **Total Images:** {stats.get('total_images', 0)}\n")
            f.write(f"- **Total Time:** {stats.get('total_time_seconds', 0):.2f} s\n")
            f.write(f"- **Average Time:** {stats.get('avg_time_ms', 0):.2f} ms\n")
            f.write(f"- **FPS:** {stats.get('fps', 0):.2f}\n")
            f.write(f"- **Memory Used:** {stats.get('memory_used_mb', 0):.2f} MB\n")
            f.write(f"- **Avg CPU Usage:** {stats.get('avg_cpu_usage_percent', 0):.1f}%\n")
            f.write(f"- **Avg RAM Usage:** {stats.get('avg_ram_usage_percent', 0):.1f}%\n")
            if stats.get('avg_gpu_utilization_percent', 0) > 0:
                f.write(f"- **Avg GPU Util:** {stats.get('avg_gpu_utilization_percent', 0):.1f}%\n")
            f.write("\n")
            
            f.write("## Classification Statistics\n\n")
            cls_report = self.results.get('classification_report', {})
            f.write(f"- **Average Confidence:** {cls_report.get('avg_confidence', 0):.4f}\n")
            f.write(f"- **Min Confidence:** {cls_report.get('min_confidence', 0):.4f}\n")
            f.write(f"- **Max Confidence:** {cls_report.get('max_confidence', 0):.4f}\n")
            f.write(f"- **Std Confidence:** {cls_report.get('std_confidence', 0):.4f}\n\n")
            
            f.write("### Class Distribution\n\n")
            f.write("| Class | Count |\n")
            f.write("|-------|-------|\n")
            for cls, count in cls_report.get('class_distribution', {}).items():
                f.write(f"| {cls} | {count} |\n")
    
    def _generate_html_report(self, output_path):
        """Generate HTML format report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #3498db; color: white; padding: 12px; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                .metric {{ font-weight: bold; color: #27ae60; }}
                .section {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Model Performance Report</h1>
            <p><strong>Model:</strong> {self.results['model_name']}</p>
            <p><strong>Date:</strong> {self.results['timestamp']}</p>
            <p><strong>Device:</strong> {self.results['device']}</p>
            
            <div class="section">
                <h2>System Information</h2>
                <table>
                    <tr><th>Component</th><th>Specification</th></tr>
                    <tr><td>OS</td><td>{self.results['system_info'].get('os', 'N/A')}</td></tr>
                    <tr><td>CPU</td><td>{self.results['system_info'].get('cpu', {}).get('cpu_count', 'N/A')} cores</td></tr>
                    <tr><td>RAM</td><td>{self.results['system_info'].get('memory', {}).get('total_ram_gb', 0):.2f} GB</td></tr>
        """
        
        if self.results['system_info'].get('gpu'):
            html_content += f"""
                    <tr><td>GPU</td><td>{self.results['system_info']['gpu'].get('gpu_name', 'N/A')}</td></tr>
                    <tr><td>GPU Memory</td><td>{self.results['system_info']['gpu'].get('gpu_memory_total', 0):.2f} GB</td></tr>
            """
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Model Size</td><td>{self.results['performance'].get('model_size_mb', 0):.2f} MB</td></tr>
                    <tr><td>Number of Classes</td><td>{self.results['performance'].get('num_classes', 0)}</td></tr>
                    <tr><td>Average Inference Time</td><td>{self.results.get('inference_stats', {}).get('avg_time_ms', 0):.2f} ms</td></tr>
                    <tr><td>FPS</td><td>{self.results.get('inference_stats', {}).get('fps', 0):.2f}</td></tr>
                    <tr><td>Peak Memory Usage</td><td>{self.results.get('performance', {}).get('batch_benchmark', {}).get('peak_memory_mb', 0):.2f} MB</td></tr>
                    <tr><td>Avg CPU Usage</td><td>{self.results.get('inference_stats', {}).get('avg_cpu_usage_percent', 0):.1f}%</td></tr>
                    <tr><td>Avg RAM Usage</td><td>{self.results.get('inference_stats', {}).get('avg_ram_usage_percent', 0):.1f}%</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Classification Statistics</h2>
                <table>
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Average Confidence</td><td>{self.results.get('classification_report', {}).get('avg_confidence', 0):.4f}</td></tr>
                    <tr><td>Min Confidence</td><td>{self.results.get('classification_report', {}).get('min_confidence', 0):.4f}</td></tr>
                    <tr><td>Max Confidence</td><td>{self.results.get('classification_report', {}).get('max_confidence', 0):.4f}</td></tr>
                    <tr><td>Std Confidence</td><td>{self.results.get('classification_report', {}).get('std_confidence', 0):.4f}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_visualizations(self, output_path):
        """Generate performance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Model Performance Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Class Distribution
        ax1 = axes[0, 0]
        cls_dist = self.results.get('classification_report', {}).get('class_distribution', {})
        if cls_dist:
            classes = list(cls_dist.keys())
            counts = list(cls_dist.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            bars = ax1.barh(classes, counts, color=colors)
            ax1.set_xlabel('Count')
            ax1.set_title('Class Distribution')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{count}', va='center')
        
        # 2. Confidence Distribution
        ax2 = axes[0, 1]
        avg_conf = self.results.get('classification_report', {}).get('avg_confidence', 0)
        classes = list(cls_dist.keys()) if cls_dist else []
        conf_values = [avg_conf] * len(classes)
        ax2.bar(classes, conf_values, color='orange', alpha=0.7)
        ax2.set_ylabel('Confidence')
        ax2.set_title('Average Confidence by Class')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Performance Metrics Gauge
        ax3 = axes[1, 0]
        metrics = ['FPS', 'Memory (MB)', 'Time (ms)']
        perf_vals = [
            self.results.get('inference_stats', {}).get('fps', 0),
            self.results.get('performance', {}).get('batch_benchmark', {}).get('peak_memory_mb', 0),
            self.results.get('inference_stats', {}).get('avg_time_ms', 0)
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax3.bar(metrics, perf_vals, color=colors)
        ax3.set_title('Key Performance Indicators')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, perf_vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 4. System Resources
        ax4 = axes[1, 1]
        sys_info = self.results.get('system_info', {})
        
        resources = ['CPU Usage', 'RAM Usage', 'GPU Memory']
        if sys_info.get('gpu'):
            usage_values = [
                sys_info.get('cpu', {}).get('cpu_percent', 0),
                sys_info.get('memory', {}).get('used_percent', 0),
                (sys_info['gpu'].get('gpu_memory_allocated', 0) / 
                 sys_info['gpu'].get('gpu_memory_total', 1)) * 100 if sys_info.get('gpu') else 0
            ]
        else:
            usage_values = [
                sys_info.get('cpu', {}).get('cpu_percent', 0),
                sys_info.get('memory', {}).get('used_percent', 0),
                0
            ]
        
        # Avoid pie chart with all zeros
        if sum(usage_values) > 0:
            wedges, texts, autotexts = ax4.pie(usage_values, labels=resources, autopct='%1.1f%%',
                                                startangle=90, colors=['#3498db', '#2ecc71', '#e74c3c'])
        else:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('System Resource Usage')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='YOLO Model Performance Analyzer')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the model file (.pt, .onnx, .engine)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or folder containing images')
    parser.add_argument('--output', type=str, default='./reports',
                       help='Output directory for reports (default: ./reports)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search images recursively in input folder')
    parser.add_argument('--save-vis', action='store_true',
                       help='Save annotated images to the report output directory')
    parser.add_argument('--format', type=str, choices=['json', 'md', 'html', 'all'],
                       default='all', help='Report format (default: all)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = ModelPerformanceAnalyzer(args.model, device=args.device)
    
    # Load model
    if not analyzer.load_model():
        sys.exit(1)
    
    # Get system info
    analyzer.get_system_info()
    
    # Run inference
    if os.path.isfile(args.input):
        # Single image
        print(f"\nProcessing single image: {args.input}")
        result = analyzer.benchmark_single_image(args.input)
        if result:
            print(f"\nResults:")
            print(f"  Predicted: {result['predicted_class']} (conf: {result['confidence']:.4f})")
            print(f"  Time: {result['inference_time_ms']:.2f} ms")
            print(f"  Memory: {result['memory_used_mb']:.2f} MB")
            
            # Update results structure
            analyzer.results['inference_stats'] = {
                'total_images': 1,
                'total_time_seconds': result['inference_time_ms'] / 1000,
                'avg_time_ms': result['inference_time_ms'],
                'fps': 1000 / result['inference_time_ms'],
                'memory_used_mb': result['memory_used_mb']
            }
            
            analyzer.results['classification_report'] = {
                'class_distribution': {result['predicted_class']: 1},
                'avg_confidence': result['confidence'],
                'min_confidence': result['confidence'],
                'max_confidence': result['confidence'],
                'std_confidence': 0
            }
    else:
        # Folder of images
        results = analyzer.run_inference_on_folder(
            args.input, 
            recursive=args.recursive,
            save_vis=args.save_vis,
            output_dir=args.output
        )
    
    # Generate report
    analyzer.generate_report(args.output)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
