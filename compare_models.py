#!/usr/bin/env python3
"""
Model Comparison Tool
Compares multiple models on the same dataset
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Optional tabulate for nice tables
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

import atexit
import gc

@atexit.register
def cleanup():
    """Cleanup CUDA memory on exit to prevent Jetson Orin glibc corruption."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class ModelComparator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.test_images = []
        
    def load_model(self, name, path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Load a model"""
        print(f"Loading model '{name}': {path}")
        try:
            model = YOLO(path)
            if device == 'cuda':
                model.to('cuda')
                try:
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy_img, verbose=False, device=device, half=True)
                except Exception as e:
                    print(f"  [!] WARNING: CUDA architecture unsupported. Falling back to CPU.")
                    device = 'cpu'
                    model.to('cpu')
            self.models[name] = {
                'model': model,
                'path': path,
                'device': device,
                'size_mb': os.path.getsize(path) / (1024 * 1024)
            }
            print(f"  ✓ Size: {self.models[name]['size_mb']:.2f} MB")
            return True
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            return False
    
    def load_test_images(self, test_dir, recursive=True):
        """Load test images"""
        print(f"\nLoading test images from: {test_dir}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        if recursive:
            for ext in image_extensions:
                image_paths.extend(Path(test_dir).rglob(f'*{ext}'))
                image_paths.extend(Path(test_dir).rglob(f'*{ext.upper()}'))
        else:
            for ext in image_extensions:
                image_paths.extend(Path(test_dir).glob(f'*{ext}'))
                image_paths.extend(Path(test_dir).glob(f'*{ext.upper()}'))
        
        self.test_images = [str(p) for p in image_paths]
        print(f"Found {len(self.test_images)} test images")
        
        return self.test_images
    
    def benchmark_model(self, model_name, num_images=100):
        """Benchmark a single model"""
        if model_name not in self.models:
            return None
        
        model_info = self.models[model_name]
        model = model_info['model']
        device = model_info['device']
        
        # Select subset of images
        test_subset = self.test_images[:min(num_images, len(self.test_images))]
        
        print(f"\nBenchmarking '{model_name}' on {len(test_subset)} images...")
        
        results = {
            'model_name': model_name,
            'model_size_mb': model_info['size_mb'],
            'inference_times': [],
            'confidences': [],
            'predictions': [],
            'total_time': 0,
            'fps': 0,
            'memory_usage': []
        }
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        for img_path in tqdm(test_subset, desc=f"  {model_name}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Record memory before
            mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Inference
            infer_start = time.time()
            is_half = (device == 'cuda')
            preds = model.predict(img, verbose=False, augment=False, device=device, half=is_half)
            infer_time = (time.time() - infer_start) * 1000  # ms
            
            # Record memory after
            mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            mem_used = (mem_after - mem_before) / (1024**2)  # MB
            
            results['inference_times'].append(infer_time)
            results['memory_usage'].append(mem_used)
            
            # Get predictions
            if preds and len(preds) > 0 and preds[0].probs is not None:
                probs = preds[0].probs
                top1_idx = probs.top1
                top1_conf = float(probs.data[top1_idx])
                top1_class = preds[0].names[top1_idx]
                
                results['confidences'].append(top1_conf)
                results['predictions'].append({
                    'image': os.path.basename(img_path),
                    'class': top1_class,
                    'confidence': top1_conf
                })
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['fps'] = len(test_subset) / total_time if total_time > 0 else 0
        results['avg_time_ms'] = np.mean(results['inference_times']) if results['inference_times'] else 0
        results['std_time_ms'] = np.std(results['inference_times']) if results['inference_times'] else 0
        results['min_time_ms'] = min(results['inference_times']) if results['inference_times'] else 0
        results['max_time_ms'] = max(results['inference_times']) if results['inference_times'] else 0
        results['avg_confidence'] = np.mean(results['confidences']) if results['confidences'] else 0
        results['peak_memory_mb'] = max(results['memory_usage']) if results['memory_usage'] else 0
        
        self.results[model_name] = results
        
        # Print quick summary
        print(f"  FPS: {results['fps']:.1f} | Avg time: {results['avg_time_ms']:.2f} ms | "
              f"Avg conf: {results['avg_confidence']:.4f}")
        
        return results
    
    def benchmark_all(self, num_images=100):
        """Benchmark all loaded models"""
        print(f"\n{'='*60}")
        print("Benchmarking All Models")
        print(f"{'='*60}")
        
        for model_name in self.models:
            self.benchmark_model(model_name, num_images)
    
    def generate_comparison_report(self, output_dir='./model_comparison'):
        """Generate comparison report"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comparison table
        comparison = []
        for name, res in self.results.items():
            comparison.append({
                'Model': name,
                'Size (MB)': f"{res['model_size_mb']:.2f}",
                'Avg Time (ms)': f"{res['avg_time_ms']:.2f} ± {res['std_time_ms']:.2f}",
                'FPS': f"{res['fps']:.1f}",
                'Peak Memory (MB)': f"{res['peak_memory_mb']:.2f}",
                'Avg Confidence': f"{res['avg_confidence']:.4f}"
            })
        
        df = pd.DataFrame(comparison)
        
        # Print comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        if HAS_TABULATE:
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f'comparison_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved: {csv_path}")
        
        # Generate plots
        self.generate_comparison_plots(output_dir, timestamp)
        
        # Save detailed results (drop raw lists to keep JSON lean)
        exportable = {}
        for name, res in self.results.items():
            exportable[name] = {k: v for k, v in res.items()
                                if k not in ('inference_times', 'memory_usage', 'predictions')}
        
        json_path = os.path.join(output_dir, f'detailed_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(exportable, f, indent=2, default=str)
        print(f"✓ Detailed results: {json_path}")
        
        return df
    
    def generate_comparison_plots(self, output_dir, timestamp):
        """Generate comparison visualizations"""
        model_names = list(self.results.keys())
        
        if not model_names:
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Inference Time Comparison
        ax1 = axes[0, 0]
        times = [self.results[m]['avg_time_ms'] for m in model_names]
        errors = [self.results[m]['std_time_ms'] for m in model_names]
        bars = ax1.bar(model_names, times, yerr=errors, capsize=5, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Average Inference Time')
        ax1.grid(axis='y', alpha=0.3)
        for bar, t in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{t:.1f}ms', ha='center', va='bottom', fontsize=9)
        
        # 2. FPS Comparison
        ax2 = axes[0, 1]
        fps_values = [self.results[m]['fps'] for m in model_names]
        bars = ax2.bar(model_names, fps_values, color='lightgreen', alpha=0.7)
        ax2.set_ylabel('FPS')
        ax2.set_title('Frames Per Second')
        ax2.grid(axis='y', alpha=0.3)
        for bar, f in zip(bars, fps_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{f:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Memory & Model Size Comparison
        ax3 = axes[1, 0]
        memory = [self.results[m]['peak_memory_mb'] for m in model_names]
        size = [self.results[m]['model_size_mb'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, memory, width, label='Peak Memory', color='orange', alpha=0.7)
        bars2 = ax3.bar(x + width/2, size, width, label='Model Size', color='purple', alpha=0.7)
        
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('Memory Usage vs Model Size')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Confidence Comparison
        ax4 = axes[1, 1]
        confidences = [self.results[m]['avg_confidence'] for m in model_names]
        bars = ax4.bar(model_names, confidences, color='coral', alpha=0.7)
        ax4.set_ylabel('Average Confidence')
        ax4.set_title('Prediction Confidence')
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)
        for bar, c in zip(bars, confidences):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{c:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'comparison_plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plots saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Model Comparison Tool')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='List of model paths to compare')
    parser.add_argument('--names', type=str, nargs='+',
                       help='Names for the models (must match number of models)')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output', type=str, default='./model_comparison',
                       help='Output directory for results')
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of images to test per model')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Validate model count
    if args.names and len(args.names) != len(args.models):
        print("Error: Number of model names must match number of model paths")
        return
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load models
    print(f"\n{'='*60}")
    print(f"Loading {len(args.models)} models")
    print(f"{'='*60}")
    
    for i, model_path in enumerate(args.models):
        if not os.path.exists(model_path):
            print(f"Error: Model not found: {model_path}")
            continue
        
        name = args.names[i] if args.names else f"Model_{i+1}"
        comparator.load_model(name, model_path, args.device)
    
    # Load test images
    comparator.load_test_images(args.test_dir)
    
    # Run benchmarks
    comparator.benchmark_all(args.num_images)
    
    # Generate report
    comparator.generate_comparison_report(args.output)
    
    print(f"\n✓ Comparison complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()
