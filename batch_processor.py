#!/usr/bin/env python3
"""
Batch Image Processor for YOLO Model
Processes multiple images and saves results with metadata
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

class BatchProcessor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.results = []
        self.processing_time = 0
        
    def load_model(self):
        """Load model"""
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        if self.device == 'cuda':
            self.model.to('cuda')
            try:
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model.predict(dummy_img, verbose=False)
            except Exception as e:
                print(f"[!] WARNING: CUDA architecture unsupported. Falling back to CPU.\n")
                self.device = 'cpu'
                self.model.to('cpu')
        print("Model loaded successfully")
    
    def process_image(self, image_path, save_visualization=False, output_dir=None):
        """Process single image"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Get original dimensions
        h, w = img.shape[:2]
        
        # Run inference
        start_time = time.time()
        results = self.model.predict(img, verbose=False, augment=False)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Parse results
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                # Get top predictions
                top5_indices = probs.data.topk(5).indices.tolist()
                top5_confidences = probs.data.topk(5).values.tolist()
                top5_classes = [results[0].names[idx] for idx in top5_indices]
                
                result = {
                    'image_path': image_path,
                    'filename': os.path.basename(image_path),
                    'image_width': w,
                    'image_height': h,
                    'inference_time_ms': inference_time,
                    'top1_class': top5_classes[0],
                    'top1_confidence': top5_confidences[0],
                    'top2_class': top5_classes[1] if len(top5_classes) > 1 else None,
                    'top2_confidence': top5_confidences[1] if len(top5_confidences) > 1 else None,
                    'top3_class': top5_classes[2] if len(top5_classes) > 2 else None,
                    'top3_confidence': top5_confidences[2] if len(top5_confidences) > 2 else None,
                    'top4_class': top5_classes[3] if len(top5_classes) > 3 else None,
                    'top4_confidence': top5_confidences[3] if len(top5_confidences) > 3 else None,
                    'top5_class': top5_classes[4] if len(top5_classes) > 4 else None,
                    'top5_confidence': top5_confidences[4] if len(top5_confidences) > 4 else None,
                    'all_classes': top5_classes,
                    'all_confidences': top5_confidences
                }
                
                # Save visualization if requested
                if save_visualization and output_dir:
                    self.save_visualization(img, result, output_dir)
                
                return result
        
        return None
    
    def save_visualization(self, img, result, output_dir):
        """Save image with prediction overlay"""
        # Create class-specific folder
        class_dir = os.path.join(output_dir, result['top1_class'])
        os.makedirs(class_dir, exist_ok=True)
        
        # Create copy for visualization
        vis_img = img.copy()
        
        # Add prediction text
        text = f"{result['top1_class']}: {result['top1_confidence']:.3f}"
        cv2.putText(vis_img, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save without prefix, inside class folder
        filename = os.path.basename(result['image_path'])
        output_path = os.path.join(class_dir, filename)
        cv2.imwrite(output_path, vis_img)
    
    def process_batch(self, input_path, recursive=True, save_visualizations=False, 
                      output_dir='./batch_output'):
        """Process multiple images"""
        print(f"\nProcessing images from: {input_path}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, 'visualizations') if save_visualizations else None
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        if recursive:
            for ext in image_extensions:
                image_paths.extend(Path(input_path).rglob(f'*{ext}'))
                image_paths.extend(Path(input_path).rglob(f'*{ext.upper()}'))
        else:
            for ext in image_extensions:
                image_paths.extend(Path(input_path).glob(f'*{ext}'))
                image_paths.extend(Path(input_path).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in {input_path}")
            return None
        
        print(f"Found {len(image_paths)} images")
        
        # Process images
        start_time = time.time()
        
        for img_path in tqdm(image_paths, desc="Processing"):
            result = self.process_image(img_path, save_visualizations, vis_dir)
            if result:
                self.results.append(result)
        
        self.processing_time = time.time() - start_time
        
        print(f"\n✓ Processed {len(self.results)} images")
        print(f"✓ Total time: {self.processing_time:.2f}s")
        if self.results:
            print(f"✓ Average time: {self.processing_time*1000/len(self.results):.2f} ms per image")
        
        return self.results
    
    def save_results(self, output_dir='./batch_output'):
        """Save results to various formats"""
        if not self.results:
            print("No results to save.")
            return {}

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f'results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ CSV saved: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(output_dir, f'results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON saved: {json_path}")
        
        # Save summary statistics
        summary = {
            'total_images': len(self.results),
            'total_time_seconds': self.processing_time,
            'avg_time_ms': self.processing_time * 1000 / len(self.results) if self.results else 0,
            'fps': len(self.results) / self.processing_time if self.processing_time > 0 else 0,
            'class_distribution': df['top1_class'].value_counts().to_dict(),
            'avg_confidence': df['top1_confidence'].mean(),
            'min_confidence': df['top1_confidence'].min(),
            'max_confidence': df['top1_confidence'].max(),
            'std_confidence': df['top1_confidence'].std()
        }
        
        summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_path}")
        
        # Generate plots
        self.generate_plots(df, output_dir, timestamp)
        
        return {
            'csv': csv_path,
            'json': json_path,
            'summary': summary_path
        }
    
    def generate_plots(self, df, output_dir, timestamp):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Class distribution
        ax1 = axes[0, 0]
        class_counts = df['top1_class'].value_counts()
        ax1.barh(class_counts.index, class_counts.values, color='skyblue')
        ax1.set_xlabel('Count')
        ax1.set_title('Predicted Class Distribution')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Confidence distribution
        ax2 = axes[0, 1]
        ax2.hist(df['top1_confidence'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Score Distribution')
        ax2.grid(alpha=0.3)
        
        # 3. Inference time distribution
        ax3 = axes[1, 0]
        ax3.hist(df['inference_time_ms'], bins=30, color='green', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Inference Time Distribution')
        ax3.grid(alpha=0.3)
        
        # 4. Confidence vs Image resolution
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['image_width'] * df['image_height'], 
                             df['top1_confidence'], 
                             c=df['inference_time_ms'], 
                             cmap='viridis', alpha=0.6, s=30)
        ax4.set_xlabel('Image Resolution (pixels)')
        ax4.set_ylabel('Confidence')
        ax4.set_title('Confidence vs Image Resolution')
        plt.colorbar(scatter, ax=ax4, label='Time (ms)')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'plots_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plots saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Batch Image Processor for YOLO Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to best.pt model file')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', type=str, default='./batch_output',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--recursive', action='store_true',
                       help='Search images recursively')
    parser.add_argument('--save-vis', action='store_true',
                       help='Save visualizations with predictions')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        return
    
    processor = BatchProcessor(args.model, args.device)
    processor.load_model()
    processor.process_batch(args.input, args.recursive, args.save_vis, args.output)
    processor.save_results(args.output)
    
    print(f"\n✓ All results saved to: {args.output}")

if __name__ == "__main__":
    main()
