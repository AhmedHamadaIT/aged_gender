#!/usr/bin/env python3
"""
Real-time Performance Monitor for YOLO Model
Shows live FPS, memory usage, and predictions
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
import psutil
from ultralytics import YOLO
from datetime import datetime
import argparse
from collections import deque
from pathlib import Path

class RealTimeMonitor:
    def __init__(self, model_path, source=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_path = model_path
        self.source = source
        self.device = device
        self.model = None
        self.fps_buffer = deque(maxlen=30)
        self.memory_buffer = deque(maxlen=30)
        self.running = False
        self.frame_count = 0
        self.start_time = None
        
    def load_model(self):
        """Load YOLO model"""
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
        
    def get_system_stats(self):
        """Get current system stats"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        stats = {
            'cpu': cpu_percent,
            'ram': memory_percent,
            'gpu': 0,
            'gpu_mem': 0
        }
        
        if torch.cuda.is_available():
            stats['gpu'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats['gpu_mem'] = (allocated / total) * 100 if total > 0 else 0
            
        return stats
    
    def draw_stats(self, frame, stats, prediction):
        """Draw statistics on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # FPS
        fps = np.mean(self.fps_buffer) if self.fps_buffer else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # System stats
        cv2.putText(frame, f"CPU: {stats['cpu']:.1f}%", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"RAM: {stats['ram']:.1f}%", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if stats['gpu'] > 0:
            cv2.putText(frame, f"GPU: {stats['gpu']:.1f}%", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"GPU Mem: {stats['gpu_mem']:.1f}%", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Prediction
        if prediction:
            cls_name = prediction['class']
            conf = prediction['confidence']
            
            # Top prediction
            cv2.putText(frame, f"Pred: {cls_name}", (20, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Conf: {conf:.3f}", (20, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Top-5
            if 'top5' in prediction:
                y_offset = h - 150
                cv2.putText(frame, "Top-5:", (w - 250, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                for i, (cls, conf_val) in enumerate(zip(prediction['top5'][:5], 
                                                    prediction['top5_conf'][:5])):
                    cv2.putText(frame, f"{i+1}. {cls}: {conf_val:.3f}", 
                               (w - 250, y_offset + 25 + i*20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Frame count and time
        if self.start_time:
            elapsed = time.time() - self.start_time
            cv2.putText(frame, f"Frames: {self.frame_count} | Time: {elapsed:.1f}s",
                       (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process single frame"""
        # Measure inference time
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(frame, verbose=False, augment=False)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        self.fps_buffer.append(1000 / inference_time if inference_time > 0 else 0)
        
        # Get prediction
        prediction = None
        if results and len(results) > 0:
            probs = results[0].probs
            if probs is not None:
                top1_idx = probs.top1
                top1_conf = float(probs.data[top1_idx])
                top1_class = results[0].names[top1_idx]
                
                # Get top-5
                top5_indices = probs.data.topk(5).indices.tolist()
                top5_confidences = probs.data.topk(5).values.tolist()
                top5_classes = [results[0].names[idx] for idx in top5_indices]
                
                prediction = {
                    'class': top1_class,
                    'confidence': top1_conf,
                    'top5': top5_classes,
                    'top5_conf': top5_confidences
                }
        
        return prediction
    
    def run(self):
        """Main loop"""
        self.load_model()
        
        # Open video source
        if isinstance(self.source, str) and self.source.isdigit():
            cap = cv2.VideoCapture(int(self.source))
        else:
            cap = cv2.VideoCapture(self.source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("\n" + "="*50)
        print("Real-Time Monitor Started")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'p' - Pause/Resume")
        print("="*50 + "\n")
        
        self.running = True
        self.start_time = time.time()
        paused = False
        frame = None
        
        while self.running:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Get system stats
                stats = self.get_system_stats()
                
                # Process frame
                prediction = self.process_frame(frame)
                
                # Draw stats
                frame = self.draw_stats(frame, stats, prediction)
                
                # Show frame
                cv2.imshow('YOLO Real-Time Monitor', frame)
                
                self.frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                if frame is not None:
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\nSession Summary:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Total time:   {elapsed:.2f}s")
        print(f"  Average FPS:  {avg_fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Real-time YOLO Model Monitor')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to best.pt model file')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    monitor = RealTimeMonitor(args.model, args.source, args.device)
    monitor.run()

if __name__ == "__main__":
    main()
