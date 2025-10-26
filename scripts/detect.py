"""
Inference Script for YOLO-UDD v2.0
Run detection on images or videos
"""

import os
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_yolo_udd


class Detector:
    """
    Detector for running inference with YOLO-UDD v2.0
    """
    
    def __init__(self, weights_path, conf_thresh=0.25, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thresh = conf_thresh
        
        # Load model
        print(f"Loading model from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        self.model = build_yolo_udd(num_classes=3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = ['Trash', 'Animal', 'ROV']
        self.colors = [
            (0, 255, 0),   # Green for Trash
            (255, 0, 0),   # Blue for Animal
            (0, 0, 255)    # Red for ROV
        ]
    
    def preprocess(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: numpy array [H, W, 3]
        
        Returns:
            tensor: [1, 3, 640, 640]
        """
        # Resize to 640x640
        img = cv2.resize(image, (640, 640))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor [1, 3, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img.to(self.device)
    
    @torch.no_grad()
    def detect(self, image):
        """
        Run detection on a single image
        
        Args:
            image: numpy array [H, W, 3] in BGR format
        
        Returns:
            detections: List of (bbox, conf, class, turbidity)
        """
        # Preprocess
        img_tensor = self.preprocess(image)
        
        # Forward pass
        predictions, turb_score = self.model(img_tensor)
        
        # Post-process predictions (simplified)
        # In actual implementation, would decode YOLO outputs and apply NMS
        detections = []
        
        # Dummy detections for demonstration
        # Replace with actual decoding logic
        
        return detections, turb_score.item()
    
    def draw_detections(self, image, detections, turbidity):
        """
        Draw bounding boxes on image
        
        Args:
            image: numpy array [H, W, 3]
            detections: List of (bbox, conf, class)
            turbidity: Turbidity score
        
        Returns:
            image: Image with drawn boxes
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # Draw turbidity indicator
        turb_text = f"Turbidity: {turbidity:.3f}"
        cv2.putText(img, turb_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        
        # Draw each detection
        for bbox, conf, cls in detections:
            x1, y1, x2, y2 = bbox
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            
            # Draw box
            color = self.colors[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[cls]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        return img
    
    def detect_image(self, image_path, save_path=None):
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to image file
            save_path: Path to save result (optional)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Detect
        detections, turbidity = self.detect(image)
        
        # Draw results
        result = self.draw_detections(image, detections, turbidity)
        
        # Save or display
        if save_path:
            cv2.imwrite(str(save_path), result)
            print(f"Saved result to {save_path}")
        else:
            cv2.imshow('YOLO-UDD Detection', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections, turbidity
    
    def detect_folder(self, source_dir, save_dir):
        """
        Detect objects in all images in a folder
        
        Args:
            source_dir: Directory containing images
            save_dir: Directory to save results
        """
        source_dir = Path(source_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in source_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc='Processing images'):
            save_path = save_dir / img_path.name
            self.detect_image(img_path, save_path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO-UDD v2.0 Inference')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or directory')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                        help='Directory to save results')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create detector
    detector = Detector(
        weights_path=args.weights,
        conf_thresh=args.conf_thresh,
        device=args.device
    )
    
    # Check if source is file or directory
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single image
        save_path = Path(args.save_dir) / source_path.name
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        detector.detect_image(source_path, save_path)
    elif source_path.is_dir():
        # Directory of images
        detector.detect_folder(source_path, args.save_dir)
    else:
        print(f"Error: {args.source} is not a valid file or directory")


if __name__ == '__main__':
    main()
