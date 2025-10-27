"""
Evaluation Script for YOLO-UDD v2.0
Comprehensive model evaluation on test set
"""

import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_yolo_udd
from data.dataset import create_dataloaders
from utils.metrics import compute_metrics, measure_fps, MetricsCalculator


class Evaluator:
    """
    Evaluator for YOLO-UDD v2.0
    """
    
    def __init__(self, weights_path, data_dir, save_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        self.model = build_yolo_udd(num_classes=3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load data
        print(f"Loading data from {data_dir}...")
        self.dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=1,  # Use batch size 1 for evaluation
            num_workers=4,
            img_size=640
        )
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(num_classes=3)
    
    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate model on test set
        """
        print("\n" + "=" * 80)
        print("Starting Evaluation")
        print("=" * 80)
        
        self.metrics_calc.reset()
        all_turbidity_scores = []
        
        pbar = tqdm(self.dataloaders['test'], desc='Evaluating')
        for batch in pbar:
            images = batch['images'].to(self.device)
            bboxes = batch['bboxes']
            labels = batch['labels']
            
            # Forward pass
            predictions, turb_score = self.model(images)
            
            # Store turbidity scores
            all_turbidity_scores.append(turb_score.cpu().numpy())
            
            # Update metrics
            self.metrics_calc.update(predictions, list(zip(bboxes, labels)))
        
        # Compute final metrics
        metrics = self.metrics_calc.compute()
        
        # Measure FPS
        print("\nMeasuring inference speed...")
        fps = measure_fps(self.model, device=self.device)
        
        # Compute average turbidity
        import numpy as np
        avg_turbidity = np.mean(all_turbidity_scores)
        
        # Print results
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"mAP@50: {metrics['map50']:.4f}")
        print(f"mAP@50:95: {metrics['map']:.4f}")
        print(f"mAP@75: {metrics['map75']:.4f}")
        print(f"FPS: {fps:.2f}")
        print(f"Average Turbidity Score: {avg_turbidity:.4f}")
        print("=" * 80)
        
        # Save results
        results = {
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'mAP@50': float(metrics['map50']),
            'mAP@50:95': float(metrics['map']),
            'mAP@75': float(metrics['map75']),
            'FPS': float(fps),
            'avg_turbidity': float(avg_turbidity),
            'num_images': len(self.dataloaders['test'].dataset)
        }
        
        results_file = self.save_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {results_file}")
        
        return results
    
    def compare_with_baseline(self):
        """
        Compare with baseline performance
        """
        results = self.evaluate()
        
        # Baseline from Section 2
        baseline_map = 0.759
        target_map = 0.82
        
        improvement = results['mAP@50:95'] - baseline_map
        target_achievement = (results['mAP@50:95'] / target_map) * 100
        
        print("\n" + "=" * 80)
        print("Comparison with Baseline")
        print("=" * 80)
        print(f"Baseline (YOLOv9c): {baseline_map:.4f} mAP@50:95")
        print(f"YOLO-UDD v2.0: {results['mAP@50:95']:.4f} mAP@50:95")
        print(f"Improvement: {improvement:.4f} ({improvement/baseline_map*100:.2f}%)")
        print(f"Target (>82%): {target_map:.4f}")
        print(f"Target Achievement: {target_achievement:.2f}%")
        print("=" * 80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate YOLO-UDD v2.0')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--save-dir', type=str, default='runs/eval',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Compare with baseline performance')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create evaluator
    evaluator = Evaluator(
        weights_path=args.weights,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=args.device
    )
    
    # Run evaluation
    if args.compare_baseline:
        evaluator.compare_with_baseline()
    else:
        evaluator.evaluate()


if __name__ == '__main__':
    main()
