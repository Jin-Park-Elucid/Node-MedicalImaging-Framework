"""
Inference nodes for model prediction.

Provides inference capabilities with batch processing.
"""

import torch
import torch.nn as nn
from typing import Optional
from ...core import BaseNode, NodeRegistry, DataType


@NodeRegistry.register('inference', 'Predictor',
                      description='Run model inference on data')
class PredictorNode(BaseNode):
    """
    Inference node for running predictions.

    Handles batch prediction with proper device management.
    """

    def _setup_ports(self):
        self.add_input('model', DataType.MODEL)
        self.add_input('input', DataType.TENSOR)
        self.add_output('predictions', DataType.TENSOR)
        self.add_output('probabilities', DataType.TENSOR)

    def execute(self) -> bool:
        try:
            model = self.get_input_value('model')
            input_data = self.get_input_value('input')

            if model is None or input_data is None:
                print("Model or input data not provided")
                return False

            device = self._device
            model.to(device)
            model.eval()

            input_data = input_data.to(device)

            with torch.no_grad():
                outputs = model(input_data)

                # Get probabilities
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).long()

            self.set_output_value('predictions', predictions)
            self.set_output_value('probabilities', probabilities)

            return True

        except Exception as e:
            print(f"Error in PredictorNode: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_field_definitions(self):
        return {}


@NodeRegistry.register('inference', 'BatchPredictor',
                      description='Run inference on entire dataloader')
class BatchPredictorNode(BaseNode):
    """Batch inference node."""

    def _setup_ports(self):
        self.add_input('model', DataType.MODEL)
        self.add_input('dataloader', DataType.BATCH)
        self.add_output('all_predictions', DataType.TENSOR)
        self.add_output('all_labels', DataType.TENSOR)
        self.add_output('all_images', DataType.TENSOR)

    def execute(self) -> bool:
        try:
            model = self.get_input_value('model')
            dataloader = self.get_input_value('dataloader')

            if model is None or dataloader is None:
                return False

            device = self._device
            model.to(device)
            model.eval()

            all_predictions = []
            all_labels = []
            all_images = []

            with torch.no_grad():
                for images, labels in dataloader:
                    images_cpu = images.cpu()  # Save before moving to device
                    images = images.to(device)

                    outputs = model(images)
                    predictions = torch.argmax(outputs, dim=1)

                    all_predictions.append(predictions.cpu())
                    all_labels.append(labels.cpu())
                    all_images.append(images_cpu)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_images = torch.cat(all_images, dim=0)

            self.set_output_value('all_predictions', all_predictions)
            self.set_output_value('all_labels', all_labels)
            self.set_output_value('all_images', all_images)

            return True

        except Exception as e:
            print(f"Error in BatchPredictorNode: {e}")
            return False

    def get_field_definitions(self):
        return {}


@NodeRegistry.register('inference', 'MetricsCalculator',
                      description='Calculate segmentation/classification metrics')
class MetricsCalculatorNode(BaseNode):
    """Calculate metrics from predictions and labels."""

    def _setup_ports(self):
        self.add_input('predictions', DataType.TENSOR)
        self.add_input('labels', DataType.TENSOR)
        self.add_output('metrics', DataType.METRICS)

    def execute(self) -> bool:
        try:
            predictions = self.get_input_value('predictions')
            labels = self.get_input_value('labels')

            if predictions is None or labels is None:
                return False

            task_type = self.get_config('task_type', 'classification').lower()
            metrics = {}

            # Accuracy
            correct = (predictions == labels).sum().item()
            total = labels.numel()
            metrics['accuracy'] = correct / total

            # Per-class metrics
            num_classes = int(predictions.max()) + 1
            dice_scores = []
            iou_scores = []

            for cls in range(num_classes):
                pred_cls = (predictions == cls)
                label_cls = (labels == cls)

                tp = (pred_cls & label_cls).sum().item()
                fp = (pred_cls & ~label_cls).sum().item()
                fn = (~pred_cls & label_cls).sum().item()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                # Dice coefficient (same as F1)
                dice = f1

                # IoU (Intersection over Union)
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

                metrics[f'class_{cls}_precision'] = precision
                metrics[f'class_{cls}_recall'] = recall
                metrics[f'class_{cls}_f1'] = f1
                metrics[f'class_{cls}_dice'] = dice
                metrics[f'class_{cls}_iou'] = iou

                dice_scores.append(dice)
                iou_scores.append(iou)

            # Segmentation-specific summary metrics
            if task_type == 'segmentation':
                # Mean Dice and IoU (averaged across classes)
                metrics['mean_dice'] = sum(dice_scores) / len(dice_scores)
                metrics['mean_iou'] = sum(iou_scores) / len(iou_scores)

                # For binary segmentation, highlight foreground metrics
                if num_classes == 2:
                    metrics['foreground_dice'] = dice_scores[1]
                    metrics['foreground_iou'] = iou_scores[1]

            self.set_output_value('metrics', metrics)

            # Print metrics with better formatting
            print("\n" + "="*60)
            if task_type == 'segmentation':
                print("SEGMENTATION METRICS")
            else:
                print("CLASSIFICATION METRICS")
            print("="*60)

            # Print overall metrics first
            if task_type == 'segmentation':
                print(f"\nOverall Metrics:")
                print(f"  Pixel Accuracy: {metrics['accuracy']:.4f}")
                if 'mean_dice' in metrics:
                    print(f"  Mean Dice:      {metrics['mean_dice']:.4f}")
                    print(f"  Mean IoU:       {metrics['mean_iou']:.4f}")
                if 'foreground_dice' in metrics:
                    print(f"\nForeground (Class 1) - Key Metrics:")
                    print(f"  Dice Score:     {metrics['foreground_dice']:.4f}")
                    print(f"  IoU Score:      {metrics['foreground_iou']:.4f}")
                    print(f"  Precision:      {metrics['class_1_precision']:.4f}")
                    print(f"  Recall:         {metrics['class_1_recall']:.4f}")
            else:
                print(f"\nAccuracy: {metrics['accuracy']:.4f}")

            # Print per-class details
            print(f"\nPer-Class Metrics:")
            for cls in range(num_classes):
                print(f"\n  Class {cls}:")
                print(f"    Precision: {metrics[f'class_{cls}_precision']:.4f}")
                print(f"    Recall:    {metrics[f'class_{cls}_recall']:.4f}")
                print(f"    F1/Dice:   {metrics[f'class_{cls}_dice']:.4f}")
                print(f"    IoU:       {metrics[f'class_{cls}_iou']:.4f}")

            print("="*60 + "\n")

            return True

        except Exception as e:
            print(f"Error in MetricsCalculatorNode: {e}")
            return False

    def get_field_definitions(self):
        return {
            'task_type': {
                'type': 'choice',
                'label': 'Task Type',
                'choices': ['classification', 'segmentation'],
                'default': 'classification'
            }
        }
