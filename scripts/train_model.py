#!/usr/bin/env python3
"""
Train ResNet18 model for land cover classification.

Loads train/test splits, builds DataLoaders with augmentations, trains with
validation, computes metrics (macro F1, per-class metrics), and saves
checkpoints and visualizations.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchmetrics import F1Score
from tqdm import tqdm
from datetime import datetime
import sys

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.resnet_model import ResNetLandCover, save_model


class LandCoverDataset(Dataset):
    """Dataset for land cover classification."""
    
    def __init__(self, metadata_df, img_dir, transform=None):
        """
        Initialize dataset.
        
        Args:
            metadata_df: DataFrame with columns: filename, assigned_code, label
            img_dir: Directory containing RGB images
            transform: Optional transform to apply
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Map assigned_code to class index
        unique_codes = sorted(self.metadata['assigned_code'].unique())
        self.code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
        self.idx_to_code = {idx: code for code, idx in self.code_to_idx.items()}
        self.idx_to_label = {
            idx: self.metadata[self.metadata['assigned_code'] == code]['label'].iloc[0]
            for idx, code in self.idx_to_code.items()
        }
        self.num_classes = len(unique_codes)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row['filename']
        assigned_code = row['assigned_code']
        
        # Load image
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (128, 128), (0, 0, 0))
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get class index
        class_idx = self.code_to_idx[assigned_code]
        
        return image, class_idx, filename, assigned_code


def compute_macro_f1(y_true, y_pred, num_classes):
    """
    Compute macro F1 score manually.
    
    Returns per-class precision, recall, F1, and macro F1.
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Per-class metrics
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    macro_f1 = np.mean(f1_scores)
    
    return {
        'per_class_precision': precisions,
        'per_class_recall': recalls,
        'per_class_f1': f1_scores,
        'macro_f1': macro_f1
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, device, num_classes):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels, filenames, codes in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 for land cover classification')
    parser.add_argument('--model', type=str, default='cnn', help='Model type (default: cnn)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--img-dir', type=str, default='Data/rgb', help='Image directory (default: Data/rgb)')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory (default: models)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--use-cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--class-weight', action='store_true', help='Use class weights')
    parser.add_argument('--save-best-only', action='store_true', help='Save only best model')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"[train_model] Using device: {device}")
    
    # Load data
    print("[train_model] Loading train/test splits...")
    train_df = pd.read_csv('Dataset/train_metadata.csv')
    test_df = pd.read_csv('Dataset/test_metadata.csv')
    print(f"[train_model] Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = LandCoverDataset(train_df, args.img_dir, transform=train_transform)
    test_dataset = LandCoverDataset(test_df, args.img_dir, transform=eval_transform)
    
    num_classes = train_dataset.num_classes
    print(f"[train_model] Number of classes: {num_classes}")
    
    # Class weights
    sampler = None
    criterion = nn.CrossEntropyLoss()
    if args.class_weight:
        class_counts = train_df['assigned_code'].value_counts().sort_index()
        total = len(train_df)
        weights = [total / (len(class_counts) * count) for count in class_counts.values]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[train_model] Using class weights: {dict(zip(class_counts.index, weights))}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = ResNetLandCover(num_classes=num_classes, pretrained=True)
    model.to(device)
    print(model.get_model_summary())
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"[train_model] Starting training for {args.epochs} epochs...")
    
    train_losses = []
    val_losses = []
    val_macro_f1s = []
    val_torchmetrics_f1s = []
    best_f1 = 0.0
    best_epoch = 0
    
    # Create runs directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    runs_dir = f'runs/{timestamp}'
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs('Visualizations', exist_ok=True)
    
    # Torchmetrics F1
    f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    
    for epoch in range(args.epochs):
        print(f"\n[train_model] Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_preds, val_labels, val_probs = validate_epoch(model, test_loader, criterion, device, num_classes)
        val_losses.append(val_loss)
        
        # Metrics
        macro_f1_dict = compute_macro_f1(val_labels, val_preds, num_classes)
        macro_f1 = macro_f1_dict['macro_f1']
        
        # Torchmetrics F1
        val_tensor = torch.tensor(val_labels).to(device)
        pred_tensor = torch.tensor(val_preds).to(device)
        torchmetrics_f1 = f1_metric(pred_tensor, val_tensor).item()
        
        val_macro_f1s.append(macro_f1)
        val_torchmetrics_f1s.append(torchmetrics_f1)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Macro F1 (custom): {macro_f1:.4f}")
        print(f"  Val Macro F1 (torchmetrics): {torchmetrics_f1:.4f}")
        
        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            model_path = os.path.join(args.output_dir, f'resnet18_best_f1{best_f1:.3f}.pth')
            os.makedirs(args.output_dir, exist_ok=True)
            save_model(model, model_path, metadata={
                'epoch': epoch,
                'macro_f1': best_f1,
                'torchmetrics_f1': torchmetrics_f1,
                'num_classes': num_classes
            })
            print(f"  [BEST] Saved model to {model_path}")
    
    # Final evaluation on test set
    print("\n[train_model] Final evaluation on test set...")
    model.eval()
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    all_test_filenames = []
    all_test_codes = []
    all_test_true_labels = []
    
    idx_to_code = test_dataset.idx_to_code
    idx_to_label = test_dataset.idx_to_label
    
    with torch.no_grad():
        for images, labels, filenames, codes in tqdm(test_loader, desc="Final eval"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            all_test_probs.extend(probs.cpu().numpy())
            all_test_filenames.extend(filenames)
            all_test_codes.extend(codes.numpy())
            all_test_true_labels.extend([idx_to_label[label] for label in labels.cpu().numpy()])
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'filename': all_test_filenames,
        'true_code': [idx_to_code[idx] for idx in all_test_labels],
        'true_label': all_test_true_labels,
        'pred_code': [idx_to_code[idx] for idx in all_test_preds],
        'pred_label': [idx_to_label[idx] for idx in all_test_preds],
        'probabilities': [json.dumps(probs.tolist()) for probs in all_test_probs]
    })
    predictions_df.to_csv(os.path.join(runs_dir, 'predictions.csv'), index=False)
    predictions_df.to_csv('runs/predictions.csv', index=False)  # Also save to runs/ for downstream scripts
    print(f"[train_model] Saved predictions to runs/predictions.csv")
    
    # Final metrics
    final_macro_f1_dict = compute_macro_f1(all_test_labels, all_test_preds, num_classes)
    final_torchmetrics_f1 = f1_metric(torch.tensor(all_test_preds).to(device), 
                                      torch.tensor(all_test_labels).to(device)).item()
    
    # Save metrics
    metrics = {
        'epochs': args.epochs,
        'best_epoch': best_epoch,
        'best_macro_f1': best_f1,
        'final_macro_f1': final_macro_f1_dict['macro_f1'],
        'final_torchmetrics_f1': final_torchmetrics_f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_macro_f1s': val_macro_f1s,
        'val_torchmetrics_f1s': val_torchmetrics_f1s,
        'per_class_precision': final_macro_f1_dict['per_class_precision'],
        'per_class_recall': final_macro_f1_dict['per_class_recall'],
        'per_class_f1': final_macro_f1_dict['per_class_f1'],
        'class_names': [idx_to_label[i] for i in range(num_classes)],
        'num_classes': num_classes
    }
    
    with open(os.path.join(runs_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_model] Saved metrics to {runs_dir}/metrics.json")
    
    # Visualizations
    print("[train_model] Generating visualizations...")
    
    # 1. Class distribution
    class_counts = train_df['assigned_code'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_counts)), class_counts.values)
    plt.xlabel('Class Index')
    plt.ylabel('Count')
    plt.title('Class Distribution (Training Set)')
    plt.xticks(range(len(class_counts)), [idx_to_label[i] for i in range(num_classes)], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Visualizations/class_distribution.png', dpi=150)
    plt.close()
    
    # 2. Loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_macro_f1s, label='Macro F1 (custom)')
    plt.plot(val_torchmetrics_f1s, label='Macro F1 (torchmetrics)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Visualizations/loss_accuracy.png', dpi=150)
    plt.close()

    # 3. Confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[idx_to_label[i] for i in range(num_classes)],
                yticklabels=[idx_to_label[i] for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('Visualizations/confusion_matrix.png', dpi=150)
    plt.close()
    
    # 4. Precision/Recall boxplot
    precisions = final_macro_f1_dict['per_class_precision']
    recalls = final_macro_f1_dict['per_class_recall']
    f1s = final_macro_f1_dict['per_class_f1']
    
    plt.figure(figsize=(12, 6))
    data_to_plot = [precisions, recalls, f1s]
    labels = ['Precision', 'Recall', 'F1']
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score Distribution per Class')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('Visualizations/precision_recall_boxplot.png', dpi=150)
    plt.close()
    
    print(f"[train_model] Training complete!")
    print(f"  Final Macro F1 (custom): {final_macro_f1_dict['macro_f1']:.4f}")
    print(f"  Final Macro F1 (torchmetrics): {final_torchmetrics_f1:.4f}")
    print(f"  Best model saved: {model_path}")
    print(f"  Metrics saved: {runs_dir}/metrics.json")


if __name__ == '__main__':
    main()
