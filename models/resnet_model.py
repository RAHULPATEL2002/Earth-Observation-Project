"""
ResNet18-based PyTorch model for land cover classification.

Provides a pretrained ResNet18 backbone with a custom classification head.
Includes save/load helpers and model summary functionality.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


class ResNetLandCover(nn.Module):
    """
    ResNet18-based model for land cover classification.
    
    Uses a pretrained ResNet18 backbone and replaces the final fully connected
    layer with a custom head for N classes.
    """
    
    def __init__(self, num_classes=11, pretrained=True):
        """
        Initialize ResNet18 model.
        
        Args:
            num_classes: Number of output classes (default: 11 for ESA WorldCover)
            pretrained: Whether to use pretrained weights (default: True)
        """
        super(ResNetLandCover, self).__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get the number of features from the backbone
        self.num_features = resnet.fc.in_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        # Extract features
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def get_model_summary(self):
        """
        Return a string summary of the model architecture.
        
        Returns:
            str: Model summary including layer counts and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
ResNetLandCover Model Summary:
=============================
Backbone: ResNet18 (pretrained)
Number of classes: {self.classifier[-1].out_features}
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}

Architecture:
- Backbone: ResNet18 feature extractor
- Classifier: Linear(512) -> ReLU -> Dropout(0.5) -> Linear({self.classifier[-1].out_features})
"""
        return summary


def save_model(model, filepath, metadata=None):
    """
    Save model checkpoint with optional metadata.
    
    Args:
        model: The model to save
        filepath: Path to save the checkpoint
        metadata: Optional dict of metadata to include
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': 'ResNetLandCover',
        'num_classes': model.classifier[-1].out_features,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, num_classes=None, device='cpu'):
    """
    Load model from checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        num_classes: Number of classes (if None, inferred from checkpoint)
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Infer num_classes if not provided
    if num_classes is None:
        num_classes = checkpoint.get('num_classes', 11)
    
    # Create model
    model = ResNetLandCover(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {filepath}")
    if 'metadata' in checkpoint:
        print(f"Checkpoint metadata: {checkpoint['metadata']}")
    
    return model


if __name__ == '__main__':
    # Test model creation and summary
    model = ResNetLandCover(num_classes=11, pretrained=True)
    print(model.get_model_summary())
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print(f"\nTest forward pass: input shape {x.shape} -> output shape {y.shape}")
