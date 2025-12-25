"""
Model Architecture for Sugarcane Disease Detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
import config


class SugarcaneDiseaseClassifier(nn.Module):
    """CNN Model for Sugarcane Disease Classification"""

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True, backbone='resnet50'):
        """
        Args:
            num_classes: Number of disease classes
            pretrained: Whether to use pretrained weights
            backbone: Backbone architecture ('resnet50', 'resnet18', 'efficientnet_b0', etc.)
        """
        super(SugarcaneDiseaseClassifier, self).__init__()

        self.backbone_name = backbone

        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer

        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, 3, H, W]

        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x):
        """Extract features for visualization/analysis"""
        return self.backbone(x)


class CustomCNN(nn.Module):
    """Custom CNN architecture (lighter weight alternative)"""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(model_type='resnet50', num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Factory function to create model

    Args:
        model_type: Type of model ('resnet50', 'resnet18', 'efficientnet_b0', 'custom')
        num_classes: Number of output classes
        pretrained: Use pretrained weights

    Returns:
        model: PyTorch model
    """
    if model_type == 'custom':
        model = CustomCNN(num_classes=num_classes)
    else:
        model = SugarcaneDiseaseClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=model_type
        )

    return model.to(config.DEVICE)


if __name__ == '__main__':
    # Test model creation
    model = create_model('resnet50')
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(config.DEVICE)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
