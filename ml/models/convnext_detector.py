import torch
import torch.nn as nn
import timm


class ConvNeXtDetector(nn.Module):
    """ConvNeXt-based image detector for deepfake classification."""

    def __init__(self, pretrained=True, num_classes=1, backbone_name='convnext_base'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self._replace_classifier(num_classes)

    def _replace_classifier(self, num_classes):
        if hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
            num_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        elif hasattr(self.backbone, 'fc'):
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            raise AttributeError('Unexpected backbone head structure')

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    model = ConvNeXtDetector(pretrained=False, backbone_name='convnext_base')
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"ConvNeXt output shape: {output.shape}")
