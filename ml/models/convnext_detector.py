import torch
import torch.nn as nn
import timm


class ConvNeXtDetector(nn.Module):
    """ConvNeXt-based image detector for deepfake classification."""

    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=pretrained)

        if hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
            num_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        else:
            raise AttributeError('Unexpected ConvNeXt backbone head structure')

    def forward(self, x):
        logits = self.backbone(x)
        return torch.sigmoid(logits)

if __name__ == '__main__':
    model = ConvNeXtDetector(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"ConvNeXt output shape: {output.shape}")
