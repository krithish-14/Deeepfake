import torch
import torch.nn as nn
import timm

class XceptionBaseline(nn.Module):
    """
    Xception CNN baseline model fine-tuned for Deepfake detection.
    Often used as a baseline on FaceForensics++.
    """
    def __init__(self, pretrained=True, num_classes=1):
        super(XceptionBaseline, self).__init__()
        # Load pre-trained Xception from timm
        # Using a model close to original Xception
        self.model = timm.create_model('xception', pretrained=pretrained)
        
        # Replace the classifier head for binary classification (Real vs Fake)
        num_features = self.model.get_classifier().in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # x is expected to be [batch_size, 3, 299, 299] for Xception
        logits = self.model(x)
        # Return probability of being a deepfake (class 1)
        return torch.sigmoid(logits)

if __name__ == '__main__':
    # Quick test
    model = XceptionBaseline(pretrained=False)
    dummy_input = torch.randn(2, 3, 299, 299)
    output = model(dummy_input)
    print(f"Xception output shape: {output.shape}")
