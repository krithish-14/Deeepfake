import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBiLSTM(nn.Module):
    """
    ResNet-Swish-BiLSTM hybrid for identifying temporal artifacts in video sequences.
    Processes sequences of frames.
    """
    def __init__(self, sequence_length=10, hidden_dim=256, num_layers=2, num_classes=1):
        super(ResNetBiLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Feature Extractor (ResNet18 as an example, using Swish/SiLU activation is a modern choice)
        resnet = models.resnet18(pretrained=True)
        # Remove the fully connected layer
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        # We can optionally replace ReLU with SiLU (Swish) in ResNet here if strictly adhering to paper,
        # but for baseline, standard ResNet works well.
        
        # BiLSTM for temporal sequence
        resnet_out_features = resnet.fc.in_features
        self.lstm = nn.LSTM(
            input_size=resnet_out_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x is expected to be [batch_size, sequence_length, C, H, W]
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten sequence into batch dimension for feature extraction
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.feature_extractor(x) # [batch_size * seq_len, features, 1, 1]
        features = features.view(batch_size, seq_len, -1) # [batch_size, seq_len, features]
        
        # Temporal analysis
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :] # [batch_size, hidden_dim * 2]
        
        # Classify
        logits = self.fc(last_out)
        return logits  # Return raw logits for BCEWithLogitsLoss compatibility

if __name__ == '__main__':
    model = ResNetBiLSTM()
    dummy_input = torch.randn(2, 10, 3, 224, 224) # Batch=2, Seq=10
    output = model(dummy_input)
    print(f"ResNet+BiLSTM output shape: {output.shape}")
