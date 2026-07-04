import torch
import torch.nn as nn


class MultimodalEnsemble(nn.Module):
    """Weighted ensemble that fuses multiple deepfake detector outputs."""

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.weights = nn.Parameter(torch.tensor([0.2, 0.35, 0.3, 0.15], dtype=torch.float32))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, xception_score, convnext_score, genconvit_score, resnet_score):
        scores = torch.cat([xception_score, convnext_score, genconvit_score, resnet_score], dim=1)
        normalized_weights = torch.softmax(self.weights, dim=0)
        ensemble_score = torch.sum(scores * normalized_weights.view(1, -1, 1) if scores.dim() == 3 else scores * normalized_weights.view(1, -1), dim=1, keepdim=True)
        return torch.clamp(ensemble_score, 0.0, 1.0)

if __name__ == '__main__':
    ensemble = MultimodalEnsemble()
    # Dummy scores (batch_size=2)
    s1 = torch.tensor([[0.8], [0.1]])
    s2 = torch.tensor([[0.9], [0.2]])
    s3 = torch.tensor([[0.85], [0.15]])
    s4 = torch.tensor([[0.7], [0.3]])
    
    out = ensemble(s1, s2, s3, s4)
    print(f"Ensemble output: \n{out}")
