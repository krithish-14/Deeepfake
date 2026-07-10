import torch  # type: ignore
from sklearn.metrics import roc_auc_score, precision_score, recall_score  # type: ignore
from data.dataset_loader import get_dataloaders


def evaluate_real_model(model_path='weights/convnext_best.pth', data_dir='datasets/ffpp'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from models.convnext_detector import ConvNeXtDetector

    import os

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Train the model first or provide correct path.")

    model = ConvNeXtDetector(pretrained=False, num_classes=1).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    try:
        _, val_loader = get_dataloaders(data_dir, batch_size=16)
    except ValueError as e:
        print(f"Dataset error: {e}")
        return

    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs).cpu().view(-1)
            probs = torch.sigmoid(logits)
            preds.extend(probs.tolist())
            targets.extend(labels.view(-1).tolist())

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    auc = roc_auc_score(targets.numpy(), preds.numpy())
    preds_bin = (preds >= 0.5).float()
    precision = precision_score(targets.numpy(), preds_bin.numpy(), zero_division=0)
    recall = recall_score(targets.numpy(), preds_bin.numpy(), zero_division=0)

    print(f'AUC: {auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')


if __name__ == '__main__':
    evaluate_real_model()
