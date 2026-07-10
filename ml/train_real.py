import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext

# mlflow is optional for lightweight runs; training will proceed without it if missing
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    _MLFLOW_AVAILABLE = False

from models.convnext_detector import ConvNeXtDetector
from data.dataset_loader import get_dataloaders


def train_real_model(model_name='convnext', data_dir='datasets/ffpp', epochs=6, batch_size=16, lr=1e-4, save_dir='weights', backbone_name='convnext_large'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if model_name == 'convnext':
        model = ConvNeXtDetector(pretrained=True, num_classes=1, backbone_name=backbone_name).to(device)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    try:
        train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    except ValueError as e:
        print(f"Dataset error: {e}")
        print("Please prepare a dataset with the following layout:\n  <data_dir>/train/real/   (real images)\n  <data_dir>/train/fake/   (fake images)\n  <data_dir>/val/real/     (validation real images)\n  <data_dir>/val/fake/     (validation fake images)")
        return

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f'{model_name}_best.pth')
    backbone_path = os.path.join(save_dir, f'{backbone_name}_best.pth')

    if _MLFLOW_AVAILABLE:
        mlflow.set_experiment('deepfake-detection')

    run_ctx = mlflow.start_run(run_name=f'{model_name}-training') if _MLFLOW_AVAILABLE else nullcontext()
    with run_ctx:
        if _MLFLOW_AVAILABLE:
            mlflow.log_params({'model': model_name, 'epochs': epochs, 'batch_size': batch_size, 'lr': lr, 'data_dir': data_dir})

        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).view_as(labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / max(1, len(train_loader))
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs).view_as(labels)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= max(1, len(val_loader))

            if _MLFLOW_AVAILABLE:
                mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

            print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                torch.save(model.state_dict(), backbone_path)

        if _MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(save_dir)
            except Exception:
                # don't fail training if logging artifacts fails
                pass

    print(f'Training complete. Best weights saved to {best_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a deepfake image detector.')
    parser.add_argument('--model', default='convnext', help='Model architecture to train')
    parser.add_argument('--data-dir', default='datasets/ffpp', help='Root dataset directory')
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', default='weights', help='Directory to save model weights')
    parser.add_argument('--backbone', default='convnext_large', help='timm backbone name to use')
    args = parser.parse_args()

    train_real_model(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        backbone_name=args.backbone,
    )
