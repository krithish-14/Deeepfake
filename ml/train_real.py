import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast

# mlflow is optional for lightweight runs; training will proceed without it if missing
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    _MLFLOW_AVAILABLE = False

from models.convnext_detector import ConvNeXtDetector
from data.dataset_loader import get_dataloaders


def train_real_model(
    model_name='convnext',
    data_dir='datasets/ffpp',
    epochs=6,
    batch_size=16,
    lr=1e-4,
    save_dir='weights',
    backbone_name='convnext_base',  # Use smaller backbone by default for efficiency
    is_csv=False,
    csv_split_column=None,
    num_workers=0,
    use_amp=True,
    gradient_accumulation_steps=1,
    image_size=224,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using AMP: {use_amp}')
    print(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    print(f'Number of workers: {num_workers}')

    if model_name == 'convnext':
        model = ConvNeXtDetector(pretrained=True, num_classes=1, backbone_name=backbone_name).to(device)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    try:
        train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size, is_csv=is_csv, csv_split_column=csv_split_column, image_size=image_size, num_workers=num_workers)
        
        # Cosine annealing learning rate scheduler (now that we have train_loader)
        total_steps = epochs * len(train_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    except ValueError as e:
        print(f"Dataset error: {e}")
        if is_csv:
            print("Please prepare a CSV dataset with 'image_path'/'image_url' and 'label_numeric'/'label' columns.")
            if csv_split_column:
                print(f"CSV should also have '{csv_split_column}' column with 'train'/'val' values.")
        else:
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
            mlflow.log_params({
                'model': model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'data_dir': data_dir,
                'backbone': backbone_name,
                'use_amp': use_amp,
                'gradient_accumulation': gradient_accumulation_steps,
                'image_size': image_size,
            })

        best_val_loss = float('inf')
        global_step = 0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp and device.type == 'cuda'):
                    outputs = model(inputs).view_as(labels)
                    loss = criterion(outputs, labels) / gradient_accumulation_steps
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                
                running_loss += loss.item() * gradient_accumulation_steps

            train_loss = running_loss / max(1, len(train_loader))
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast(enabled=use_amp and device.type == 'cuda'):
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
                print(f'Saved best model at epoch {epoch + 1} (val_loss: {best_val_loss:.4f})')

        if _MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(save_dir)
            except Exception:
                # don't fail training if logging artifacts fails
                pass

    print(f'Training complete. Best weights saved to {best_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a deepfake image detector efficiently.')
    parser.add_argument('--model', default='convnext', help='Model architecture to train')
    parser.add_argument('--data-dir', default='datasets/ffpp', help='Root dataset directory or CSV file path')
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', default='weights', help='Directory to save model weights')
    parser.add_argument('--backbone', default='convnext_base', help='timm backbone name to use (convnext_tiny/convnext_base for efficiency)')
    parser.add_argument('--is-csv', action='store_true', help='Whether dataset is CSV-based')
    parser.add_argument('--csv-split-column', default=None, help='Column name in CSV to use for train/val split')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers (use 0 on Windows)')
    parser.add_argument('--disable-amp', action='store_true', help='Disable automatic mixed precision training')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    args = parser.parse_args()

    train_real_model(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        backbone_name=args.backbone,
        is_csv=args.is_csv,
        csv_split_column=args.csv_split_column,
        num_workers=args.num_workers,
        use_amp=not args.disable_amp,
        gradient_accumulation_steps=args.gradient_accumulation,
        image_size=args.image_size,
    )
