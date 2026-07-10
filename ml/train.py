
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import sys
from pathlib import Path

# Add parent directory to path to import models and dataset loader
sys.path.insert(0, str(Path(__file__).parent))

from models.xception_baseline import XceptionBaseline
from models.convnext_detector import ConvNeXtDetector
from data.dataset_loader import get_dataloaders

def train_model(model_name, data_dir, epochs=10, batch_size=32, lr=1e-4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing {model_name} on {device}...")
    if model_name == 'xception':
        model = XceptionBaseline().to(device)
    elif model_name == 'convnext':
        model = ConvNeXtDetector().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Load real dataloaders
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    mlflow.start_run()
    mlflow.log_params({
        "model": model_name, 
        "epochs": epochs, 
        "batch_size": batch_size, 
        "lr": lr,
        "device": str(device)
    })

    best_val_loss = float('inf')
    os.makedirs('weights', exist_ok=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
            # Calculate training accuracy
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item()})
            
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                
                # Calculate validation accuracy
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item()})
                
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Log metrics
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
        
        mlflow.log_metrics({
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc
        }, step=epoch)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = f'weights/{model_name}_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"  Best model saved to {best_model_path} (val_loss: {best_val_loss:.4f})")
            mlflow.log_artifact(best_model_path)

    # Save final model
    final_model_path = f'weights/{model_name}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    mlflow.log_artifact(final_model_path)
    mlflow.end_run()

if __name__ == '__main__':
    # Train with your FF++ dataset
    train_model(
        model_name='convnext', 
        data_dir='datasets/ffpp', 
        epochs=20,  # Train longer for better results
        batch_size=16,  # Smaller batch size if you have limited VRAM
        lr=1e-4
    )
