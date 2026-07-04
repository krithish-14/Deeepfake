import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow

# Import models (we would dynamically choose in a real setup)
from models.xception_baseline import XceptionBaseline
from models.convnext_detector import ConvNeXtDetector

def train_model(model_name, data_dir, epochs=10, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing {model_name} on {device}...")
    if model_name == 'xception':
        model = XceptionBaseline().to(device)
    elif model_name == 'convnext':
        model = ConvNeXtDetector().to(device)
    else:
        raise ValueError("Unsupported model")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # In a real setup, we use get_dataloaders from dataset_loader
    # from data.dataset_loader import get_dataloaders
    # train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    
    # Mock DataLoader for demonstration
    train_loader = [(torch.randn(batch_size, 3, 224, 224).to(device), torch.randint(0, 2, (batch_size, 1)).float().to(device)) for _ in range(5)]
    
    mlflow.start_run()
    mlflow.log_params({"model": model_name, "epochs": epochs, "batch_size": batch_size, "lr": lr})

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)

    mlflow.end_run()
    
    # Save model
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), f'weights/{model_name}_final.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    # Test run
    train_model('convnext', 'datasets/ffpp')
