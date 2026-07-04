import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    """
    General dataset loader for image-based deepfake datasets (FF++, Celeb-DF, etc.)
    Assumes a directory structure where images are separated into 'real' and 'fake' folders,
    or a CSV containing image paths and labels.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Simple directory traversal assuming: data_dir/real/ and data_dir/fake/
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                self.samples.append((os.path.join(real_dir, f), 0)) # 0 for real
                
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                self.samples.append((os.path.join(fake_dir, f), 1)) # 1 for fake
        
        # Filter out non-files and common hidden entries
        self.samples = [(p, l) for (p, l) in self.samples if os.path.isfile(p)]

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in '{data_dir}'. Expected subfolders: '{real_dir}' and '{fake_dir}' containing image files."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Fallback if image reading fails (dummy image)
            image = torch.zeros(3, 224, 224).numpy()
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)

def get_transforms(image_size=224):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=32, image_size=224):
    train_transform, val_transform = get_transforms(image_size)
    
    # Mocking train/val split by using same directory for now
    train_dataset = DeepfakeDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = DeepfakeDataset(os.path.join(data_dir, 'val'), transform=val_transform)

    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty. Make sure '{os.path.join(data_dir, 'train')}' contains 'real/' and 'fake/' image folders.")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty. Make sure '{os.path.join(data_dir, 'val')}' contains 'real/' and 'fake/' image folders.")

    # Use fewer workers on Windows to avoid multiprocessing issues; callers may increase if desired
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
