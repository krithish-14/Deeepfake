import os
import csv
import cv2
import torch
import numpy as np
from urllib.request import urlopen
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class DeepfakeDataset(Dataset):
    """
    General dataset loader for image-based deepfake datasets (FF++, Celeb-DF, etc.)
    Supports two formats:
    1. Directory structure: data_dir/real/ and data_dir/fake/
    2. CSV manifest: CSV file with 'image_path' (or 'image_url') and 'label_numeric' (or 'label') columns
    """
    def __init__(self, data_source, transform=None, is_csv=False):
        self.data_source = data_source
        self.transform = transform
        self.samples = []
        
        if is_csv:
            # Load from CSV
            df = pd.read_csv(data_source)
            
            # Determine image path column
            image_col = None
            for col in ['image_path', 'image_url']:
                if col in df.columns:
                    image_col = col
                    break
            
            # Determine label column
            label_col = None
            for col in ['label_numeric', 'label']:
                if col in df.columns:
                    label_col = col
                    break
            
            if image_col is None or label_col is None:
                raise ValueError(
                    f"CSV must contain image path/url column ('image_path' or 'image_url') "
                    f"and label column ('label_numeric' or 'label')"
                )
            
            for _, row in df.iterrows():
                img_path = row[image_col]
                if label_col == 'label':
                    label = 1 if str(row[label_col]).strip().lower() in ['fake', '1'] else 0
                else:
                    label = int(row[label_col])
                self.samples.append((img_path, label))
        else:
            # Load from directory structure: data_dir/real/ and data_dir/fake/
            real_dir = os.path.join(data_source, 'real')
            fake_dir = os.path.join(data_source, 'fake')
            
            if os.path.exists(real_dir):
                for f in os.listdir(real_dir):
                    self.samples.append((os.path.join(real_dir, f), 0)) # 0 for real
                    
            if os.path.exists(fake_dir):
                for f in os.listdir(fake_dir):
                    self.samples.append((os.path.join(fake_dir, f), 1)) # 1 for fake
            
            # Filter out non-files and common hidden entries
            self.samples = [(p, l) for (p, l) in self.samples if os.path.isfile(p)]

        if len(self.samples) == 0:
            if is_csv:
                raise ValueError(f"No samples found in CSV file '{data_source}'.")
            else:
                raise ValueError(
                    f"No images found in '{data_source}'. Expected subfolders: '{real_dir}' and '{fake_dir}' containing image files."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            if img_path.startswith(('http://', 'https://')):
                # Load from URL
                resp = urlopen(img_path, timeout=10)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            else:
                # Load from local file
                image = cv2.imread(img_path)
            
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Image is None")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback if image loading fails (dummy image)
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
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

def get_dataloaders(data_source, batch_size=32, image_size=224, is_csv=False, csv_split_column=None, num_workers=0):
    train_transform, val_transform = get_transforms(image_size)
    
    if is_csv and csv_split_column:
        # Load from single CSV and split using dataset_split column
        df = pd.read_csv(data_source)
        train_df = df[df[csv_split_column] == 'train']
        val_df = df[df[csv_split_column] == 'val']
        
        # Save temporary CSVs for train and val
        import tempfile
        train_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        val_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        train_csv.close()
        val_csv.close()
        
        train_dataset = DeepfakeDataset(train_csv.name, transform=train_transform, is_csv=True)
        val_dataset = DeepfakeDataset(val_csv.name, transform=val_transform, is_csv=True)
    elif is_csv:
        # Separate CSVs for train and val (data_source is directory containing train.csv and val.csv)
        train_csv = os.path.join(data_source, 'train.csv')
        val_csv = os.path.join(data_source, 'val.csv')
        train_dataset = DeepfakeDataset(train_csv, transform=train_transform, is_csv=True)
        val_dataset = DeepfakeDataset(val_csv, transform=val_transform, is_csv=True)
    else:
        # Directory structure: data_dir/train and data_dir/val
        train_dataset = DeepfakeDataset(os.path.join(data_source, 'train'), transform=train_transform)
        val_dataset = DeepfakeDataset(os.path.join(data_source, 'val'), transform=val_transform)

    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError(f"Validation dataset is empty.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
