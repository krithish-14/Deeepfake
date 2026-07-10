import os
import cv2
import torch
from models.xception_baseline import XceptionBaseline
from models.convnext_detector import ConvNeXtDetector
from models.ensemble import MultimodalEnsemble
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def test_inference():
    print("Initializing models (with random weights)...")
    # Setting pretrained=False so it doesn't try to download weights from internet during this test
    # In a real scenario, we would load our trained checkpoints here.
    xception = XceptionBaseline(pretrained=False)
    xception.eval()
    
    convnext = ConvNeXtDetector(pretrained=False)
    convnext.eval()
    
    ensemble = MultimodalEnsemble()
    ensemble.eval()
    
    transform = get_transform()
    
    image_dir = 'dummy_data/images'
    print("\n--- Testing Images ---")
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        tensor = transform(image=image)['image'].unsqueeze(0) # Add batch dimension
        
        with torch.no_grad():
            xcep_logits = xception(tensor)
            conv_logits = convnext(tensor)
            
            xcep_out = torch.sigmoid(xcep_logits)
            conv_out = torch.sigmoid(conv_logits)
            
            # Mocking the video/temporal models for image-only inference
            mock_genconvit = torch.rand(1, 1)
            mock_resnet = torch.rand(1, 1)
            
            ens_out = ensemble(xcep_out, conv_out, mock_genconvit, mock_resnet)
            
        print(f"File: {img_name}")
        print(f"  Xception Confidence: {xcep_out.item():.4f}")
        print(f"  ConvNeXt Confidence: {conv_out.item():.4f}")
        print(f"  Ensemble Confidence: {ens_out.item():.4f}")
        verdict = "DEEPFAKE" if ens_out.item() > 0.5 else "REAL"
        print(f"  Verdict: {verdict}")

if __name__ == '__main__':
    test_inference()
