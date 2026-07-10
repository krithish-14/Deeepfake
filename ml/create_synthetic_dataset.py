from PIL import Image
import os

base = 'd:/Deep-Fake Project/ml/datasets/ffpp'
for split in ['train', 'val']:
    for label in ['real', 'fake']:
        folder = os.path.join(base, split, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(4):
            color = (255, 0, 0) if label == 'fake' else (0, 255, 0)
            img = Image.new('RGB', (224, 224), color=color)
            img.save(os.path.join(folder, f'{label}_{i}.jpg'))

print('created synthetic images')
