"""Create a minimal dataset layout expected by the training scripts.

Creates the following structure under `datasets/ffpp` (relative to the `ml` folder):

datasets/ffpp/train/real/
datasets/ffpp/train/fake/
datasets/ffpp/val/real/
datasets/ffpp/val/fake/

Generates small random images so you can run a smoke-training quickly.
"""
import os
from PIL import Image
import numpy as np

base = os.path.join('datasets', 'ffpp')
paths = [
    os.path.join(base, 'train', 'real'),
    os.path.join(base, 'train', 'fake'),
    os.path.join(base, 'val', 'real'),
    os.path.join(base, 'val', 'fake'),
]

for p in paths:
    os.makedirs(p, exist_ok=True)

# create a few images
def make_images(folder, count=5):
    for i in range(count):
        arr = (np.random.rand(224,224,3) * 255).astype('uint8')
        img = Image.fromarray(arr)
        img.save(os.path.join(folder, f'{os.path.basename(folder)}_{i}.jpg'))

# train: 8 real / 8 fake
make_images(paths[0], count=8)
make_images(paths[1], count=8)
# val: 4 real / 4 fake
make_images(paths[2], count=4)
make_images(paths[3], count=4)

print(f"Minimal dataset created at {base}")
