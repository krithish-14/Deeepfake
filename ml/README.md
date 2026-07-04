# Deepfake Detection ML Pipeline

## Quick start

1. Install dependencies
   - pip install -r requirements.txt
2. Train the image detector
   - python train_real.py --data-dir datasets/ffpp --epochs 10 --batch-size 16 --lr 1e-4
3. Evaluate the detector
   - python evaluate_real.py --model-path weights/convnext_best.pth --data-dir datasets/ffpp
4. Use the model from the backend
   - The Django backend can call the trained weights from `ml/weights/convnext_best.pth`.

## Notes
- The current training code expects a dataset directory with `train/` and `val/` subfolders containing `real/` and `fake/` image folders.
- For video deepfake detection, train with frame-extracted images or extend the model with a video back-end.
- For best accuracy, use FF++, Celeb-DF-v2, and DFDC data with proper labels.

## Training command examples

- Train with default FF++ data layout:
  - `python train_real.py --data-dir datasets/ffpp --epochs 10 --batch-size 16`
- Train with a custom dataset path:
  - `python train_real.py --data-dir /path/to/deepfake-dataset --epochs 12 --batch-size 8`
- Save weights to a custom directory:
  - `python train_real.py --save-dir my_weights`
