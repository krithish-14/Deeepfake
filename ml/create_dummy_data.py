import cv2
import numpy as np
import os

os.makedirs('dummy_data/images', exist_ok=True)
os.makedirs('dummy_data/videos', exist_ok=True)

# Create 3 dummy images
for i in range(3):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(f'dummy_data/images/test_img_{i}.jpg', img)

# Create 1 dummy video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dummy_data/videos/test_vid_0.mp4', fourcc, 20.0, (224, 224))
for _ in range(30): # 30 frames
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out.write(frame)
out.release()

print("Dummy images and videos created successfully.")
