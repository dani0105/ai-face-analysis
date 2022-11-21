import albumentations as alb
import os, cv2, json
import numpy as np

augmentor = alb.Compose([ alb.HorizontalFlip(p=0.5), 
                          alb.RandomBrightnessContrast(p=0.2),
                          alb.VerticalFlip(p=0.5)])

for class_dir in os.listdir('lightcnn-face-recognition/data/output/train'):
  images = os.listdir('lightcnn-face-recognition/data/output/train/'+class_dir)
  if len(images) < 20:
    for image_file in images:
      img = cv2.imread(os.path.join('lightcnn-face-recognition/data/output/train',class_dir, image_file),0)
      name = image_file.split('.')[0]
      for x in range(0,3):
        augmented = augmentor(image=img)
        cv2.imwrite(os.path.join('lightcnn-face-recognition/data/output/train',class_dir,f'{name}.{x}.jpg'), augmented['image'])