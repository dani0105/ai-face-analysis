# -*- coding: utf-8 -*-
"""
this code builds a dataset for celeb_gen.py

@author: yhiro
"""
import os
import random
from keras.preprocessing import image

celeb_dir="lightcnn-face-recognition/data/input"
clean70k_list="lightcnn-face-recognition/data/input/images.txt"
out_dir="lightcnn-face-recognition/data/output"

with open(clean70k_list, 'r') as f:
    lines = f.readlines()

label_img = {}
for line in lines:
    img_path, label = line.split()
    if label not in label_img:
        label_img[label] = []
    label_img[label].append(img_path)

if os.path.exists(out_dir):
    raise Exception('out dir already exists. clear it first!')
    
os.mkdir(out_dir)

train_dir = os.path.join(out_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(out_dir, 'test')
os.mkdir(test_dir)

for i, label in enumerate(label_img.keys()):
    
    img_pathes = label_img[label]
    if len(img_pathes) < 2:
        continue
    
    random.shuffle(img_pathes)
    
    train_save_dir = os.path.join(train_dir, label)
    os.mkdir(train_save_dir)
    test_save_dir = os.path.join(test_dir, label)
    os.mkdir(test_save_dir)
                
    for ii, img_path in enumerate(img_pathes):
        fname = os.path.basename(img_path)
        
        img = image.load_img(celeb_dir+"/"+img_path, grayscale=True, target_size=(144,144))
        if ii == 0:
            img.save(os.path.join(test_save_dir, fname))
        else:
            img.save(os.path.join(train_save_dir, fname))
            
    if (i+1) % 100 == 0:
        print('{} labels done.'.format(i+1))
