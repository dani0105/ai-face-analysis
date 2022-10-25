# -*- coding: utf-8 -*-
"""
this code splits the built dataset into train set and valid set.

@author: yhiro
"""

import argparse
import os
import numpy as np
import shutil


dataset_dir="result-misc/train"
out_dir="result-misc-split"
ratio=0.98

print(dataset_dir)

dirs = os.listdir(dataset_dir)
train_count = int(len(dirs) * ratio)
valid_count = len(dirs) - train_count

shuffled = np.random.permutation(len(dirs))

if os.path.exists(out_dir):
    raise Exception('out dir already exists. clear it first!')
    
os.mkdir(out_dir)

try:
    train_dir = os.path.join(out_dir, 'train')
    os.mkdir(train_dir)

    for i in range(train_count):
      path = dataset_dir+"/"+dirs[shuffled[i]]
      print(path)
      target =os.path.join(train_dir, os.path.basename(dirs[shuffled[i]]))
      print(target)
      shutil.copyfile(path, target)

    valid_dir = os.path.join(out_dir, 'valid')
    os.mkdir(valid_dir)
    for i in range(train_count, len(dirs)):
      path = os.path.join(dataset_dir, dirs[shuffled[i]])
      
      target = os.path.join(valid_dir, os.path.basename(dirs[shuffled[i]]))
      shutil.copyfile(path, target)

except OSError as e:
    e.args += ('maybe running as privileged mode would solve this error.',)
    raise e