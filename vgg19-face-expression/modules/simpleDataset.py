from __future__ import print_function
from concurrent.futures.thread import _worker
from distutils import core
from PIL import Image
import numpy as np
import torch.utils.data as data

class ValidationSet(data.Dataset):

  def __init__(self, transform, data):
    self.images = data['data_pixel']
    self.targets = data['data_label']
    self.transform = transform
  
  def __getitem__(self, index):
    img = self.images[index]
    target = self.targets[index]

    img = img[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    img = self.transform(img)
    return img, target

  def __len__(self):
    return len(self.images)