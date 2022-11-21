import cv2 
import numpy as np
from base64 import b64decode
from . import utils



class Package:
  
  def __init__( self, 
      socket_id: str, 
      image, 
      analyse_person: bool = False, 
      analyse_emotion: bool = False ):
    self.analyse_person=analyse_person
    self.analyse_emotion=analyse_emotion
    self.image=image
    self.socket_id=socket_id   

  def decode_image(self):
    image_bytes = b64decode(self.image.split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    self.image = cv2.imdecode(jpg_as_np, flags=1)

  def get_retinaface_image(self):
    img = np.float32(self.image.copy())
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad_params = utils.pad_input_image(img, max_steps=max([8, 16, 32]))
    return img, pad_params
  
  def get_image_size(self):
    return self.image.shape
  
  def to_grayscale(self):
    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

  def save_image(self, image,name='test.jpg'):
    cv2.imwrite(name,image)

