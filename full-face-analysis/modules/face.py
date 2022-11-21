import cv2 
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torch
from keras.preprocessing.image import image as keras_image
import torch.nn.functional as F

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
PERSONS=['abdou','Atal Bihari','Daniel Rojas','George W Bush','Andre Agassi','Amelie Mauresmo', 'Anna Kournikova', 'Arnold Schwarzenegger', 'Abel','Enoc Castro']

tensor_transform =  transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

class Face:
  
  def __init__( self, face_image, x1, y1, x2, y2, conf):
    self.image=face_image
    self.x1 = int(x1)
    self.y1 = int(y1)
    self.x2 = int(x2)
    self.y2 = int(y2)
    self.conf = float(conf)
    self.emotion = False
    self.recognition = False
    self.ncrops = 0

  def get_image_vgg(self):
    image = self.image.copy()
    image = cv2.resize(image,(48,48) )

    image = image[:, :, np.newaxis]
    image = np.concatenate((image, image, image), axis=2)
    image = Image.fromarray(image)
    inputs = tensor_transform(image)
    
    self.ncrops,c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)

    return Variable(inputs,volatile=True)

  def get_image_ligthcnn(self):
    image = self.image.copy()
    image = Image.fromarray(image)
    image = image.convert('L')
    image = image.resize((128,128), Image.NEAREST)
    x = keras_image.img_to_array(image, data_format='channels_last')
    batch_x = np.zeros((1,) + (128,128,1), dtype='float32')
    batch_x[0] = x
    batch_x /= 255.
    return batch_x
  
  def save_emotion(self,outputs):
    self.emotion = True

    outputs_avg = outputs.view(self.ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg).data.cpu().numpy()

    _, predicted = torch.max(outputs_avg.data, 0)
    
    id = int(predicted.cpu().numpy())
    self.emotion_result =  { 
      'class_id':id, 
      'class': EMOTIONS[id],
      'confidence': float(score[id]),
      "values":{
        "angry":float(score[0]),
        "disgust":float(score[1]),
        "fear":float(score[2]),
        "happy":float(score[3]),
        "sad":float(score[4]),
        "surprise":float(score[5]),
        "neutral":float(score[6]),
      }
    }

  def save_person(self,model_result):
    self.recognition = True
    model_result = int(model_result)
    class_name = ""
    if len(PERSONS) > model_result:
      class_name = PERSONS[model_result]

    self.person_result = {
      'class_name': class_name,
      'class_id':model_result
    }

  def get_result(self):
    result = {
      "face":{
        "box":{
          "x1":self.x1,
          "y1":self.y1,
          "x2":self.x2,
          "y2":self.y2
        },
        "confidence": self.conf
      }
    }

    if self.emotion:
      result['emotion']= self.emotion_result

    if self.recognition:
      result['recognition']= self.person_result
    
    return result


