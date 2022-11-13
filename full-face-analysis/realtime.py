from models import vgg, lightcnn, retinaface
import yaml
import cv2 
import tensorflow as tf
import numpy as np
import torch
from modules.utils import pad_input_image, recover_pad_output
from torchvision import transforms
from torch.autograd import Variable
from skimage.transform import resize
from PIL import Image
from matplotlib import cm
import torch.nn.functional as F
from keras.preprocessing.image import image as keras_image

## constants

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

PERSONS=['Daniel Rojas','Atal Bihari Vajpayee','George W Bush','Andre Agassi','Amelie Mauresmo', 'Enoc', 'Persona 1', 'Kendall', 'Persona 2']


# Methods
def load_yaml(path):  
  with open(path, 'r') as f:
    loaded = yaml.load(f, Loader=yaml.Loader)
  return loaded

def processEmotion(image, model):
    image = cv2.resize(image,(48,48) )

    image = image[:, :, np.newaxis]
    image = np.concatenate((image, image, image), axis=2)
    image = Image.fromarray(image)
    inputs = tensor_transform(image)

    ncrops,c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)

    inputs = Variable(inputs,volatile=True)
    outputs = model(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    
    score = F.softmax(outputs_avg).data.cpu().numpy()

    _, predicted = torch.max(outputs_avg.data, 0)
    
    id = int(predicted.cpu().numpy())
    return { 
      'id':id, 
      'class': EMOTIONS[id],
      "values":{
        "angry":score[0],
        "disgust":score[1],
        "fear":score[2],
        "happy":score[3],
        "sad":score[4],
        "surprise":score[5],
        "neutral":score[6],
      }
    }


def processRecognition(image, model):
    image = Image.fromarray(image)
    image = image.convert('L')
    image = image.resize((128,128), Image.NEAREST)
    x = keras_image.img_to_array(image, data_format='channels_last')
    batch_x = np.zeros((1,) + (128,128,1), dtype='float32')
    batch_x[0] = x
    batch_x /= 255.

    result = model.predict_class(batch_x)
    return result[0]


# Varibles
print("Loading config")
config_path = "full-face-analysis/config.yml"

cfg = load_yaml(config_path)

# Models
print("Loading models")
face_detector = retinaface.RetinaFaceModel(cfg, training=False)
checkpoint = tf.train.Checkpoint(model=face_detector)
if tf.train.latest_checkpoint(cfg['retina_model']):
    checkpoint.restore(tf.train.latest_checkpoint(cfg['retina_model']))
    print("[*] load ckpt from {}.".format(
        tf.train.latest_checkpoint(cfg['retina_model'])))
else:
    print("[*] Cannot find ckpt from {}.".format(cfg['retina_model']))
    exit()


emotion_detector = vgg.VGG('VGG19')
checkpoint = torch.load(cfg['vgg_model'], map_location=torch.device('cpu'))
emotion_detector.load_state_dict(checkpoint['net'])
emotion_detector.eval()

face_recognition = lightcnn.LightCNN( 
          classes = cfg['num_classes'], 
          extractor_weights = cfg['extractor_weights'],
          classifier_weights = cfg['classifier_weights'])

print("Starting camera")

# realtime prediction
cam = cv2.VideoCapture(0)
tensor_transform =  transforms.Compose([
    transforms.TenCrop(cfg['cut_size']),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
while True:
    _, frame = cam.read()
    if frame is None:
        print("no cam input")
        continue

    frame_height, frame_width, _ = frame.shape
    img = np.float32(frame.copy())
    if cfg['down_scale_factor'] < 1.0:
        img = cv2.resize( img, 
                          (0, 0), 
                          fx= cfg['down_scale_factor'],
                          fy= cfg['down_scale_factor'],
                          interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    # run model
    outputs = face_detector(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    # draw results
    for prior_index in range(len(outputs)):
        ann = outputs[prior_index]
        x1, y1, x2, y2 = int(ann[0] * frame_width), int(ann[1] * frame_height), int(ann[2] * frame_width), int(ann[3] * frame_height)
        face_image = frame[y1:y2, x1:x2]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        emotion = processEmotion(face_image, emotion_detector)
        person = processRecognition(face_image, face_recognition)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, PERSONS[person]+" "+emotion['class'], (int(ann[0] * frame_width), int(ann[1] * frame_height)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        exit()
