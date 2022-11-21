import numpy as np
import socketio
import eventlet
import tensorflow as tf
import torch
from modules import package
from models import vgg, lightcnn, retinaface
from modules import  utils
from modules import  face
from typing import List

# Variables

HOST = '192.168.1.9'
PORT = 5000

config_path = "full-face-analysis/config.yml"

cfg = utils.load_yaml(config_path)
queue: List[package.Package] = []

sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)


# Models
face_detector = retinaface.RetinaFaceModel(cfg, training=False)
checkpoint = tf.train.Checkpoint(model=face_detector)
checkpoint.restore(tf.train.latest_checkpoint(cfg['retina_model']))

emotion_detector = vgg.VGG('VGG19')
checkpoint = torch.load(cfg['vgg_model'], map_location=torch.device('cpu'))
emotion_detector.load_state_dict(checkpoint['net'])
emotion_detector.eval()

face_recognition = lightcnn.LightCNN( 
          classes = cfg['num_classes'], 
          extractor_weights = cfg['extractor_weights'],
          classifier_weights = cfg['classifier_weights'])

# Events

@sio.event
def analyse_image(sid, data):
  new_package = package.Package(sid, data['image'], data['analyse_person'], data['analyse_emotion'])
  queue.append(new_package)

@sio.event
def connect(sid, environ):
  print('connect ', sid)

def process():
  counter = 0
  while True:
    if len(queue) <= 0:
      eventlet.sleep()
      continue

    to_process = queue.pop()
    to_process.decode_image()
    #counter+=1
    #to_process.save_image(to_process.image,f"full-face-analysis/images/test{counter}.jpg")
    result = []

    # get faces
    retinaInput, pad_params = to_process.get_retinaface_image()
    outputs = face_detector(retinaInput[np.newaxis, ...]).numpy()
    outputs = utils.recover_pad_output(outputs, pad_params)

    frame_height, frame_width, _ = to_process.get_image_size()
    to_process.to_grayscale()
    ## generate faces batch
    for prior_index in range(len(outputs)):
      ann = outputs[prior_index]
      
      # loww confidence
      if(ann[15] < 0.5):
        continue

      x1, y1, x2, y2 = int(ann[0] * frame_width), int(ann[1] * frame_height), int(ann[2] * frame_width), int(ann[3] * frame_height)
      face_image = to_process.image[y1:y2, x1:x2]
      
      ## should be create a batch and process all in one
      face_to_process = face.Face(face_image,x1, y1, x2, y2,ann[15])

      if to_process.analyse_emotion:
        emotions = emotion_detector(face_to_process.get_image_vgg())
        face_to_process.save_emotion(emotions)

      if to_process.analyse_person:
        person = face_recognition.predict_class(face_to_process.get_image_ligthcnn())
        face_to_process.save_person(person[0])
      
      result.append(face_to_process.get_result())
    sio.emit(
        "result", 
        data=result, 
        room=to_process.socket_id)

    eventlet.sleep()

# Run
if __name__ == '__main__':
  sio.start_background_task(process)
  eventlet.wsgi.server(eventlet.listen((HOST, PORT)), app)
