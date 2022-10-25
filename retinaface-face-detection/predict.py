from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
import uuid

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm, pad_input_image, recover_pad_output)

flags.DEFINE_string('cfg_path', 'retinaface-face-detection/config.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')
flags.DEFINE_string('model_path', 'retinaface-face-detection/model/checkpoints', 'which gpu to use')

def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = FLAGS.model_path + "/" +cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    cam = cv2.VideoCapture(0)

    start_time = time.time()
    while True:
        _, frame = cam.read()
        if frame is None:
            print("no cam input")

        frame_height, frame_width, _ = frame.shape
        img = np.float32(frame.copy())
        if FLAGS.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                              fy=FLAGS.down_scale_factor,
                              interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        outputs = model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # draw results
        for prior_index in range(len(outputs)):
            draw_bbox_landm(frame, outputs[prior_index], frame_height,
                            frame_width)

        # calculate fps
        fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
        start_time = time.time()
        cv2.putText(frame, fps_str, (25, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        # show frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass