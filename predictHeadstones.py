import argparse
import json
import cv2
import os
import time
from utils.utils import get_yolo_boxes, makedirs
from keras.models import load_model
from tqdm import tqdm


def _main_(args):
    config_path = args.conf
    input_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Set some parameter
    ###############################
    height, width = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    model = load_model(config['train']['saved_weights_name'])
    anchors = config['model']['anchors']

    video_reader = cv2.VideoCapture(input_path)

    while True:
        _, image = video_reader.read()
        if (image is None): break

        start = time.time()
        batch_boxes = get_yolo_boxes(model, [image], height, width, anchors, obj_thresh, nms_thresh)
        end = time.time()
        print(end - start)

    video_reader.release()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')

    args = argparser.parse_args()
    _main_(args)