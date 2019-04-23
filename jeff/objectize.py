import json
import nltk
import random
import sys

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from io import BytesIO
from PIL import Image
import numpy as np
import cv2

from nltk.util import ngrams

objs = [json.loads(line) for line in open("/home/jzda/code/nlvr/nlvr/nlvr2/data/train.json").readlines()]

outfile = open("/home/jzda/code/nlvr/kitchen_sink/raw_representations/object_detection/features_train_cuda.txt", "w")

def get_left_link(identifier, directory):
    if 'directory' is not None: # training image
        return "/home/jzda/nlvr2/images/train/" + str(directory) + "/" + str(identifier)[:-2] + "-img0.png"
    else: # dev image
        return "/home/jzda/nlvr2/dev/" + str(identifier)[:-2] + "-img0.png"

def get_right_link(identifier, directory):
    if 'directory' is not None: # training image
        return "/home/jzda/nlvr2/images/train/" + str(directory) + "/" + str(identifier)[:-2] + "-img1.png"
    else: # dev image
        return "/home/jzda/nlvr2/dev/" + str(identifier)[:-2] + "-img1.png"

config_file = "/home/jzda/code/nlvr/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

for i, obj in enumerate(objs):
    identifier = obj["identifier"]
    directory = None
    if "directory" in obj:
        directory = obj["directory"]

    image_left = load(get_left_link(identifier, directory))
    image_right = load(get_right_link(identifier, directory))

    predictions_left = coco_demo.run_on_opencv_image(image_left)
    predictions_right = coco_demo.run_on_opencv_image(image_right)

    left_features = []
    right_features = []

    for item in range(len(CATEGORIES)):
        count_left = predictions_left.get_field("labels").tolist().count(item)
        count_right = predictions_right.get_field("labels").tolist().count(item)
        left_features.append(count_left)
        right_features.append(count_right)

    print(obj["directory"])
    result = {
        "identifier": obj["identifier"],
        "left_vec": left_features,
        "right_vec": right_features
    }

    outfile.write(json.dumps(result) + "\n")
