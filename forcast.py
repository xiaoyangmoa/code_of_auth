import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
import get_result as get_result
import ImagePath as ImagePath

ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils

import mrcnn.model as modellib

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 36  # background + 10 numbers

    # IMAGE_MIN_DIM = 160
    # IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small , 目标的尺寸,自己去测量几个放到这里
    RPN_ANCHOR_SCALES = (40, 50, 60)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 100

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50

# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights('logs/shapes20190521T0957/mask_rcnn_shapes_0100.h5', by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# Load a random image from the images folder

data_path = 'test_jpg'

flle_list = ImagePath.read_image(data_path, 'jpg')

for file in flle_list:
    image = skimage.io.imread(file)
    a = datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b = datetime.now()
    # Visualize results
    print("time cost", (b - a).seconds)
    r = results[0]
    masked_image = get_result.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    file = file.replace("test_jpg","test_result")
    masked_image.savefig(file)

print('end predict!')
