import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
#import utils
from mrcnn import model as modellib,utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

ROOT_DIR = os.getcwd()
#模型保存目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
iter_num=0

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# 基础设置
dataset_root_path = "E:\\pycharm\\code_of_auth\\"
labelme_json_path = dataset_root_path + 'labelme_json'
img_floder = dataset_root_path + "pic"
# yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)


class ShapesConfig(Config):
    NAME = "shapes"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 36  # background + 10 numbers + 26 char

    # Use smaller anchors because our image and objects are small , 目标的尺寸,自己去测量几个放到这里
    RPN_ANCHOR_SCALES = (40, 50, 60) # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 100

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 50

config = ShapesConfig()
config.display()


class TrainDataset(utils.Dataset):

    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(),Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
            return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "0")
        self.add_class("shapes", 2, "1")
        self.add_class("shapes", 3, "2")
        self.add_class("shapes", 4, "3")
        self.add_class("shapes", 5, "4")
        self.add_class("shapes", 6, "5")
        self.add_class("shapes", 7, "6")
        self.add_class("shapes", 8, "7")
        self.add_class("shapes", 9, "8")
        self.add_class("shapes", 10, "9")

        self.add_class("shapes", 11, "a")
        self.add_class("shapes", 12, "b")
        self.add_class("shapes", 13, "c")
        self.add_class("shapes", 14, "d")
        self.add_class("shapes", 15, "e")
        self.add_class("shapes", 16, "f")
        self.add_class("shapes", 17, "g")

        self.add_class("shapes", 18, "h")
        self.add_class("shapes", 19, "i")
        self.add_class("shapes", 20, "j")
        self.add_class("shapes", 21, "k")
        self.add_class("shapes", 22, "l")
        self.add_class("shapes", 23, "m")
        self.add_class("shapes", 24, "n")

        self.add_class("shapes", 25, "o")
        self.add_class("shapes", 26, "p")
        self.add_class("shapes", 27, "q")
        self.add_class("shapes", 28, "r")
        self.add_class("shapes", 29, "s")
        self.add_class("shapes", 30, "t")

        self.add_class("shapes", 31, "u")
        self.add_class("shapes", 32, "v")
        self.add_class("shapes", 33, "w")
        self.add_class("shapes", 34, "x")
        self.add_class("shapes", 35, "y")
        self.add_class("shapes", 36, "z")

        for i in range(count):
            # 获取图片宽和高
            print(i)
            filestr = imglist[i].replace(".jpg", "")

            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = dataset_root_path + "\\" + filestr + "_json\\label.png"
            yaml_path = dataset_root_path + "\\" + filestr + "_json\\info.yaml"
            img_path = dataset_root_path + "\\" + filestr + "_json\\img.png"
            print(img_path)
            cv_img = cv2.imread(img_path)
            self.add_image("shapes", image_id=i, path=img_floder + "\\" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("0") != -1:
                labels_form.append("0")
            elif labels[i].find("1") != -1:
                labels_form.append("1")
            elif labels[i].find("2") != -1:
                labels_form.append("2")
            elif labels[i].find("3") != -1:
                labels_form.append("3")
            elif labels[i].find("4") != -1:
                labels_form.append("4")
            elif labels[i].find("5") != -1:
                labels_form.append("5")
            elif labels[i].find("6") != -1:
                labels_form.append("6")
            elif labels[i].find("7") != -1:
                labels_form.append("7")
            elif labels[i].find("8") != -1:
                labels_form.append("8")
            elif labels[i].find("9") != -1:
                labels_form.append("9")
            elif labels[i].find("a") != -1:
                labels_form.append("a")
            elif labels[i].find("b") != -1:
                labels_form.append("b")
            elif labels[i].find("c") != -1:
                labels_form.append("c")
            elif labels[i].find("d") != -1:
                labels_form.append("d")
            elif labels[i].find("e") != -1:
                labels_form.append("e")
            elif labels[i].find("f") != -1:
                labels_form.append("f")
            elif labels[i].find("g") != -1:
                labels_form.append("g")
            elif labels[i].find("h") != -1:
                labels_form.append("h")
            elif labels[i].find("i") != -1:
                labels_form.append("i")
            elif labels[i].find("j") != -1:
                labels_form.append("j")
            elif labels[i].find("k") != -1:
                labels_form.append("k")
            elif labels[i].find("l") != -1:
                labels_form.append("l")
            elif labels[i].find("m") != -1:
                labels_form.append("m")
            elif labels[i].find("n") != -1:
                labels_form.append("n")
            elif labels[i].find("o") != -1:
                labels_form.append("o")
            elif labels[i].find("p") != -1:
                labels_form.append("p")
            elif labels[i].find("q") != -1:
                labels_form.append("q")
            elif labels[i].find("r") != -1:
                labels_form.append("r")
            elif labels[i].find("s") != -1:
                labels_form.append("s")
            elif labels[i].find("t") != -1:
                labels_form.append("t")
            elif labels[i].find("u") != -1:
                labels_form.append("u")
            elif labels[i].find("v") != -1:
                labels_form.append("v")
            elif labels[i].find("w") != -1:
                labels_form.append("w")
            elif labels[i].find("x") != -1:
                labels_form.append("x")
            elif labels[i].find("y") != -1:
                labels_form.append("y")
            elif labels[i].find("z") != -1:
                labels_form.append("z")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# train与val数据集准备
dataset_train = TrainDataset()
dataset_train.load_shapes(count, img_floder, imglist, labelme_json_path)
dataset_train.prepare()
# print("dataset_train-->",dataset_train._image_ids)

dataset_val = TrainDataset()
dataset_val.load_shapes(count, img_floder, imglist, labelme_json_path)
dataset_val.prepare()
# print("dataset_val-->",dataset_val._image_ids)
# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # print(COCO_MODEL_PATH)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE, epochs=100, layers='heads')
# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE / 10, epochs=100, layers="all")
