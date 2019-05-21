import glob
import os

#读取所有图片的路径
def read_image(paths, type):
    imgs = []
    for im in glob.glob(paths+'/*.' + type):
        imgs.append(im)
    return imgs

#获取一级子目录
def read_child_path(paths):
    paths_save = []
    for cpath in os.listdir(paths):
        paths_save.append(cpath)
    return paths_save
