import numpy as np
from PIL import Image
import os
import fnmatch
import config as conf
from collections import deque
import cv2

my_pattern = "*.png"


def convert_to_npz(num, type, paths, label):
    array_of_images = []
    labels = []
    for i in range(len(paths)):
        imgs = []
        for _, file in enumerate(os.listdir(paths[i])):
            if fnmatch.fnmatch(file, my_pattern):
                # single_im = Image.open(os.path.join(paths[i], file))
                # single_array = np.array(single_im)
                # read the input image as grayscale image
                img = cv2.imread(os.path.join(paths[i], file),0)

                # normalize the binary image
                img_normalized = cv2.normalize(img, None, 0, 1.0,
                cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                img_normalized = np.reshape(img_normalized, (conf.height, conf.width, 1))
                imgs.append(img_normalized)
        array_of_images.append(imgs)
        labels.append(label)
    npz_path = os.path.join(conf.train2_folder, f"{num}-{type}.npz")
    try:
        np.savez(npz_path, images=array_of_images, labels=labels)
    except Exception as e:
        print(e)


data_type = "collision"

datapath = conf.autopiot_data_path
if data_type == "collision":
    datapath = conf.collision_data_path
col_data = os.listdir(datapath)
imgs_paths = deque([], 8)
for i in range(len(col_data)):
    imgs_paths.append(os.path.join(datapath, str(col_data[i])))
    if (i + 1) % 8 == 0 or i + 8 > len(col_data):
        label = (1, 0)
        if data_type == "collision":
            label = (0, 1)
        convert_to_npz(col_data[i], data_type, imgs_paths, label)
