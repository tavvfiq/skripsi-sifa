import numpy as np
from PIL import Image
import os
import fnmatch
import config as conf
from collections import deque
import cv2

my_pattern = "*.png"


# convert n batch of sequences into .npz
# 1 sequence = 8 images
# filename: filename
# type: collision or safe
# paths: list of sequences's path
# label: this sequences label (safe or not safe in tuple)
def convert_to_npz(filename, type, paths, label):
    array_of_images = []
    labels = []
    for i in range(len(paths)):
        imgs = []
        for _, file in enumerate(os.listdir(paths[i])):
            if fnmatch.fnmatch(file, my_pattern):
                single_im = Image.open(os.path.join(paths[i], file))
                single_array = np.array(single_im)
                imgs.append(single_array)
        array_of_images.append(imgs)
        labels.append(label)
    npz_path = os.path.join(conf.train2_folder, f"{filename}-{type}.npz")
    try:
        np.savez(npz_path, images=array_of_images, labels=labels)
    except Exception as e:
        print(e)


if __name__ == "__main__":
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
            

