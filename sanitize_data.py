import os
import config as conf
import shutil

col_data = os.listdir(conf.collision_data_path)
counter = 0
for i in range(len(col_data)):
    imgs_path = os.path.join(conf.collision_data_path, str(col_data[i]))
    num_of_imgs = os.listdir(imgs_path)
    if len(num_of_imgs) < conf.time or len(num_of_imgs) > conf.time:
        counter+=1
        shutil.rmtree(imgs_path)
        
print(f'collision_data total {counter}')

auto_data = os.listdir(conf.autopiot_data_path)
counter = 0
for i in range(len(col_data)):
    imgs_path = os.path.join(conf.autopiot_data_path, str(auto_data[i]))
    num_of_imgs = os.listdir(imgs_path)
    if len(num_of_imgs) < conf.time or len(num_of_imgs) > conf.time:
        counter+=1
        shutil.rmtree(imgs_path)
        
print(f'autopiot_data total {counter}')