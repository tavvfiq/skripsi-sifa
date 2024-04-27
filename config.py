import os

epochs = 24
time = 8
n_classes = 2
width,height,color_channels = 210,140,1
number_of_hiddenunits = 32
batch_size = 16

model_name = 'inception'
mode = 'test_video'

#config
base_folder = os.path.abspath(os.curdir)
data_path = os.path.join(base_folder,r'/mnt/d/Skripsi_Sifa/SourceCode/datasets_2')
dataset_path = os.path.join(base_folder,r'/mnt/d/Skripsi_Sifa/SourceCode/datasets')
train_folder = os.path.join(data_path, 'train_set')
train2_folder = os.path.join(dataset_path, 'train_set')
test_folder = os.path.join(data_path,'test_set')
valid2_folder = os.path.join(dataset_path, 'valid_set')
model_save_folder = os.path.join(base_folder,'files_2',model_name,'model_folder')
tensorboard_save_folder = os.path.join(base_folder,'files_2',model_name,'tensorboard_folder')
checkpoint_path = os.path.join(model_save_folder,"model_weights_{epoch:03d}.ckpt")
collision_data_path = os.path.join(base_folder, 'data_make_collision')
autopiot_data_path = os.path.join(base_folder, 'data_autopilot')