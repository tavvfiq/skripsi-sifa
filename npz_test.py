import numpy as np
import matplotlib.pyplot as plt
import config as conf
import os
  
npz_path=os.path.join(conf.test_folder, '0.npz')
npz_path2=os.path.join(conf.train2_folder, '99-collision.npz')
np_data = np.load(npz_path, "r")
np_data2 = np.load(npz_path2, "r")
# print(np_data['name2'])
images2 = np_data['name1']
images = np_data2['images']

labels2 = np_data['name2']
labels = np_data2['labels']

print(np.shape(images), np.shape(images2), np.shape(labels), np.shape(labels2))
# plt.figure()
# f, axarr = plt.subplots(2,1) 
for i in range(len(images)):
    for frames in images[i]:
        print(np.shape(frames))
        #subplot(r,c) provide the no. of rows and columns
        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        plt.imshow(frames)
        plt.show()
    print(f'label: {labels[i]}')