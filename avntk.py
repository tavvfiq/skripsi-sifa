import os, random
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import numpy as np
import json
import shutil
from collections import deque
from model_architecture import build_tools
import utils
import config as conf
import gpu

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print(os.getenv("TF_GPU_ALLOCATOR"))

ops.reset_default_graph()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

gpu.set_memory_limit(11*1024) #change this

if conf.mode == "train":
    if not os.path.exists(conf.model_save_folder):
        os.makedirs(conf.model_save_folder)
    else:
        shutil.rmtree(conf.model_save_folder)
        os.makedirs(conf.model_save_folder)

    if not os.path.exists(conf.tensorboard_save_folder):
        os.makedirs(conf.tensorboard_save_folder)
    else:
        shutil.rmtree(conf.tensorboard_save_folder)
        os.makedirs(conf.tensorboard_save_folder)


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=conf.checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq="epoch",
    period=4,
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=conf.tensorboard_save_folder,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(
    os.path.join(conf.base_folder, "files_2", "inference_video.avi"),
    fourcc,
    30.0,
    (800, 600),
)


def _trainer(network, train_generator, val_generator):
    network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    network.save_weights(conf.checkpoint_path.format(epoch=0))
    history = network.fit(
        train_generator,
        epochs=conf.epochs,
        steps_per_epoch=len(os.listdir(conf.train2_folder)) // conf.batch_size,
        validation_data=val_generator,
        validation_steps=1,
        callbacks=[cp_callback, tensorboard_callback],
    )
    with open(
        os.path.join(conf.base_folder, "files_2", conf.model_name, "training_logs.json"), "w"
    ) as w:
        json.dump(history.history, w)


def inference(network, video_file):
    print("debugging")
    image_seq = deque([], 8)
    cap = cv2.VideoCapture(video_file)
    counter = 0
    stat = "aman"
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            _frame = cv2.resize(frame, (conf.width, conf.height))
            image_seq.append(_frame)
            if counter % 2 == 0:
                if len(image_seq) == 8:
                    print(np.shape(image_seq))
                    np_image_seqs = np.reshape(
                        np.array(image_seq) / 255,
                        (1, conf.time, conf.height, conf.width, conf.color_channels),
                    )
                    r = network.predict(np_image_seqs)
                    print(r)
                    stat = ["aman", "tidak aman"][np.argmax(r, 1)[0]]
            green = (0, 255, 0)
            red = (0, 0, 255)
            color = green
            if stat == "tidak aman":
                color = red
            cv2.putText(
                frame,
                f"keadaan: {stat}",
                (200, 550),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                1,
                cv2.LINE_AA,
            )
            out.write(frame)
            counter += 1
            print(counter)
        else:
            cap.release()
            out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    model_tools = build_tools()
    network = model_tools.create_network(conf.model_name)

    if conf.mode == "train":
        train_generator = utils.data_tools(conf.train2_folder, "train")
        valid_generator = utils.data_tools(conf.valid2_folder, "valid")
        _trainer(
            network, train_generator.batch_dispatch(), valid_generator.batch_dispatch()
        )

    elif conf.mode == "test_video":
        network.load_weights(os.path.join(conf.model_save_folder, "model_weights_024.ckpt"))
        inference(network, os.path.join(conf.base_folder, "files_2", "input_video.mp4"))
    
    else:
        p = os.path.join(conf.train2_folder, '96-collision.npz')
        np_data = np.load(p, "r")
        imgs = np_data['images']
        np_image_seqs = np.reshape(
                        np.array(imgs[0]) / 255,
                        (1, conf.time, conf.height, conf.width, conf.color_channels),
                    )
        r = network.predict(np_image_seqs)
        print (np.argmax(r, 1))
        
