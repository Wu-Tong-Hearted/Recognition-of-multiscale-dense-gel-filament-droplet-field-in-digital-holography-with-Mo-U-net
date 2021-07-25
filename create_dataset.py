import os
import tensorflow as tf
import pathlib
import numpy as np
import random

def read_path(root, trainimg, trainlab):
    trainimglist = sorted([os.path.join(root, trainimg, i) for i in os.listdir(os.path.join(root, trainimg))])
    trainlablist = sorted([os.path.join(root, trainlab, i) for i in os.listdir(os.path.join(root, trainlab))])
    return [trainimglist, trainlablist]

# load images from paths, and resize and normalize them
def load_and_preprocess_from_paths(path_ds, label_ds, IMG_LENGTH=224*3, IMG_WIDTH=224*3, padding=True):
    IMG_LENGTH = IMG_LENGTH
    IMG_WIDTH = IMG_WIDTH
    img, img_lab = tf.io.read_file(path_ds), tf.io.read_file(label_ds)
    img, img_lab = tf.image.decode_jpeg(img, channels=3), tf.image.decode_jpeg(img_lab, channels=1)
    img, img_lab = tf.image.resize_with_crop_or_pad(img, IMG_LENGTH, IMG_WIDTH), \
                 tf.image.resize_with_crop_or_pad(img_lab, IMG_LENGTH, IMG_WIDTH)

    # --------------------------random translation-----------------------------
    print(img.shape)
    h, w, c = img.shape
    flag = random.random()
    if flag < 0.5:
        ratio = random.random()
        if flag > 0.25:
            sw = int(w*ratio)
            a = img[:, sw:w, :]
            b = img[:, 0:sw, :]
            img = tf.concat([a, b], axis=1)
            a = img_lab[0:h, sw:w, :]
            b = img_lab[0:h, 0:sw, :]
            img_lab = tf.concat([a, b], axis=1)
        else:
            sh = int(h*ratio)
            a = img[sh:h, :, :]
            b = img[0:sh, :, :]
            img = tf.concat([a, b], axis=0)
            a = img_lab[sh:h, :, :]
            b = img_lab[0:sh, :, :]
            img_lab = tf.concat([a, b], axis=0)
    # --------------------------random translation-----------------------------


    # img, img_lab = tf.image.resize(img, [IMG_LENGTH, IMG_WIDTH]), \
    #              tf.image.resize(img_lab, [IMG_LENGTH, IMG_WIDTH])
    img, img_lab = tf.cast(img, dtype=tf.float32) / 255.0, tf.squeeze(tf.one_hot(tf.cast(img_lab > 125, dtype=tf.int32), depth=2))

    return img, img_lab


def slice_ds(ds, train_rate=1):
    if train_rate == 1:
        index = int(len(ds) * train_rate)
    else:
        index = int(len(ds) * train_rate) - 1
    test_ds = ds[index:]
    train_ds = ds[:index]
    return train_ds, test_ds


def load_ds(path='C:/Users/Pangzhentao/learn_keras/data', TRAIN_RATE = 0.8, BATCH_SIZE = 3):
    img_paths, lab_paths = read_path(path, 'train_img', 'train_label')
    all_paths = list(zip(img_paths, lab_paths))
    np.random.shuffle(all_paths)
    img_paths = [t[0] for t in all_paths]
    lab_paths = [t[1] for t in all_paths]
    train_paths, test_paths = slice_ds(img_paths, train_rate=TRAIN_RATE)
    train_lab_paths, test_lab_paths = slice_ds(lab_paths, train_rate=TRAIN_RATE)
    train_ds, test_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_lab_paths)), tf.data.Dataset.from_tensor_slices((test_paths, test_lab_paths))
    train_data = train_ds.shuffle(10000).map(load_and_preprocess_from_paths).batch(BATCH_SIZE)
    test_data = test_ds.shuffle(10000).map(load_and_preprocess_from_paths).batch(BATCH_SIZE)
    return train_data, test_data

def load_test_ds(path='C:/Users/Pangzhentao/learn_keras/data', BATCH_SIZE=3, test_input='test_img', test_label='ground_truth'):
    img_paths, lab_paths = read_path(path, test_input, test_label)
    test_paths = img_paths
    test_lab_paths = lab_paths
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_lab_paths))
    test_data = test_ds.map(load_and_preprocess_from_paths).batch(BATCH_SIZE)
    return test_data

def display(display_list):
    # plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    # for i in range(len(display_list)):
    i = 1
    # plt.subplot(1, len(display_list), i+1)
    # plt.title(title[i])
    img = tf.keras.preprocessing.image.array_to_img(display_list[i])
    # plt.axis('off')
    img.save(r'C:\Users\Pangzhentao\learn_keras\1.jpg')
    img.show()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type="GPU")

    train_data, test_data = load_ds()
    x, y = next(iter(train_data))
    # x.numpy(), y.numpy()
    # y0 = y[0, :, :, 0]
    # y1 = y[0, :, :, 1]
    # y0, y1 = tf.squeeze(y0), tf.squeeze(y1)
    # y0 = tf.squeeze(y0)
    #
    x = x[0, :, :, :]
    x = tf.squeeze(x)
    y = y[0, :, :, 0]
    y = tf.squeeze(y)
    y = tf.expand_dims(y, axis=-1)
    x = tf.keras.preprocessing.image.array_to_img(x)


    y = tf.keras.preprocessing.image.array_to_img(y)


    # x, y0, y1 = x * 255.0, y0 * 255.0, y1 * 255.0

    # im_y0 = Image.fromarray(np.uint8(y0))
    # im_y1 = Image.fromarray(np.uint8(y1))
    # im_x = Image.fromarray(np.uint8(x))
    # im_y0.show(title='im_y0')
    # im_y1.show(title='im_y1')
    # im_x.show(title='im_x')
    # display([x, y])
    # y3 = tf.reduce_sum(y, axis=-1)
    # y3 = tf.squeeze(y3)
    # y3 *= 255.0
    # im_y3 = Image.fromarray(np.uint8(y3))
    # im_y3.show(title='im_y3')
