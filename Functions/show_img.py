import tensorflow as tf
from matplotlib import pyplot as plt
import os
from scipy.ndimage import distance_transform_edt as distance

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def display(display_list, show=False, save=False, path='', order=None, file_name=None):

    # print('start dispaly')
    plt.figure(figsize=(15, 15))
    # dir_name = ['result', 'result', 'result']
    if file_name != None:
        dir_name = ['Input Image', 'True Mask', file_name]
    else:
        dir_name = ['Input Image', 'True Mask', 'Predicted Mask']

    for dir in dir_name:
        if os.path.exists(os.path.join(path, dir)) != True and None == file_name:
            os.mkdir(os.path.join(path, dir))
        elif os.path.exists(os.path.join(path, dir)) != True:
            os.mkdir(os.path.join(path, dir))
            # print('make model_pred dir')
        # print('check finished')

    if order != None:
        title = ['Input Image' + "{0:05d}".format(order),
                 'True Mask' + "{0:05d}".format(order),
                 'Predicted Mask' + "{0:05d}".format(order)
                 ]
    else:
        title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        img = tf.keras.preprocessing.image.array_to_img(display_list[i])
        # if save and dir_name[i] == file_name:
        if save:
            if file_name != None and dir_name[i] == file_name:
                img_path = os.path.join(path, dir_name[i], title[i] + '.png')
                img.save(img_path)
            elif file_name == None:
                img_path = os.path.join(path, dir_name[i], title[i] + '.png')
                img.save(img_path)
        if show:
            plt.imshow(img)
            plt.axis('off')

    # thumbnail_path = path + '/thumbnail' + str(order) + '.png'
    # if save:
    #     plt.savefig(thumbnail_path)
    if show:
        plt.show()
        plt.close('all')

def show_predict(number=10, source=None, model=None, show=False, save=False, path=None, epoch=None, file_name=None):
    '''
    :param number: means the number of prediction you would like to produce
    :param source: always your test_data
    :param model: your model
    :return: just display x, y, predict
    '''
    test_data = iter(source)
    for i in range(number):
        try:
            x, y = next(test_data)
        except Exception as e:
            print(e)
            break
        # print('predict x:', x.shape)
        # print('label y:', y.shape)

        model = model
        pred = model(x)

        x=x[0,...]
        y=y[0,...]
        pred = pred[0, ...]

        x = tf.squeeze(x)
        x = x.numpy() * 255.0

        y = tf.squeeze(y)
        y = y[:, :, 0]
        y = tf.expand_dims(y, axis=-1)
        y = tf.broadcast_to(y, [y.shape[0], y.shape[1], 3])
        y = y.numpy() * 255.0

        pred = tf.squeeze(pred)
        pred = tf.argmax(pred, axis=-1)
        pred = tf.expand_dims(pred, axis=-1)
        pred = tf.broadcast_to(pred, [pred.shape[0], pred.shape[1], 3])
        pred = pred.numpy() * 255.0
        pred = 255.0 - pred

        if save:
            if epoch != None:
                path = path + '/epoch_' + str(epoch)
            if not os.path.exists(path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(path)

        display_list = [x, y, pred]
        display(display_list, show=show, save=save, path=path, order=i+1, file_name=file_name)


def show_voting_predict(number=10, weight_list=[0.5, 0.5], show=False, source=None, basic_model=None, fusing_model=None, save=False, path=None, epoch=None):
    '''
    :param number: means the number of prediction you would like to produce
    :param source: always your test_data
    :param model: your model
    :return: just display x, y, predict
    '''
    test_data = iter(source)
    for i in range(number):
        try:
            x, y = next(test_data)
        except Exception as e:
            print(e)
            break
        print('predict x:', x.shape)
        print('label y:', y.shape)

        basic_model = basic_model
        fusing_model = fusing_model
        basic_pred = basic_model(x)
        fusing_pred = fusing_model(x)

        x = x[0, ...]
        y = y[0, ...]
        basic_pred = basic_pred[0, ...]
        fusing_pred = fusing_pred[0, ...]

        x = tf.squeeze(x)
        x = x.numpy() * 255.0

        y = tf.squeeze(y)

        # ---------------------------融合时揭开----------------------------
        # basic_pred_0 = basic_pred[:, :, 0]
        # basic_pred_1 = basic_pred[:, :, 1]
        # fusing_pred_1 = fusing_pred[:, :, 1]
        # ---------------------------融合时揭开----------------------------

        y = y[:, :, 0]
        y = tf.expand_dims(y, axis=-1)
        y = tf.broadcast_to(y, [y.shape[0], y.shape[1], 3])
        y = y.numpy() * 255.0

        basic_pred = tf.squeeze(basic_pred)
        fusing_pred = tf.squeeze(fusing_pred)

        pred = weight_list[0] * basic_pred + weight_list[1] * fusing_pred

        # ---------------------------融合时揭开----------------------------
        # pred_0 = pred[..., 0]
        # pred_0 = tf.expand_dims(pred_0, axis=-1)
        # pred_1 = pred[..., 1]
        # pred_1 = pred_1 + beta * y_1
        # pred_1 = tf.expand_dims(pred_1,axis=-1)
        # pred = tf.concat([pred_0, pred_1], axis=-1)
        # ---------------------------融合时揭开----------------------------

        pred = tf.argmax(pred, axis=-1)
        pred = tf.expand_dims(pred, axis=-1)
        pred = tf.broadcast_to(pred, [pred.shape[0], pred.shape[1], 3])
        pred = pred.numpy() * 255.0
        pred = 255.0 - pred

        if save:
            if epoch != None:
                path = path + '/epoch_' + str(epoch)
            if not os.path.exists(path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(path)

        display_list = [x, y, pred]
        display(display_list, show=show, save=save, path=path, order=i+1)
