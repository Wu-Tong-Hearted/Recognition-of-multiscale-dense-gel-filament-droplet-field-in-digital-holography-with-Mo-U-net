import tensorflow as tf
import create_dataset
import model.Unet_tiny as Unet_tiny
from Functions.datascripe.read_pred_vision import save_change_plt as rpv

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(devices=gpus[5:6], device_type="GPU")
if gpus:
    try:
        # set GPU usage with a demanding mode
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ---------------------------reveal it if you donot use a GPU--------------------------
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ---------------------------reveal it if you donot use a GPU--------------------------

# ---------------------------reveal it if you do use a GPU--------------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(devices=gpus['3'], device_type="GPU")
# ---------------------------reveal it if you do use a GPU--------------------------

IMG_HEIGHT = 224*3
IMG_WIDTH = 224*3
CHANNELS = 3
BATCH_SIZE = 1
# train_rate = 0.8

PATH = r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_2.58mm velocity_118.2'
test_dir_name = r'\cut_672'
PATH += test_dir_name
# PATH = r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_1.74mm velocity_92.9\cut_672'

# load train set
test_data = create_dataset.load_test_ds(path=PATH,
                                        BATCH_SIZE=BATCH_SIZE,
                                        test_input='test_img',
                                        test_label='ground_truth')

# train_data, test_data = create_dataset.load_ds(path=r'C:\Users\Pangzhentao\learn_keras\data', TRAIN_RATE=train_rate, BATCH_SIZE=BATCH_SIZE)

# test
import Functions.show_img as simg

print('-------------------------testing!!!-----------------------------')
# if you want load model, reveal codes below, when a specific weight is demanded, change the root in checkpoint file
# load basic_model
Unet_model = Unet_tiny.UnetTiny(classes=2, input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS], trainable=False) # 加载Unet
basic_model = Unet_model
checkpoint_dir = r'C:\Users\Pangzhentao\learn_keras\train_log\20210626-185451-unet_tiny-categorical_crossentropy-100e-0.001'
latest = tf.train.latest_checkpoint(checkpoint_dir)
basic_model.load_weights(latest)

pred_file_name = 'NP_CEL_pred'
# model_path = r'C:\Users\Pangzhentao\learn_keras\result\models\20210629-234752-unet_tiny-categorical_crossentropy-1e-0.001'
# basic_model = tf.saved_model.load(model_path)


# -----------------------------save prediction-----------------------------
simg.show_predict(number=100, source=test_data, model=basic_model, save=True, path=PATH, file_name=pred_file_name)

# simg.show_voting_predict(number=10000,
#                          source=test_data,
#                          weight_list=[0.5, 0.5],
#                          basic_model=basic_model,
#                          fusing_model=fusing_model,
#                          show=True,
#                          save=True,
#                          path=PATH
#                          )
print('---------------------testing finished!!!------------------------')

print('------------------------generating avi!!!-----------------------')
name = 'test'
rpv(name=name, path=PATH)
print('----------------------avi generated successfully!!!-------------')

# print('------------------------counting metrics!!!----------------------')

# import numpy as np
# from Functions import metrics_in_novel as metrics
#
# result_PIOU = metrics.Count_Metric(metric='PIOU',
#                                    file_path=PATH,
#                                    pred_file_name=pred_file_name,
#                                    true_file_nam='ground_truth')
# result_PIOU = np.array(result_PIOU)
# result_PIOU = np.round(result_PIOU, 4)
# print('The mean PIOU is %.4f, the min PIOU is %.4f, the max PIOU is %.4f' % (result_PIOU[0], result_PIOU[1], result_PIOU[2]))
# print('all result is: ', result_PIOU)
#
# result_ASSD = metrics.Count_Metric(metric='ASSD',
#                                    file_path=PATH,
#                                    pred_file_name=pred_file_name,
#                                    true_file_nam='ground_truth')
#
# result_ASSD = np.array(result_ASSD)
# result_ASSD = np.round(result_ASSD, 4)
# print('The mean ASSD is %.4f, the min ASSD is %.4f, the max ASSD is %.4f' % (result_ASSD[0], result_ASSD[1], result_ASSD[2]))
# print('all result is: ', result_ASSD)

# print('------------------------counting metrics finished!!!----------------------')