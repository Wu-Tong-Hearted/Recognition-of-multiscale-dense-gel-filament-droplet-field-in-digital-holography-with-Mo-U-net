import tensorflow as tf
import create_dataset
import model.Unet_tiny as Unet_tiny
from Functions.datascripe.read_pred_vision import save_change_plt as rpv

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(devices=gpus[5:6], device_type="GPU")
if gpus:
    try:
        # 设置 GPU 显存占用为按需分配，增长式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 异常处理
        print(e)

# ---------------------------不使用GPU时揭开--------------------------
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# ---------------------------不使用GPU时揭开--------------------------

# ---------------------------使用GPU时揭开--------------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(devices=gpus['3'], device_type="GPU")
# ---------------------------不使用GPU时揭开--------------------------

IMG_HEIGHT = 224*3
IMG_WIDTH = 224*3
CHANNELS = 3
BATCH_SIZE = 1
# train_rate = 0.8

PATH = r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_2.58mm velocity_118.2'
test_dir_name = r'\cut_672'
PATH += test_dir_name
# PATH = r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_1.74mm velocity_92.9\cut_672'

# 加载数据集
test_data = create_dataset.load_test_ds(path=PATH,
                                        BATCH_SIZE=BATCH_SIZE,
                                        test_input='test_img',
                                        test_label='ground_truth')

# train_data, test_data = create_dataset.load_ds(path=r'C:\Users\Pangzhentao\learn_keras\data', TRAIN_RATE=train_rate, BATCH_SIZE=BATCH_SIZE)

# 检测训练结果，输入一个test数据，看pred
import Functions.show_img as simg

print('-------------------------testing!!!-----------------------------')
# 如果需要加载模型，将下面注释揭开，需要指向特定weight的时候在checkpoint文件中修改指向的文件名
# 加载basic_model
Unet_model = Unet_tiny.UnetTiny(classes=2, input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS], trainable=False) # 加载Unet
basic_model = Unet_model
checkpoint_dir = r'C:\Users\Pangzhentao\learn_keras\train_log\20210626-185451-unet_tiny-categorical_crossentropy-100e-0.001'
latest = tf.train.latest_checkpoint(checkpoint_dir)
basic_model.load_weights(latest)

pred_file_name = 'NP_CEL_pred'
# model_path = r'C:\Users\Pangzhentao\learn_keras\result\models\20210629-234752-unet_tiny-categorical_crossentropy-1e-0.001'
# basic_model = tf.saved_model.load(model_path)

# 加载fusing_model
# SAVED_MODEL = r'C:\Users\Pangzhentao\learn_keras\result\models\20210606-115654-Unet-CEL-50e-0.001'
# fusing_model = tf.saved_model.load(SAVED_MODEL)

# PATH = r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_2.13mm velocity_92.9'
# test_dir_name = r'\cut_672'
# PATH += test_dir_name

# train_dir_name = r'\train_result'
# test_dir_name = r'\test_result'

# -----------------------------保存预测-----------------------------
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