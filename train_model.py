# 加载依赖库

import os
# ---------------------------不使用GPU时揭开--------------------------
# os.environ['CUDA_VISIBLE_DEVICES'] = ' 0,1,2,3'
# ---------------------------不使用GPU时揭开--------------------------
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
# ---------------------------使用GPU时揭开--------------------------
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(devices=gpus[1:4], device_type="GPU")
# ---------------------------使用GPU时揭开--------------------------
import datetime
import importlib
import create_dataset
import Functions.Losses.balenced_losses_for_sematic_segmentation as myloss
import model.Unet_tiny as Unet_tiny
import model.FCN_8s as FCN_8s
import Functions.callbacks as callbacks
from Functions.metrics import MeanIoU, CategoricalTruePositivesRecall
importlib.reload(create_dataset)

# ---------------------------使用GPU时揭开--------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(devices=gpus[1], device_type="GPU")
# ---------------------------使用GPU时揭开--------------------------


if gpus:
    try:
        # 设置 GPU 显存占用为按需分配，增长式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 异常处理
        print(e)

# --------------------------多GPU训练-------------------------------
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    IMG_HEIGHT = 224*3
    IMG_WIDTH = 224*3
    CHANNELS = 3
    # BATCH_SIZE = 4
    BATCH_SIZE = 8

    # 加载数据集
    dataset_name = 'at_data_enhance'
    train_data, test_data = create_dataset.load_ds(path='C:/Users/Pangzhentao/learn_keras/data/' + dataset_name, BATCH_SIZE=BATCH_SIZE, TRAIN_RATE=0.8)

    # 加载测试数据集
    # train_data, test_data = create_dataset.load_ds(path=r'C:\Users\Pangzhentao\learn_keras\test data', BATCH_SIZE=BATCH_SIZE, TRAIN_RATE=1)

    # 加载模型
    Unet_model = Unet_tiny.UnetTiny(classes=2, input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS], trainable=False) # 加载Unet
    FCN_8s = FCN_8s.fcn_8s(num_classes=2, input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS], baseline_trainable=True)

    # 加载已经之前保存的模型
    # SAVED_MODEL = r'C:\Users\Pangzhentao\learn_keras\result\models\20210605-102105-Unet-TL-50e-0.001'
    # model = tf.saved_model.load(SAVED_MODEL)
    # CONTINUETRAIN = True

    # 选择需要测试的model
    # model = Unet_model
    # MODEL_NAME = model.name
    # MODEL_NAME = 'Unet-TL-50e-0.001'
    # model = FCN_8s
    # MODEL_NAME = 'FCN_8s'

    # # 打印模型信息
    # model.summary()
    # tf.keras.utils.plot_model(model, MODEL_NAME + '.png', show_shapes=True, show_layer_names=True)

    # 设置训练参数
    learning_rate = 0.001
    alpha = tf.keras.backend.variable(1, dtype='float32')  # 当使用SAD_B_Loss时揭开
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.SGD(momentum=0.99)
    loss_list = [
                 # myloss.Tversky_Loss(alpha=1, beta=0),  # 0
                 # myloss.Tversky_Loss(alpha=0, beta=1),  # 1
                 # myloss.Self_Adaptive_Dice_Loss(gama=2),  # 2
                 # myloss.Normal_Dice_Loss(gama=1),  # 3
                 tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 4
                 # myloss.Weighted_Cross_Entropy_Loss(),  # 5
                 # myloss.Focal_Loss(),  # 6
                 # myloss.Weighted_Focal_Loss(),  # 7
                 # myloss.Boundary_Loss(),  # 8
                 # myloss.SAD_B_Loss(alpha=alpha, gama=2),  # 9
                 # myloss.ND_B_Loss(alpha=alpha, gama=2),  # 10
                 # myloss.CE_B_Loss(alpha=alpha)  # 11
                 ]
    loss = loss_list[0]
    epoch = 100


    # 加载模型
    model = Unet_model
    MODEL_NAME = model.name

    learning_rate = 0.001
    alpha = tf.keras.backend.variable(1, dtype='float32')  # 当使用SAD_B_Loss时揭开
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 创建数据记录文件
    SAVED_MODEL_NAME = MODEL_NAME + '-' + loss.name + '-' + str(epoch) + 'e-' + str(learning_rate)
    train_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir="logs/fit/" + train_time + '-' + SAVED_MODEL_NAME
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    PATH = './result/predictions/' + train_time + '-' + dataset_name + '-' + SAVED_MODEL_NAME


    # checkpoint_path 写模型的保存路径
    checkpoint_path = "train_log/" + train_time + '-' + SAVED_MODEL_NAME + "/cp-epoch{epoch:04d}-loss{loss:5.4f}-val{val_loss:5.4f}.ckpt"
    # checkpoint_path = r'C:\Users\Pangzhentao\learn_keras\train_log\tune'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # 如果需要加载模型，将下面注释揭开，需要指向特定weight的时候在checkpoint文件中修改指向的文件名
    # latest = tf.train.latest_checkpoint(checkpoint_path)
    # model.load_weights(latest)

    # 需要保存参数训练的时候，调用这个callback函数
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        period=0  # 这个参数表示多少个epoch后保存一个weight
    )

    # 创建模型训练compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['categorical_accuracy', CategoricalTruePositivesRecall(), MeanIoU(num_classes=2)],
        # metrics=['categorical_accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
        run_eagerly=True
    )

    # 开始训练模型
    history = model.fit(
        train_data,
        epochs=epoch,
        validation_data=test_data,
        validation_freq=1,
        callbacks=[tensorboard_callback,
                   cp_callback,
                   callbacks.ShowAtEpoch(test_data, number=1, save=True, save_epochs=1, path=PATH),
                   callbacks.LearningRateChangeByLoss(patience=3, lr_change_time=3, change_rate=0.1),
                   callbacks.AlphaScheduler(alpha=alpha, update_step=0.01, wait_epoch=0)
                   ]
        )

    # 保存模型结构与模型参数到文件
    SAVED_MODEL = './result/models/' + train_time + '-' + SAVED_MODEL_NAME
    tf.saved_model.save(model, SAVED_MODEL)
    print('saving savedmodel.')
    del model # 删除网络对象
    # 检测训练结果，输入一个test数据，看pred
    import Functions.show_img as simg

    print('testing!!!')
    model = tf.saved_model.load(SAVED_MODEL)
    PATH = PATH + '/last_test'
    simg.show_predict(number=10, source=test_data, model=model, show=False, save=True, path=PATH)
    del model


