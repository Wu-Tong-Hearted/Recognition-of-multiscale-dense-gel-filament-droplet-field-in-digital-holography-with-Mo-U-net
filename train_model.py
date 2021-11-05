# 加载依赖库
# 训练模型


import os
# ---------------------------不使用GPU时揭开--------------------------
# os.environ['CUDA_VISIBLE_DEVICES'] = ' 0,1,2,3'
# ---------------------------不使用GPU时揭开--------------------------
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import datetime
import importlib
import create_dataset
import Functions.Losses.balenced_losses_for_sematic_segmentation as myloss
import model.Mo_U_net as Unet_tiny

import Functions.callbacks as callbacks
from Functions.metrics import MeanIoU, CategoricalTruePositivesRecall
importlib.reload(create_dataset)

# ---------------------------when use GPU, reveal it--------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')# 这里会给出是不是有gpu

# 有gpu的情况下设置gpu的序号
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type="GPU")
# ---------------------------when use GPU, reveal it--------------------------


if gpus:
    try:
        # set GPU usage with a demanding mode
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # warning
        print(e)

# --------------------------use multi GPU traiing-------------------------------
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    # 基本参数
    IMG_HEIGHT = 224*3
    IMG_WIDTH = 224*3
    CHANNELS = 3

    # BATCH_SIZE = 4
    BATCH_SIZE = 8
    TRAIN_RATE = 0.8  # 调整训练和验证集比例

    # load train set
    dataset_name = 'data'
    datapath=r'C:\Users\wxyice\Documents\GitHub\Recognition-of-multiscale-dense-gel-filament-droplet-field-in-digital-holography-with-Mo-U-net\data'


    train_data, test_data = create_dataset.load_ds(path=datapath, BATCH_SIZE=BATCH_SIZE, TRAIN_RATE=0.8)

    # load test set
    # train_data, test_data = create_dataset.load_ds(path=r'C:\Users\Pangzhentao\learn_keras\test data', BATCH_SIZE=BATCH_SIZE, TRAIN_RATE=1)

    # load model
    Unet_model = Unet_tiny.UnetTiny(classes=2, input_shape=[IMG_HEIGHT, IMG_WIDTH, CHANNELS], trainable=False) # 加载Mo-U-net模型，重新训练

    # ---------------------------加载训练模型--------------------------
    # 可以继续训练
    # load model that has been saved previously 或者加载之前已经训练过的模型
    # SAVED_MODEL = r'C:\Users\Pangzhentao\learn_keras\result\models\20210605-102105-Unet-TL-50e-0.001' # 这里写的是checkpoint的保存路径
    # model = tf.saved_model.load(SAVED_MODEL)
    # CONTINUETRAIN = True
    # ------------------------------------------------------------------

    # choose which model to test

    # model = Unet_model
    # MODEL_NAME = model.name
    # MODEL_NAME = 'Unet-TL-50e-0.001'
    # model = FCN_8s
    # MODEL_NAME = 'FCN_8s'

    # ------------------------------------------------------------------


    # -----------print the basic information of the model---------------
    # model.summary()
    # tf.keras.utils.plot_model(model, MODEL_NAME + '.png', show_shapes=True, show_layer_names=True)
    # ------------------------------------------------------------------

    # -------------------------training setting--------------------------
    learning_rate = 0.001
    #alpha = tf.keras.backend.variable(1, dtype='float32')  # when use SAD_B_Loss, reveal it

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.SGD(momentum=0.99)

    loss_list = [
                 # myloss.Tversky_Loss(alpha=1, beta=0),  # 0
                 # myloss.Tversky_Loss(alpha=0, beta=1),  # 1
                 # myloss.Self_Adaptive_Dice_Loss(gama=2),  # 2
                 # myloss.Normal_Dice_Loss(gama=1),  # 3
                 tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 4 交叉熵loss
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

    # -------------------------load model--------------------------
    model = Unet_model
    MODEL_NAME = model.name

    # create training logs
    SAVED_MODEL_NAME = MODEL_NAME + '-' + loss.name + '-' + str(epoch) + 'e-' + str(learning_rate)
    train_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir="logs/fit/" + train_time + '-' + SAVED_MODEL_NAME
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    PATH = './result/predictions/' + train_time + '-' + dataset_name + '-' + SAVED_MODEL_NAME


    # save checkpoint path
    checkpoint_path = "train_log/" + train_time + '-' + SAVED_MODEL_NAME + "/cp-epoch{epoch:04d}-loss{loss:5.4f}-val{val_loss:5.4f}.ckpt"
    # checkpoint_path = r'C:\Users\Pangzhentao\learn_keras\train_log\tune'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # if you want load model, reveal codes below, when a specific weight is demanded, change the root in checkpoint file
    # latest = tf.train.latest_checkpoint(checkpoint_path)
    # model.load_weights(latest)

    # when you need to save the training logs, use this callback function
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        period=0  # this parameter means how much epochs do you want to save weights
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['categorical_accuracy', CategoricalTruePositivesRecall(), MeanIoU(num_classes=2)],
        # metrics=['categorical_accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
        run_eagerly=True
    )

    # start training
    history = model.fit(
        train_data,
        epochs=epoch,
        validation_data=test_data,
        validation_freq=1,
        callbacks=[tensorboard_callback,
                   cp_callback,
                   callbacks.ShowAtEpoch(test_data, number=1, save=True, save_epochs=1, path=PATH),
                   callbacks.LearningRateChangeByLoss(patience=3, lr_change_time=3, change_rate=0.1)
                   #callbacks.AlphaScheduler(alpha=alpha, update_step=0.01, wait_epoch=0)
                   ]
        )

    # save the model
    SAVED_MODEL = './result/models/' + train_time + '-' + SAVED_MODEL_NAME
    tf.saved_model.save(model, SAVED_MODEL)
    print('saving savedmodel.')
    del model


    # # test
    # import Functions.show_img as simg

    # print('testing!!!')
    # model = tf.saved_model.load(SAVED_MODEL)
    # PATH = PATH + '/last_test'
    # simg.show_predict(number=10, source=test_data, model=model, show=False, save=True, path=PATH)
    # del model


