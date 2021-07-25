import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import datetime
import Functions.show_img as simg
import numpy as np
import os

'''
==================================================================
The function of CallBacks as follows：
1. record the processing of training
2. print the pred results after certain epochs
3. change the learning rate when loss stop decreasing 

Update_Time: 2021/6/2
Author: Wu Tong
==================================================================
'''

# 1. record the processing of training
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H.%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# 2. print the pred results after certain epochs
class ShowAtEpoch(Callback):

    def __init__(self, sourse=None, number=3, show=False, save=False, show_epochs=10, save_epochs=10, path=''):
        '''

        :param sourse: need dataset that you want to test
        :param number: means how many times do you want predict
        :param show: if you want to show predictions on the end of each epoch
        :param save: if you want to save predictions on the end of each epoch
        :param show_epochs: how often do you want to show
        :param save_epochs: how often do you want to save
        :param path: where do you want to save the predict results
        '''
        super(ShowAtEpoch, self).__init__()
        self.souse = sourse
        self.number = number
        self.show = show
        self.save = save
        self.show_epochs = show_epochs
        self.save_epochs = save_epochs
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_epochs == 0:
            self.save = True
        if epoch % self.show_epochs == 0:
            self.show = True
        if not os.path.exists(self.path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(self.path)
        if self.show or self.save:
            simg.show_predict(self.number, self.souse, self.model, save=self.save, path=self.path, epoch=epoch)
        self.save = False
        self.show = False


# 3. change the learning rate when loss stop decreasing
class LearningRateChangeByLoss(Callback):
    '''
    :param patience shows your max tolerance to wait for loss decreasing
    :param lr_change_time means how many times you like to change the lr before stop training.
    note that the initial number is 3, and each time you decrease the lr use rate to multiply.
    :param change_rate how many you want to decrease your lr every change, it initialized with value 10
    '''

    def __init__(self, patience=0, lr_change_time=3, change_rate=0.1):
        super(LearningRateChangeByLoss, self).__init__()
        self.patience = patience
        self.lr_change_time = lr_change_time
        self.change_rate = change_rate
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        # The epoch the learning rate change
        self.change_lr_epoch =[]
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Initialize the learning rate list
        self.learning_rate = [lr]

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Get the current learning rate from model's optimizer.
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if self.lr_change_time != 0.0:
                    lr *= self.change_rate
                    self.learning_rate += [lr]
                    self.change_lr_epoch += [epoch]
                    print('================================================', '\n')
                    print('lr变为：', lr, '\n')
                    print('================================================', '\n')
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                    self.lr_change_time -= 1
                    self.wait = 0
                else:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        if self.change_lr_epoch == []:
            print('Learning rate was not change during training!!!')
        else:
            print('Learning rate changed at epoch: ', self.change_lr_epoch, '\n')
            print('Learning rate changed to value: ', self.learning_rate)
        print('---------------------------------Train Finished!!!-----------------------------------')

class AlphaScheduler(Callback):

    def __init__(self, alpha, update_step, wait_epoch=3):
        super(AlphaScheduler, self).__init__()
        self.alpha = alpha
        self.update_step = update_step
        self.wait_epoch = wait_epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.wait_epoch != 0:
            self.wait_epoch -= 1
            print('wait: ', str(self.wait_epoch))
        else:
            alpha = tf.keras.backend.get_value(self.alpha)
            if alpha > 0.01:
                updated_alpha = alpha - self.update_step
                tf.keras.backend.set_value(self.alpha, updated_alpha)
            print('Now the rate between xL and BL is: ', [self.alpha.numpy(), 1-self.alpha.numpy()])



if __name__ == '__main__':
    from tensorflow import keras
    # Define the Keras model to add callbacks to
    def get_model():
        model = keras.Sequential()
        model.add(keras.layers.Dense(1, input_dim=784))
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
            loss="mean_squared_error",
            metrics=["mean_absolute_error"],
        )
        return model


    # Load example MNIST data and pre-process it
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # Limit the data to 1000 samples
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    model = get_model()
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=20,
        verbose=0,
        validation_split=0.5,
        callbacks=[LearningRateChangeByLoss(patience=2, lr_change_time=3, change_rate=0.1)],
    )

