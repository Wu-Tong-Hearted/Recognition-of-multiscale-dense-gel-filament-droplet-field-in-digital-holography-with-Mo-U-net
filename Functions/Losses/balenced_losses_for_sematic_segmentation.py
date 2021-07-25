#tf.__version__==2.4.0
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


class Weighted_Cross_Entropy_Loss(tf.keras.losses.Loss):

    def __init__(self, weights=[0.1, 0.9]):
        super(Weighted_Cross_Entropy_Loss, self).__init__()
        self.name = 'WCEL'
        self.weights = weights

    # 统计标签中不同类别的数量
    def count_category(self, y_true):
        NUMBER = y_true.shape[-1]
        category_list = []
        for i in range(NUMBER):
            category_list += [tf.reduce_sum(y_true[..., i]).numpy()]
        return category_list

    # 计算权重矩阵
    def count_weights(self, category_list=[]):
        all_number = sum(category_list)
        weights_list = []
        for i in category_list:
            weights_list += [(all_number - i) / all_number]
        return weights_list

    # 计算标签对应的权重矩阵
    def creat_weights_matrix(self, y_true, weights_list=[]):
        weights_list_matrix = tf.cast(tf.broadcast_to(weights_list, y_true.shape), dtype=tf.float32)
        weights_matrix = tf.multiply(weights_list_matrix, y_true)
        return weights_matrix

    def call(self, y_true, y_pred):
        # gain weights matrix
        category_list = self.count_category(y_true)
        weights_list = self.count_weights(category_list)
        weights_matrix = self.creat_weights_matrix(y_true, weights_list=weights_list)
        # calculate cross entropy loss
        log_y_pred = -tf.math.log(y_pred + 1e-7)
        multipled_y = tf.multiply(log_y_pred, y_true)
        result_matrix = tf.multiply(multipled_y, weights_matrix)
        result = tf.math.reduce_sum(result_matrix) / (y_pred.shape[-2] * y_pred.shape[-3])
        return result


class Weighted_Focal_Loss(tf.keras.losses.Loss):

    def __init__(self, gama=2):
        super(Weighted_Focal_Loss, self).__init__()
        self.name = 'WFL'
        self.gama = gama
        # self.alpha = param[1] #

    # calculate the statistic numbers of each category
    def count_category(self, y_true):
        NUMBER = y_true.shape[-1]
        category_list = []
        for i in range(NUMBER):
            category_list += [tf.reduce_sum(y_true[..., i]).numpy()]
        return category_list

    # calculate the number weights
    def count_number_weights(self, y_true, category_list=[]):
        all_number = sum(category_list)
        number_weights_list = []
        for i in category_list:
            number_weights_list += [(all_number - i) / all_number]
        weights_list_matrix = tf.cast(tf.broadcast_to(number_weights_list, y_true.shape), dtype=tf.float32)
        number_weights_matrix = tf.multiply(weights_list_matrix, y_true)
        return number_weights_matrix

    # calculate the difficulties weights
    def count_difficulty_weights(self, y_true, y_pred):
        multipled_y = tf.multiply(y_pred, y_true)
        difficulty_weights_matrix = (1 - multipled_y) ** self.gama
        return difficulty_weights_matrix

    def creat_weights_matrix(self, y_true, weights_list=[]):
        number_weights_matrix = weights_list[0]
        difficulty_weights_matrix = weights_list[1]
        weights_matrix = tf.multiply(number_weights_matrix, difficulty_weights_matrix)
        return weights_matrix

    def call(self, y_true, y_pred):
        category_list = self.count_category(y_true)
        number_weights_matrix = self.count_number_weights(y_true, category_list=category_list)
        difficulty_weights_matrix = self.count_difficulty_weights(y_true, y_pred)
        weights_matrix = self.creat_weights_matrix(y_true,
                                                   weights_list=[number_weights_matrix, difficulty_weights_matrix])

        log_y_pred = -tf.math.log(y_pred + 1e-7)
        multipled_y = tf.multiply(log_y_pred, y_true)
        result_matrix = tf.multiply(multipled_y, weights_matrix)
        result = tf.math.reduce_sum(result_matrix) / (y_pred.shape[-2] * y_pred.shape[-3])
        return result


class Focal_Loss(tf.keras.losses.Loss):

    def __init__(self, gama=2):
        super(Focal_Loss, self).__init__()
        self.name = 'FL'
        self.gama = gama

    def count_difficulty_weights(self, y_true, y_pred):
        multipled_y = tf.multiply(y_pred, y_true)
        difficulty_weights_matrix = (1 - multipled_y) ** self.gama
        return difficulty_weights_matrix

    def call(self, y_true, y_pred):
        difficulty_weights_matrix = self.count_difficulty_weights(y_true, y_pred)
        log_y_pred = -tf.math.log(y_pred + 1e-7)
        multipled_y = tf.multiply(log_y_pred, y_true)
        result_matrix = tf.multiply(multipled_y, difficulty_weights_matrix)
        result = tf.math.reduce_sum(result_matrix) / (y_pred.shape[-2] * y_pred.shape[-3])
        return result


class Self_Adaptive_Dice_Loss(tf.keras.losses.Loss):

    def __init__(self, smooth=1e-6, gama=1):
        super(Self_Adaptive_Dice_Loss, self).__init__()
        self.name = 'SADL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        nominator = 2 * tf.reduce_sum(tf.multiply(tf.multiply((1 - y_pred) ** self.gama, y_pred), y_true)) + self.smooth
        denominator = tf.reduce_sum(tf.multiply((1 - y_pred) ** self.gama, y_pred)) + tf.reduce_sum(y_true) + self.smooth
        result = tf.divide(nominator, denominator)
        return result


class Normal_Dice_Loss(tf.keras.losses.Loss):

    def __init__(self, smooth=1e-6, gama=2):
        super(Normal_Dice_Loss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


class Tversky_Loss(tf.keras.losses.Loss):
    '''
    当alpha和beta都等于0.5时，TL退化为DL；
    当alpha和beta都等于1时，TL退化为IOU Loss;
    一般来说alpha和beta和为1
    '''

    def __init__(self, smooth=1e-6, alpha=0.5, beta=0.5):
        super(Tversky_Loss, self).__init__()
        self.name = 'TL'
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        nominator = tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        false_neg = tf.multiply(y_true, 1 - y_pred)
        false_pos = tf.multiply(1 - y_true, y_pred)
        denominator = tf.reduce_sum(y_true * y_pred) + self.alpha * tf.reduce_sum(false_neg) + self.beta * tf.reduce_sum(false_pos) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result


class True_Boundary_Loss(tf.keras.losses.Loss):

    def __init__(self):
        super(True_Boundary_Loss, self).__init__()
        self.name = 'TBL'

    def dist_map(self, y_true_2D):
        y_true_2D = y_true_2D.numpy()
        dist_map = np.zeros_like(y_true_2D)
        posmask = y_true_2D.astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            dist_map = (distance(negmask) - 1) * negmask + distance(posmask) * posmask

        return dist_map

    def call(self, y_true, y_pred):
        y_pred_pos = y_pred[..., 1]
        y_true_2D = tf.argmax(y_true, axis=-1)
        y_true_dis_map = self.dist_map(y_true_2D)
        result = tf.reduce_sum(tf.multiply(y_pred_pos, y_true_dis_map))
        return result

class Boundary_Loss(tf.keras.losses.Loss):

    def __init__(self):
        super(Boundary_Loss, self).__init__()
        self.name = 'BL'

    def dist_map(self, y_true_2D):
        y_true_2D = y_true_2D.numpy()
        dist_map = np.zeros_like(y_true_2D)
        posmask = y_true_2D.astype(np.bool_)

        if posmask.any():
            negmask = ~posmask
            dist_map = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

        return dist_map

    def call(self, y_true, y_pred):
        y_pred_pos = y_pred[..., 1]
        y_true_2D = tf.argmax(y_true, axis=-1)
        y_true_dis_map = self.dist_map(y_true_2D)
        result = tf.reduce_sum(tf.multiply(y_pred_pos, y_true_dis_map)) / (y_pred.shape[-2] * y_pred.shape[-3])
        return result


# the input alpha should uses the form e.g., alpha = tf.keras.backend.variable(1, dtype='float32')


class SAD_B_Loss(tf.keras.losses.Loss):

    def __init__(self, alpha, gama=1):
        super(SAD_B_Loss, self).__init__()
        self.name = 'SADL+BL'
        self.alpha = alpha
        self.gama = gama

    def call(self, y_true, y_pred):
        SADL = Self_Adaptive_Dice_Loss(smooth=1e-6, gama=self.gama)
        BL = Boundary_Loss()
        return self.alpha * SADL.call(y_pred=y_pred, y_true=y_true) + (1 - self.alpha) * BL.call(y_pred=y_pred, y_true=y_true)


class ND_B_Loss(tf.keras.losses.Loss):

    def __init__(self, alpha, gama=1):
        super(ND_B_Loss, self).__init__()
        self.name = 'NDL+BL'
        self.alpha = alpha
        self.gama = gama

    def call(self, y_true, y_pred):
        NDL = Normal_Dice_Loss(smooth=1e-6, gama=self.gama)
        BL = Boundary_Loss()
        return self.alpha * NDL.call(y_pred=y_pred, y_true=y_true) + (1 - self.alpha) * BL.call(y_pred=y_pred, y_true=y_true)


class CE_B_Loss(tf.keras.losses.Loss):

    def __init__(self, alpha):
        super(CE_B_Loss, self).__init__()
        self.name = 'CEL+BL'
        self.alpha = alpha

    def call(self, y_true, y_pred):
        BL = Boundary_Loss()
        return self.alpha * tf.keras.metrics.categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=False) + \
            (1 - self.alpha) * BL.call(y_pred=y_pred, y_true=y_true)


if __name__ == '__main__':
    a = tf.constant([[[0.9, 0.1], [0.2, 0.8]], [[1, 0], [0.3, 0.7]]])
    b = tf.cast(tf.constant([[0,1],[1,1]]), dtype=tf.int32)
    b = tf.one_hot(b, depth=2)

    WCEL = Weighted_Cross_Entropy_Loss()
    FL = Focal_Loss(gama=2)
    SADL = Self_Adaptive_Dice_Loss(smooth=1e-6, gama=1)
    NDL = Normal_Dice_Loss(smooth=1e-6, gama=2)
    TL = Tversky_Loss(smooth=1e-6, alpha=0.5, beta=0.5)
    BL = Boundary_Loss()
    CE_B_L = CE_B_Loss(alpha=1)

    result_WCEL = WCEL.call(y_pred=a, y_true=b)
    result_FL = FL.call(y_pred=a, y_true=b)
    result_SADL = SADL.call(y_pred=a, y_true=b)
    result_NDL = NDL.call(y_pred=a, y_true=b)
    result_TL = TL.call(y_pred=a, y_true=b)
    result_BL = BL.call(y_pred=a, y_true=b)
    result_CE_B_L = CE_B_L(y_pred=a, y_true=b)