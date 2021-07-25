from scipy.ndimage import distance_transform_edt as distance
import numpy as np
import os
from PIL import Image


def PIOU(y_pred, y_true):
    # print(np.sum(y_pred * y_true))
    # print(np.sum((y_pred + y_true).astype(np.bool_)))
    return np.sum(y_pred * y_true) / (np.sum((y_pred + y_true).astype(np.bool_)))


def count_dist_map(x):
    dist_map = np.zeros_like(x)
    posmask = x.astype(np.bool_)

    if posmask.any():
        negmask = ~posmask
        dist_map = (distance(negmask) - 1) * negmask + distance(posmask) * posmask

    return dist_map


def ASSD(y_pred, y_true):
    boundary_pred = distance(y_pred) == 1
    dist_map = count_dist_map(y_true)
    return np.sum(boundary_pred * dist_map) / np.sum(boundary_pred)


def PRell(y_pred, y_true):
    return  np.sum(y_pred == y_true) / np.sum(y_true)


def PPrecise(y_pred, y_true):
    return  np.sum(y_pred == y_true) / np.sum(y_pred)


def read_path(root, pred, true):
    pred_list = sorted([os.path.join(root, pred, i) for i in os.listdir(os.path.join(root, pred))])
    true_list = sorted([os.path.join(root, true, i) for i in os.listdir(os.path.join(root, true))])
    return zip(pred_list, true_list)


def Count_Metric(metric=None, file_path=None, pred_file_name=None, true_file_nam=None):

    path_list = read_path(file_path, pred_file_name, true_file_nam)
    min_result = np.inf
    max_result = 0
    sum_result = 0
    num = 0

    for path in path_list:
        pred_path = path[0]
        true_path = path[1]
        pred = np.array(Image.open(pred_path))[..., 0] < 127.0  # 儅檢測model_pred時
        # pred = np.array(Image.open(pred_path)) > 127.0  # 儅檢測at_pred時
        true = np.array(Image.open(true_path)) > 127.0
        # print(pred.shape)
        # print(true.shape)

        if metric == 'PIOU':
            step_result = PIOU(pred, true)
            print(step_result)
        if metric == 'ASSD':
            step_result = ASSD(pred, true)
            print(step_result)
        if metric == 'PR':
            step_result = PRell(pred, true)
        if metric == 'PP':
            step_result = PPrecise(pred, true)

        if step_result < min_result:
            min_result = step_result
        if step_result > max_result:
            max_result = step_result

        sum_result += step_result
        num += 1

    fin_reslut = sum_result / float(num)

    return [fin_reslut, min_result, max_result]







if __name__ == '__main__':
    a = np.array([[0, 1], [0, 1]])
    b = np.array([[1, 0], [0, 1]])
    c = PIOU(a, b)
    d = ASSD(a, b)