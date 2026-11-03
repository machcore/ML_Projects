import numpy as np


def cal_Pred_vals(slop, bias, x):
    return slop * x + bias

def calc_err(y_bar, y):
    return y_bar - y

def lossFunc(n, y_bar, y):
    return np.sum(calc_err(y_bar, y)**2) / (2*n)

def gradient_w(y_bar, y, x, n):
    return np.sum(calc_err(y_bar, y) * x) / n

def gradient_b(y_bar, y, n):
    return np.sum(calc_err(y_bar, y)) / n

def new_w(old_slop, grdw, alpha):
    return old_slop - alpha * grdw

def new_b(old_bias, grdb, alpha):
    return old_bias - alpha * grdb