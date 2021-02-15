import rasterio
import torch
import numpy as np


def norm_img(img, mean_arr, std_arr):
    res = (np.transpose(img, (1, 2, 0)) - mean_arr) / std_arr
    return np.transpose(res, (2, 0, 1))


def add_padding(arr, pad_h, pad_w):
    n_ch, h, w = arr.shape
    
    plus_h = np.zeros((n_ch, pad_h, w))
    plus_w = np.zeros((n_ch, h+pad_h, pad_w))
    
    temp = np.concatenate((arr, plus_h), axis=1)
    return np.concatenate((temp, plus_w), axis=2)


def prepare_img(fn, mean_ds, std_ds, padding=None, as_torch=False):
    with rasterio.open(fn) as ds:
        arr = ds.read()
    
    if padding:
        pad_h, pad_w = padding
        arr = add_padding(arr, pad_h, pad_w)
        
    arr = norm_img(arr, mean_ds, std_ds)
    return torch.from_numpy(arr.astype('float32')) if as_torch else arr.astype('float32')

