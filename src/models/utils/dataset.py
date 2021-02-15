import logging
import numpy as np
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import json
import fiona
from pathlib import Path
from torch.utils.data import random_split


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('model_utils_dataset')


def compute_mean_std(image_folder, n_ch):
    sum_channels = np.zeros(n_ch)  # 8, 3 or 1
    std_channels = np.zeros(n_ch)
    total_pixel = 0
    
    total_img = len(list(image_folder.iterdir()))

    for img in tqdm(image_folder.iterdir(),
                    desc="Mean computation",
                    total=total_img):
        with rasterio.open(img, 'r') as ds:
            try:
                arr = ds.read()
                 
            except:
                logger.info(f"Uh oh, {img.stem} seems to be corrupted...")
            else:
                arr = arr.reshape(arr.shape[0], -1)
                sum_channels += arr.sum(axis=-1)
                total_pixel += arr[0].size
            
    mean_channels = sum_channels / total_pixel

    for img in tqdm(image_folder.iterdir(),
                    desc="std computation",
                    total=total_img):
        with rasterio.open(img, 'r') as ds:
            try:
                arr = ds.read()
                 
            except:
                logger.info(f"Uh oh, {img.stem} seems to be corrupted...")
            else:
                arr = arr.reshape(arr.shape[0], -1)
                std_channels += np.sum((arr - mean_channels.reshape(n_ch, 1)) ** 2, axis=-1) 
            
    std_channels = np.sqrt(std_channels / total_pixel)

    stats = {'mean': mean_channels.tolist(), 'std': std_channels.tolist()}
    return stats


def load_stats(json_file):
    with open(json_file, 'r') as file:
        n_params = json.load(file)
    mean_channels = np.array(n_params['mean'])
    std_channels = np.array(n_params['std'])
    return mean_channels, std_channels


def norm_img(img, mean_arr, std_arr):
    res = (np.transpose(img, (1, 2, 0)) - mean_arr) / std_arr
    return np.transpose(res, (2, 0, 1))


def get_tif_dims(tif_file):
    with rasterio.open(tif_file) as tif:
        n_ch = tif.count
        w = tif.width
        h = tif.height
    return n_ch, h, w


def add_padding(arr, pad_h, pad_w):
    n_ch, h, w = arr.shape
    
    plus_h = np.zeros((n_ch, pad_h, w))
    plus_w = np.zeros((n_ch, h+pad_h, pad_w))
    
    temp = np.concatenate((arr, plus_h), axis=1)
    return np.concatenate((temp, plus_w), axis=2)


def load_tif(fn, df, mean_vec, std_vec, building_folder, padding=None):
    img_id = "_".join(Path(fn).stem.split("_")[1:])  # get img id

    no_building = df[df['BuildingId'] == -1]['ImageId'].unique().tolist()
    geojson_path = building_folder / f"buildings_{img_id}.geojson"

    # Extract the file as a (8 x 650 x 650) cube
    with rasterio.open(fn) as tif:
        arr = tif.read()
        info = tif.meta
    
    info['count'] = 1
    # Extract geofeatures if the image has buildings
    if img_id in no_building:
        X = np.zeros((info['height'], info['width']), dtype='uint16')
        features = []
    else:
        with fiona.open(geojson_path, "r") as geojson:
            features = [feature["geometry"] for feature in geojson]
        X = np.ones((info['height'], info['width']), dtype='uint16')

    # Write polygons as a tif whose dimensions are the same than the opened tif
    with rasterio.open('temp.tif', 'w', **info) as new_ds:
        new_ds.write(X, 1)
    
    # Extract mask if necessary
    with rasterio.open('temp.tif') as tif:
        if features:
            mask_img, _ = mask(tif, features)
        else:
            mask_img = tif.read()
    
    if padding:
        pad_h, pad_w = padding
        arr = add_padding(arr, pad_h, pad_w)
        mask_img = add_padding(mask_img, pad_h, pad_w)
    
    arr = norm_img(arr, mean_vec, std_vec)
    arr, mask_img = arr.astype('float32'), mask_img.squeeze().astype('int64')
    Path('temp.tif').unlink()

    return arr, mask_img


def split_dataset(ds, train_size=0.8):
    if type(train_size) is float:
        train_size = int(len(ds)*train_size)
    train_ds, val_ds = random_split(ds, (train_size, len(ds)-train_size))
    return train_ds, val_ds