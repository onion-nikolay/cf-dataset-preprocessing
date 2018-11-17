import configparser
import os
from image_processing import read_images, remove_chromakey, put_on_square_field
from image_processing import save_images
import numpy as np
import cv2 as cv


def read_config():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    cfg = {}
    cfg['background'] = config['SETTINGS']['background']
    cfg['render_size'] = int(config['SETTINGS']['render_size'])
    cfg['final_size'] = int(config['SETTINGS']['final_size'])
    cfg['photo_folder'] = config['SETTINGS']['photo_folder']
    cfg['render_folder'] = config['SETTINGS']['render_folder']
    cfg['save_all'] = config['SETTINGS']['save_all']
    return cfg


def save_images_where_necessary(trig, folder, step, images, names):
    if trig:
        [cv.imwrite(folder+'\\step_{}\\'.format(step)+nm, img)
         for img, nm in zip(images, names)]
    return 0


def select_square(img0, return_params=False):
    img_size = np.shape(img0)[:-1]
    max_size = max(img_size)
    img = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    img[max_size // 2 - img_size[0] // 2:
        max_size // 2 + img_size[0] // 2 + img_size[0] % 2,
        max_size // 2 - img_size[1] // 2:
        max_size // 2 + img_size[1] // 2 + img_size[1] % 2] = img0
    mask = (img[:, :, 0] != 0)
    minmax = np.argwhere(mask == 1)
    x_min, y_min = np.min(minmax, axis=0)
    x_max, y_max = np.max(minmax, axis=0)
    sz = [x_max-x_min, y_max-y_min]
    print(x_min, x_max, y_min, y_max, sz)
    if sz[0] > sz[1]:
        x0 = x_min
        y0 = y_min-(sz[0]-y_max+y_min)//2
        _size = sz[0]
    else:
        x0 = x_min-(sz[1]-x_max+x_min)//2
        y0 = y_min
        _size = sz[1]
    if return_params:
        return {'x0': x0, 'y0': y0, 'size': _size}
    else:
        return img[x0:x0+_size, y0:y0+_size, :]


def session():
    cfg = read_config()
    if not os.path.isdir(cfg['photo_folder']):
        raise OSError('incorrect path to the folder with photos!')
    if not os.path.isdir(cfg['render_folder']):
        raise OSError('incorrect path to the folder with renders!')
    current_folders = [
            p for p in os.listdir('output') if os.path.isdir('output\\'+p)]
    index = 0 if len(current_folders) == 0 else int(current_folders[-1][-4:])+1
    new_folder = 'output\\result{:04d}'.format(index)
    os.mkdir(new_folder)
    if cfg['save_all']:
        folders = ['step_{}'.format(i+1) for i in range(3)]
        [os.mkdir(new_folder+'\\'+f) for f in folders]
    os.mkdir(new_folder+'\\color')
    os.mkdir(new_folder+'\\grayscale')

    images, names = read_images(cfg['photo_folder'])
    # Step 1. Removing background
    images_1 = [remove_chromakey(img, chanel_to_remove=cfg['background'],
                                 sens=20)
                for img in images]
    save_images_where_necessary(cfg['save_all'], new_folder, 1, images_1,
                                names)
    # Step 2. Select tank from photo
    images_2 = [select_square(img) for img in images_1]
    save_images_where_necessary(cfg['save_all'], new_folder, 2, images_2,
                                names)
    # Step 3. Select parameters
    render_images, _ = read_images(cfg['render_folder'])
    params = [select_square(img, True) for img in render_images]
    images_3 = [put_on_square_field(img, cfg['final_size'], **prm)
                for img, prm in zip(images_2, params)]
    save_images_where_necessary(cfg['save_all'], new_folder, 3, images_3,
                                names)
    images_grayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        for img in images_3]
    save_images(new_folder+'\\color', images_3, names)
    save_images(new_folder+'\\grayscale', images_grayscale, names)
    return 0


if __name__ == '__main__':
    temp = session()
