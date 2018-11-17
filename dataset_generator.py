import configparser
import image_processing as IP
import numpy as np
import os
from os.path import join as pjoin


def read_config():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    cfg = {}
    cfg['backgrounds'] = config['DATASETS']['backgrounds'].split(', ')
    cfg['sizes'] = [int(s) for s in config['DATASETS']['sizes'].split(',')]
    cfg['folder'] = pjoin(*config['DATASETS']['folder'].split('\\'))
    return cfg


def session():
    cfg = read_config()
    current_folders = [p for p in os.listdir('output') if os.path.isdir(
                pjoin('output', p))]
    index = 0 if len(current_folders) == 0 else int(
                np.sort(current_folders)[-1][-4:])+1
    if not os.path.isdir(cfg['folder']):
        cfg['folder'] = pjoin('output', np.sort(current_folders)[-1],
                              'grayscale')
    new_folder = pjoin('output', 'result{:04d}'.format(index))
    os.mkdir(new_folder)
    for _size in cfg['sizes']:
        for color in cfg['backgrounds']:
            images, names = IP.read_images(cfg['folder'])
            names = [name[:-3]+'bmp' for name in names]
            os.mkdir(pjoin(new_folder, '{:03d}_{}'.format(_size, color)))
            colored_images = IP.put_on_color_field(images, color)
            images_final = [IP.resize(img, _size) for img in colored_images]
            IP.save_images(pjoin(new_folder, '{:03d}_{}'.format(_size, color)),
                           images_final, names)
    return 0


if __name__ == '__main__':
    session()
