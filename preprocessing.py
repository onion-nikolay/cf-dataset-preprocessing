import configparser
import os
import image_processing as IP
import numpy as np
import cv2 as cv
from os.path import join as pjoin


def read_config():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    cfg = {}
    cfg['background'] = config['PREPROCESING']['background']
    cfg['render_size'] = int(config['PREPROCESING']['render_size'])
    cfg['final_size'] = int(config['PREPROCESING']['final_size'])
    cfg['photo_folder'] = pjoin(
                *config['PREPROCESING']['photo_folder'].split('\\'))
    cfg['render_folder'] = pjoin(
                *config['PREPROCESING']['render_folder'].split('\\'))
    cfg['save_all'] = int(config['PREPROCESING']['save_all'])
    cfg['by_one'] = int(config['PREPROCESING']['by_one'])
    return cfg


def save_images_where_necessary(trig, folder, step, images, names):
    if trig:
        [cv.imwrite(pjoin(folder, 'step_{}'.format(step), nm), img)
         for img, nm in zip(images, names)]
    return 0


#def session():
#    cfg = read_config()
#    if not os.path.isdir(cfg['photo_folder']):
#        print(cfg['photo_folder'])
#        raise OSError('incorrect path to the folder with photos!')
#    if not os.path.isdir(cfg['render_folder']):
#        raise OSError('incorrect path to the folder with renders!')
#    current_folders = [p for p in os.listdir('output') if os.path.isdir(
#                pjoin('output', p))]
#    index = 0 if len(current_folders) == 0 else int(
#                np.sort(current_folders)[-1][-4:])+1
#    new_folder = pjoin('output', 'result{:04d}'.format(index))
#    os.mkdir(new_folder)
#    if cfg['save_all']:
#        folders = ['step_{}'.format(i+1) for i in range(3)]
#        [os.mkdir(pjoin(new_folder, f)) for f in folders]
#    os.mkdir(pjoin(new_folder, 'color'))
#    os.mkdir(pjoin(new_folder, 'grayscale'))
#
#    images, names = IP.read_images(cfg['photo_folder'])
#    # Step 1. Removing background
#    images = [IP.remove_chromakey(img, chanel_to_remove=cfg['background'],
#                                  sens=20) for img in images]
#    save_images_where_necessary(cfg['save_all'], new_folder, 1, images, names)
#    # Step 2. Select tank from photo
#    images = [IP.select_square(img) for img in images]
#    save_images_where_necessary(cfg['save_all'], new_folder, 2, images, names)
#    # Step 3. Select parameters
#    render_images, _ = IP.read_images(cfg['render_folder'])
#    params = [IP.select_square(img, True) for img in render_images]
#    images = [IP.put_on_square_field(img, cfg['final_size'], **prm)
#              for img, prm in zip(images, params)]
#    save_images_where_necessary(cfg['save_all'], new_folder, 3, images, names)
#    images_grayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in images]
#    IP.save_images(pjoin(new_folder, 'color'), images, names)
#    IP.save_images(pjoin(new_folder, 'grayscale'), images_grayscale,
#                   names)
#    return 0


def session():
    cfg = read_config()
    if not os.path.isdir(cfg['photo_folder']):
        raise OSError('incorrect path to the folder with photos!')
    if not os.path.isdir(cfg['render_folder']):
        raise OSError('incorrect path to the folder with renders!')
    current_folders = [p for p in os.listdir('output') if os.path.isdir(
                pjoin('output', p))]
    index = 0 if len(current_folders) == 0 else int(
                np.sort(current_folders)[-1][-4:])+1
    new_folder = pjoin('output', 'result{:04d}'.format(index))
    os.mkdir(new_folder)
    if cfg['save_all']:
        folders = ['step_{}'.format(i+1) for i in range(2)]
        [os.mkdir(pjoin(new_folder, f)) for f in folders]
    os.mkdir(pjoin(new_folder, 'color'))
    os.mkdir(pjoin(new_folder, 'grayscale'))
    # Refactoring
    if cfg['by_one']:
        names_photo = IP.read_images_names(cfg['photo_folder'])
        names_render = IP.read_images_names(cfg['render_folder'])
        for name_p, name_r in zip(names_photo, names_render):
            image_p = IP.read_one_image(cfg['photo_folder'], name_p)
            # Step 1. Removing background
            image_1 = IP.remove_chromakey(image_p,
                                          chanel_to_remove=cfg['background'],
                                          sens=20)
            save_images_where_necessary(
                    cfg['save_all'], new_folder, 1, [image_1], [name_p])
            # Step 2. Put tank on square and resize.
            image_2 = IP.dummy_resize(image_1, cfg['final_size'])
            save_images_where_necessary(
                    cfg['save_all'], new_folder, 2, [image_2], [name_p])
            # Final result
            image_g = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)
            IP.save_one_image(pjoin(new_folder, 'color'), image_2, name_p)
            IP.save_one_image(pjoin(new_folder, 'grayscale'), image_g, name_p)
    else:
        images, names = IP.read_images(cfg['photo_folder'])
        # Step 1. Removing background
        images = [IP.remove_chromakey(img, chanel_to_remove=cfg['background'],
                                      sens=20) for img in images]
        save_images_where_necessary(cfg['save_all'], new_folder, 1, images,
                                    names)
        # Step 2. Put tank on square and resize.
        images = [IP.dummy_resize(img) for img in images]
        save_images_where_necessary(cfg['save_all'], new_folder, 2, images,
                                    names)
        # Final result
        images_grayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in
                            images]
        IP.save_images(pjoin(new_folder, 'color'), images, names)
        IP.save_images(pjoin(new_folder, 'grayscale'), images_grayscale,
                       names)
    return 0


if __name__ == '__main__':
    session()
