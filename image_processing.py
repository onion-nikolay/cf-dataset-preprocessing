import cv2 as cv
import numpy as np
import os


def is_image(name):
    image_formats = ['jpg', 'png', 'tif', 'bmp']
    return os.path.splitext(name)[1][1:].lower() in image_formats


def read_images(path):
    names = [name for name in os.listdir(path) if is_image(name)]
    images = [cv.imread(path+'\\'+name) for name in names]
    return images, names


def save_images(path, images, names):
    [cv.imwrite(path+'\\'+nm, img) for nm, img in zip(names, images)]
    return 0


def remove_chromakey(img, sens=35, chanel_to_remove='green'):
    reds = img[:, :, 2]
    greens = img[:, :, 1]
    blues = img[:, :, 0]
    if chanel_to_remove == 'green':
        mask = (greens < sens) | (reds > greens) | (blues > greens)
    elif chanel_to_remove == 'blue':
        mask = (blues < sens) | (reds > blues) | (greens > blues)
    elif chanel_to_remove == 'red':
        mask = (reds < sens) | (blues > reds) | (greens > reds)
    else:
        raise TypeError("Chanel {} not found!".format(chanel_to_remove))
    empty_img = np.zeros_like(img)
    empty_img[mask, 0] = 255
    result = np.zeros_like(img)
    for index in range(3):
        result[:, :, index] = img[:, :, index]*np.where(mask, 1, 0)
    return result


def put_on_square_field(img, field_size, x0, y0, size):
    result = np.zeros((field_size, field_size, 3), dtype=np.uint8)
    resized_img = cv.resize(img, (size, size))
    result[x0:x0+size, y0:y0+size] = resized_img
    return result


def resize(img, size):
    return cv.resize(img, (size, size), interpolation=cv.INTER_CUBIC)
