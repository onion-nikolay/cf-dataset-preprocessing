import cv2 as cv
import numpy as np
import os


def is_image(name):
    image_formats = ['jpg', 'png', 'tif', 'bmp']
    return os.path.splitext(name)[1][1:].lower() in image_formats


def read_images(path):
    names = np.sort([name for name in os.listdir(path) if is_image(name)])
    images = [cv.imread(os.path.join(path, name)) for name in names]
    return images, names


def read_one_image(path, name):
    return cv.imread(os.path.join(path, name))


def read_images_names(path):
    return np.sort([name for name in os.listdir(path) if is_image(name)])


def save_images(path, images, names):
    [cv.imwrite(os.path.join(path, nm), img) for nm, img in zip(names, images)]
    return 0


def save_one_image(path, image, name):
    cv.imwrite(os.path.join(path, name), image)
    return 0


def remove_chromakey(img, sens=35, chanel_to_remove='green'):
    reds = img[:, :, 2]
    greens = img[:, :, 1]
    blues = img[:, :, 0]
    # Refactoring in future
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


def dummy_resize(img, size):
    _sz = np.shape(img)[:-1]
    field = np.zeros((max(_sz), max(_sz), 3), dtype=np.uint8)
    if _sz[0] > _sz[1]:
        field[:, (_sz[0]-_sz[1])//2:(_sz[0]+_sz[1])//2+(_sz[1] % 2), :] = img
    else:
        field[(_sz[1]-_sz[0])//2:(_sz[1]+_sz[0])//2+(_sz[0] % 2), :, :] = img
    return resize(field, size)
