import cv2 as cv
import numpy as np
import os


def dummy_resize(img, size):
    _sz = np.shape(img)[:-1]
    field = np.zeros((max(_sz), max(_sz), 3), dtype=np.uint8)
    if _sz[0] > _sz[1]:
        field[:, (_sz[0]-_sz[1])//2:(_sz[0]+_sz[1])//2+(_sz[1] % 2), :] = img
    else:
        field[(_sz[1]-_sz[0])//2:(_sz[1]+_sz[0])//2+(_sz[0] % 2), :, :] = img
    return resize(field, size)


def is_image(name):
    image_formats = ['jpg', 'png', 'tif', 'bmp']
    return os.path.splitext(name)[1][1:].lower() in image_formats


def put_on_color_field(images, background):
    shape = np.shape(images[0])
    if background == 'black':
        field = np.zeros(shape, dtype=np.uint8)
    elif background == 'white':
        field = 255*np.ones(shape, dtype=np.uint8)
    elif background == 'gray':
        field_color = np.mean([np.mean(img[img != 0]) for img in images])
        field = field_color * np.ones(shape, dtype=np.uint8)
    else:
        raise NameError('incorrect background name!')
    result = []
#    print(np.mean(field))
    for img in images:
        img[img < 3] = field[img < 3]
        result.append(np.uint8(img))
    return result


def put_on_square_field(img, field_size, x0, y0, size):
    result = np.zeros((field_size, field_size, 3), dtype=np.uint8)
    resized_img = cv.resize(img, (size, size))
    result[x0:x0+size, y0:y0+size] = resized_img
    return result


def read_images(path):
    names = np.sort([name for name in os.listdir(path) if is_image(name)])
    images = [cv.imread(os.path.join(path, name)) for name in names]
    return images, names


def read_one_image(path, name):
    return cv.imread(os.path.join(path, name))


def read_images_names(path):
    return np.sort([name for name in os.listdir(path) if is_image(name)])


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


def resize(img, size):
    return cv.resize(img, (size, size), interpolation=cv.INTER_CUBIC)


def save_images(path, images, names):
    [cv.imwrite(os.path.join(path, nm), img) for nm, img in zip(names, images)]
    return 0


def save_one_image(path, image, name):
    cv.imwrite(os.path.join(path, name), image)
    return 0


def select_square(img0, return_params=False):
    img_size = np.shape(img0)[:-1]
    max_size = max(img_size)
    img = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    img[max_size // 2 - img_size[0] // 2:
        max_size // 2 + img_size[0] // 2 + img_size[0] % 2,
        max_size // 2 - img_size[1] // 2:
        max_size // 2 + img_size[1] // 2 + img_size[1] % 2] = img0
    mask = cv.medianBlur(img[:, :, 0], 3) > 15
    minmax = np.argwhere(mask == 1)
    x_min, y_min = np.min(minmax, axis=0)
    x_max, y_max = np.max(minmax, axis=0)
    sz = [x_max-x_min, y_max-y_min]
#    Info: will be removed soon
#    print(x_min, x_max, y_min, y_max, sz)
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
