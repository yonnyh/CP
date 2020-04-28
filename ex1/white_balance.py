import numpy as np
import matplotlib.pyplot as plt
import rawpy
from skimage.color import rgb2xyz
from skimage.io import imread
from PIL import Image


def sub(f_path, a_path, debug=False):
    if not debug:
        f = rawpy.imread(f_path)
        f_rgb = f.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
        a = rawpy.imread(a_path)
        a_rgb = a.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    else:
        d = np.zeros((4, 5, 3))
        d[[0, 1, 2], [0, 1, 2]] = np.ones((3,))
        f_rgb = np.arange(60).reshape(4, 5, 3) + d
        a_rgb = np.arange(60).reshape(4, 5, 3)
    delta = f_rgb - a_rgb
    # plt.imshow(delta / np.array([134, 120, 127]))
    # plt.show()
    normed = np.divide(a_rgb, delta, where=delta > 0)

    # compute values to ignore
    t1 = min_threshold(a_rgb, 0.02)
    not_ignore1 = a_rgb > t1
    t2 = min_threshold(delta, 0.02)
    not_ignore2 = delta > t2
    not_ignore = (not_ignore1 & not_ignore2)

    c = np.average(normed, axis=(0, 1), weights=not_ignore)
    output = a_rgb / c
    plt.imshow(output)
    plt.show()


def min_threshold(rgb_img, percentage: float):
    reshaped = rgb_img.transpose(2, 0, 1).reshape(3, -1)
    num_of_wanted_pics = int(reshaped.shape[1] * percentage)
    r = np.max(np.partition(reshaped[0], num_of_wanted_pics)[:num_of_wanted_pics])
    g = np.max(np.partition(reshaped[1], num_of_wanted_pics)[:num_of_wanted_pics])
    b = np.max(np.partition(reshaped[2], num_of_wanted_pics)[:num_of_wanted_pics])
    return np.array([r, g, b])


def find_chromaticity_coordinates(img_p, four_points):
    # img = crop_image(img_p, four_points)
    # return np.max(img.transpose(2, 0, 1).reshape(3, -1), axis=1)
    raw = rawpy.imread(img_p)
    rgb = raw.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    x, y, z = rgb.shape
    xyz = rgb2xyz(rgb)
    xyz2lms = np.array([[0.3897,    0.6889,     -0.0786],
                        [-0.2298,   1.1834,     0.0464],
                        [0.0,       0.0,        1.0]])
    xyz_reshaped = xyz.transpose(2, 0, 1).reshape(3, -1)
    lms_reshaped = xyz2lms @ xyz_reshaped
    lms = lms_reshaped.reshape(z, x, y).transpose(1, 2, 0)

    print()


def crop_image(img, four_points):
    x1, x2, x3, x4, y1, y2, y3, y4 = four_points
    top_left_x = min([x1, x2, x3, x4])
    top_left_y = min([y1, y2, y3, y4])
    bot_right_x = max([x1, x2, x3, x4])
    bot_right_y = max([y1, y2, y3, y4])
    grey_card = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
    return grey_card


def display_image(img):
    img = img.astype(np.float) - np.min(img.astype(np.float))
    img /= np.max(img)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # sub('input/withflash.CR2', 'input/noflash.CR2', debug=True)
    sub('input/withflash.CR2', 'input/noflash.CR2', debug=False)

    # img_path = 'input/graycard.CR2'
    # img_path = 'input-tiff/graycard.tiff'
    # find_chromaticity_coordinates(img_path)
