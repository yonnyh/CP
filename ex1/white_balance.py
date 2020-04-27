import numpy as np
import matplotlib.pyplot as plt
import rawpy
from skimage.color import rgb2xyz


def sub(f_path, a_path, debug=False):
    if not debug:
        f = rawpy.imread(f_path)
        f_rgb = f.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
        a = rawpy.imread(a_path)
        a_rgb = a.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    else:
        a_rgb = np.arange(60).reshape(4, 5, 3)
        f_rgb = np.arange(60).reshape(4, 5, 3) / 2
    delta = f_rgb - a_rgb
    normed = a_rgb / delta

    t1 = min_threshold(a_rgb, 0.22)
    not_ignore1 = a_rgb > t1
    t2 = min_threshold(delta, 0.22)
    not_ignore2 = delta > t2
    not_ignore = (not_ignore1 & not_ignore2)
    c = np.average(normed, axis=(0, 1), weights=not_ignore)

    output = a_rgb / c
    plt.imshow(normed)
    plt.show()


def min_threshold(rgb_img, percentage: float):
    reshaped = rgb_img.transpose(2, 0, 1).reshape(3, -1)
    num_of_wanted_pics = int(reshaped.shape[1] * percentage)
    r = np.max(np.partition(reshaped[0], num_of_wanted_pics)[:num_of_wanted_pics])
    g = np.max(np.partition(reshaped[1], num_of_wanted_pics)[:num_of_wanted_pics])
    b = np.max(np.partition(reshaped[2], num_of_wanted_pics)[:num_of_wanted_pics])
    return np.array([r, g, b])


def find_chromaticity_coordinates(img_p):
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


if __name__ == '__main__':
    sub('input/withflash.CR2', 'input/noflash.CR2', debug=True)

    # img_path = 'input/graycard.CR2'
    # img_path = 'input-tiff/graycard.tiff'
    # find_chromaticity_coordinates(img_path)
