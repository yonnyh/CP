import numpy as np
import matplotlib.pyplot as plt
import rawpy
from skimage.color import rgb2xyz


def sub(f_path, a_path):
    f = rawpy.imread(f_path)
    f_rgb = f.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    a = rawpy.imread(a_path)
    a_rgb = a.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    t1 = min_threshold(a_rgb, 0.02)
    not_ignore = a_rgb > t1
    delta = f_rgb - a_rgb
    t2 = min_threshold(delta, 0.02)
    not_ignore = not_ignore and delta > t2
    normed = a_rgb / delta
    c = np.average(normed, where=not_ignore)
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
    sub('input/withflash.CR2', 'input/noflash.CR2')

    # img_path = 'input/graycard.CR2'
    # img_path = 'input-tiff/graycard.tiff'
    # find_chromaticity_coordinates(img_path)
