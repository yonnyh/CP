import numpy as np
import matplotlib.pyplot as plt
import rawpy
from skimage.color import rgb2xyz


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
    img_path = 'input/graycard.CR2'
    find_chromaticity_coordinates(img_path)
