import numpy as np
import matplotlib.pyplot as plt
import rawpy
from skimage.io import imread
import cv2
import imageio
from skimage.color import rgb2xyz, xyz2rgb


XYZ2LMS_VON_KRIES = np.array([[ 0.4002400,  0.7076000, -0.0808100],
                              [-0.2263000,  1.1653200,  0.0457000],
                              [ 0.0000000,  0.0000000,  0.9182200]], dtype=np.float)

XYZ2LMS_BRADFORD = np.array([[ 0.8951000,  0.2664000, -0.1614000],
                             [-0.7502000,  1.7135000,  0.0367000],
                             [ 0.0389000, -0.0685000,  1.0296000]], dtype=np.float)

XYZ2LMS_HUNT_POINTER_ESTEVEZ = np.array([[ 0.3897,  0.6889 , -0.0786],
                                        [-0.2298,  1.1834,   0.0464],
                                        [ 0.0, 0.0,  1.0]], dtype=np.float)

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


def find_chromaticity_coordinates(img, four_points):
    img = crop_image(img, four_points)
    return np.max(img.transpose(2, 0, 1).reshape(3, -1), axis=1)


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


def normalize_image(img):
    img = img.astype(np.float) - np.min(img.astype(np.float))
    img /= np.max(img)
    return img


def gamma_corrections(img, gamma):
    invGamma = 1.0 / gamma
    img = normalize_image(img)
    return np.power(img, invGamma)


def xyz2lms(img, transform_matrix):
    shape_of_img = img.shape
    reshaped_img = img.transpose(2, 0, 1).reshape(3, -1)
    lms_img = np.dot(transform_matrix, reshaped_img)
    img = img.reshape(shape_of_img)
    return img


def lms2xyz(img):
    pass
if __name__ == '__main__':
    # img_path = 'input/graycard.CR2'
    # img_path = 'input-tiff/graycard.tiff'
    img_path = 'input-tiff/noflash.tiff'
    four_points = [650, 1819, 630, 1846, 990, 1037, 2333, 2411]
    # img = cv2.imread(img_path)
    img = imread(img_path)

    xyz2lms(rgb2xyz(img), XYZ2LMS_BRADFORD)
    # img = imageio.imread(img_path)

    # correct_img = gamma_corrections(img, 2.4)
    # display_image(correct_img)
    # plt.imshow(correct_img)
    # plt.imshow(img)

    # plt.show()


    # sub('input/withflash.CR2', 'input/noflash.CR2', debug=True)

    # img_path = 'input-tiff/graycard.tiff'
    # find_chromaticity_coordinates(img_path)
