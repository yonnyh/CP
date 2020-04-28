import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2xyz
from skimage.io import imread


def read_tiff(path):
    """Read image in tiff format"""
    img = imread(path)
    img = img.astype(np.float) - np.min(img.astype(np.float))
    img /= np.max(img)
    return img


def show_img(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def wb(flash_img, no_flash_img, flash_chromatic):
    # Article addition: divide by c (mean of no_flash_img / delta) instead of
    # directly by flash_chromatic
    delta = flash_img - no_flash_img
    balanced_delta = delta / flash_chromatic
    normed = np.divide(no_flash_img, balanced_delta, where=delta > 0)

    # compute extreme values to ignore
    t1 = min_threshold(no_flash_img, 0.02)
    not_ignore1 = no_flash_img > t1
    t2 = min_threshold(delta, 0.02)
    not_ignore2 = delta > t2
    not_ignore = (not_ignore1 & not_ignore2)

    c = np.average(normed, axis=(0, 1), weights=not_ignore)
    output = no_flash_img / c
    return output


def min_threshold(rgb_img, percentage: float):
    reshaped = rgb_img.transpose(2, 0, 1).reshape(3, -1)
    num_of_wanted_pics = int(reshaped.shape[1] * percentage)
    r = np.max(np.partition(reshaped[0], num_of_wanted_pics)[:num_of_wanted_pics])
    g = np.max(np.partition(reshaped[1], num_of_wanted_pics)[:num_of_wanted_pics])
    b = np.max(np.partition(reshaped[2], num_of_wanted_pics)[:num_of_wanted_pics])
    return np.array([r, g, b])


def find_chromaticity_coordinates(img_p, left, right, top, bot):
    img = crop_image(img_p, left, right, top, bot)
    return np.max(img.transpose(2, 0, 1).reshape(3, -1), axis=1)
    # raw = rawpy.imread(img_p)
    # rgb = raw.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    # x, y, z = rgb.shape
    # xyz = rgb2xyz(rgb)
    # xyz2lms = np.array([[0.3897,    0.6889,     -0.0786],
    #                     [-0.2298,   1.1834,     0.0464],
    #                     [0.0,       0.0,        1.0]])
    # xyz_reshaped = xyz.transpose(2, 0, 1).reshape(3, -1)
    # lms_reshaped = xyz2lms @ xyz_reshaped
    # lms = lms_reshaped.reshape(z, x, y).transpose(1, 2, 0)
    #
    # print()


def crop_image(img, left, right, top, bot):
    grey_card = img[top:bot, left:right]
    return grey_card


def main(noflash_path, flash_path, gray_card_path):
    no_flash_img = read_tiff(noflash_path)
    flash_img = read_tiff(flash_path)
    gray_card_img = read_tiff(gray_card_path)
    points_of_card = 686, 1842, 971, 2324  # for given images
    flash_cromatic = find_chromaticity_coordinates(gray_card_img, *points_of_card)

    balanced = wb(flash_img, no_flash_img, flash_cromatic)
    return balanced


if __name__ == '__main__':
    noflash_path = 'input-tiff/noflash.tiff'
    flash_path = 'input-tiff/withflash.tiff'
    gray_card_path = 'input-tiff/graycard.tiff'
    balanced_img = main(noflash_path, flash_path, gray_card_path)
    show_img(balanced_img, "WB")
