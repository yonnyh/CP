import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2xyz, xyz2rgb
from skimage.io import imread


XYZ2LMS_VON_KRIES = np.array([[ 0.4002400,  0.7076000, -0.0808100],
                              [-0.2263000,  1.1653200,  0.0457000],
                              [ 0.0000000,  0.0000000,  0.9182200]], dtype=np.float)

XYZ2LMS_BRADFORD = np.array([[ 0.8951000,  0.2664000, -0.1614000],
                             [-0.7502000,  1.7135000,  0.0367000],
                             [ 0.0389000, -0.0685000,  1.0296000]], dtype=np.float)

XYZ2LMS_HUNT_POINTER_ESTEVEZ = np.array([[0.3897,  0.6889, -0.0786],
                                        [-0.2298,  1.1834,   0.0464],
                                        [0.0, 0.0,  1.0]], dtype=np.float)


def xyz2lms(img, transform_matrix):
    x, y, z = img.shape
    reshaped_img = img.transpose(2, 0, 1).reshape(3, -1)
    lms_img = np.matmul(transform_matrix, reshaped_img)
    return lms_img.reshape(z, x, y).transpose(1, 2, 0)


def lms2xyz(img, transform_matrix):
    l, m, s = img.shape
    reshaped_img = img.transpose(2, 0, 1).reshape(3, -1)
    xyz_img = np.matmul(np.linalg.inv(transform_matrix), reshaped_img)
    return xyz_img.reshape(s, l, m).transpose(1, 2, 0)


def rgb2lms(img, transform_matrix):
    return xyz2lms(rgb2xyz(img), transform_matrix)


def lms2rgb(img, transform_matrix):
    return xyz2rgb(lms2xyz(img, transform_matrix))


def read_tiff(path):
    """Read image in tiff format"""
    img = imread(path)
    img = img.astype(np.float) - np.min(img.astype(np.float))
    img /= np.max(img)
    return img


def show_img(img, title=""):
    plt.imshow((img*255).astype(np.uint8))
    plt.title(title)
    plt.show()


def simple_wb(flash_img, no_flash_img, flash_chromatic):
    delta = flash_img - no_flash_img
    balanced_delta = delta / flash_chromatic
    return no_flash_img + balanced_delta


def wb(flash_img, no_flash_img, flash_chromatic, lms_mat=None):
    # Article addition: divide by c (mean of no_flash_img / delta) instead of
    # directly by flash_chromatic
    if np.any(lms_mat):
        flash_img = rgb2lms(flash_img, lms_mat)
        no_flash_img = rgb2lms(no_flash_img, lms_mat)
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
    if np.any(lms_mat):
        return lms2rgb(output, lms_mat)
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
    chromaticity_coordinates = np.sum(img.transpose(2, 0, 1).reshape(3, -1), axis=1, dtype=np.float)
    # return np.max(img.transpose(2, 0, 1).reshape(3, -1), axis=1)
    return chromaticity_coordinates/np.sum(chromaticity_coordinates, dtype=np.float)

def crop_image(img, left, right, top, bot):
    grey_card = img[top:bot, left:right]
    return grey_card


def gamma_corrections(img, gamma):
    inv_gamma = 1.0 / gamma
    return np.power(img, inv_gamma)


def main(noflash_path, flash_path, gray_card_path, wb_func=wb, lms_mat=None):
    no_flash_img = read_tiff(noflash_path)
    flash_img = read_tiff(flash_path)
    gray_card_img = read_tiff(gray_card_path)
    points_of_card = 686, 1842, 971, 2324  # for given images
    flash_cromatic = find_chromaticity_coordinates(gray_card_img, *points_of_card)

    # balanced = wb_func(flash_img, no_flash_img, flash_cromatic, lms_mat)
    balanced = wb_func(flash_img, no_flash_img, flash_cromatic)
    return balanced


if __name__ == '__main__':
    noflash_path = 'input-tiff/noflash.tiff'
    flash_path = 'input-tiff/withflash.tiff'
    gray_card_path = 'input-tiff/graycard.tiff'

    # simple_balanced = main(noflash_path, flash_path, gray_card_path, wb_func=simple_wb)
    # show_img(simple_balanced, "simple WB")
    #
    # gamma = 2.4
    # simple_gamma_corrected = gamma_corrections(simple_balanced, gamma=gamma)
    # show_img(simple_gamma_corrected, f"simple WB + gamma ({gamma})")
    #
    # balanced_img = main(noflash_path, flash_path, gray_card_path, wb_func=wb,
    #                     lms_mat=np.eye(3, dtype=np.float))
    # balanced_img = main(noflash_path, flash_path, gray_card_path, wb_func=wb,
    #                     lms_mat=None)
    balanced_img = main(noflash_path, flash_path, gray_card_path, wb_func=simple_wb, lms_mat=None)
    show_img(gamma_corrections(balanced_img, 2.4), "wb")
    # flash_img = read_tiff(flash_path)
    # show_img(flash_img, "flash image")
    # no_flash_img = read_tiff(noflash_path)
    # show_img(no_flash_img, "no_flash image")


    # gamma_corrected = gamma_corrections(balanced_img, gamma=gamma)
    # show_img(gamma_corrected, f"article WB + gamma ({gamma})")
    # img = read_tiff(noflash_path)
    # balanced_img = main(noflash_path, flash_path, gray_card_path)
    # show_img(balanced_img, "WB")
    # rgb = xyz2rgb(lms2xyz(xyz2lms(rgb2xyz(img), XYZ2LMS_VON_KRIES), XYZ2LMS_VON_KRIES))
    # plt.imshow(gamma_corrections(rgb, 2.4))
    # plt.show()
    # print("")
