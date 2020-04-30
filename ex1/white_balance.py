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
    """
    Convert image from XYZ to LMS
    """
    x, y, z = img.shape
    reshaped_img = img.transpose(2, 0, 1).reshape(3, -1)
    lms_img = np.matmul(transform_matrix, reshaped_img)
    return lms_img.reshape(z, x, y).transpose(1, 2, 0)


def lms2xyz(img, transform_matrix):
    """
    Convert image from LMS to XYZ
    """
    l, m, s = img.shape
    reshaped_img = img.transpose(2, 0, 1).reshape(3, -1)
    xyz_img = np.matmul(np.linalg.inv(transform_matrix), reshaped_img)
    return xyz_img.reshape(s, l, m).transpose(1, 2, 0)


def rgb2lms(img, transform_matrix):
    """
    Convert image from RGB to LMS
    """
    return xyz2lms(rgb2xyz(img), transform_matrix)


def lms2rgb(img, transform_matrix):
    """
    Convert image from LMS to RGB
    """
    return xyz2rgb(lms2xyz(img, transform_matrix))


def read_tiff(path):
    """Read image in tiff format"""
    img = imread(path)
    img = img.astype(np.float) - np.min(img.astype(np.float))
    img /= np.max(img)
    return img


def show_img(img, title=""):
    """
    Show image
    :param img: Image to show
    :param title: Title to show
    :return:
    """
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.show()


def simple_wb(flash_img, no_flash_img, flash_chromatic, is_flash):
    """
    Simple white balance algorithm as we saw in class
    :param flash_img: Flash image
    :param no_flash_img: No flash image
    :param flash_chromatic: Flash chromaticity coordinates
    :param is_flash: Is the flush image need to correct
    :return: Correct image
    """
    # Correct the flush image
    if is_flash:
        delta = flash_img - no_flash_img
        balanced = delta @ (np.eye(3) * (1 / flash_chromatic))
    # Correct the no flush image
    else:
        flash_chromatic_day = np.array([0.9504, 1.0, 1.0889])[np.newaxis][np.newaxis]
        balanced = no_flash_img @ (np.eye(3) * (1 /flash_chromatic_day))
    return balanced


def wb(flash_img, no_flash_img, flash_chromatic, is_flash):
    """
    Our implementation to white balance algorithm
    :param flash_img: Flash img
    :param no_flash_img: no Flash img
    :param flash_chromatic: Flash chromaticity coordinates
    :param is_flash: Is the flush image need to correct
    :return: Correct image
    """
    # Article addition: divide by c (mean of no_flash_img / delta) instead of
    # directly by flash_chromatic

    delta = flash_img - no_flash_img
    balanced_delta = delta / flash_chromatic
    normed = np.divide(no_flash_img, balanced_delta, where=delta > 0)

    # compute extreme values to ignore
    t1 = np.percentile(no_flash_img, 2, axis=(0, 1))
    not_ignore1 = no_flash_img > t1
    t2 = np.percentile(delta, 2, axis=(0, 1))
    not_ignore2 = delta > t2
    not_ignore = (not_ignore1 & not_ignore2)

    c = np.average(normed, axis=(0, 1), weights=not_ignore)
    output = no_flash_img / c
    return output


def hist(img, title):
    """
    Show the histogram of the image
    """
    plt.hist(img.ravel(), bins=256, color='orange', )
    plt.hist(img[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
    plt.hist(img[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
    plt.hist(img[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.title(title)
    plt.show()


def find_chromaticity_coordinates(img_p, left, right, top, bot, lms_mat):
    """
    Find the chromaticity coordinates of the flush by the gary card image
    :param img_p: Gray card image
    :param lms_mat: The transform matrix to LMS
    :return: Chromaticity coordinates in RGB
    """
    img = crop_image(img_p, left, right, top, bot)

    # Transfer the image to LMS
    lms_img = rgb2lms(img, lms_mat)

    # Calculate the avg of each chanel
    avg_lms = np.mean(lms_img, axis=(0, 1))
    chromaticity = avg_lms / np.sum(avg_lms)
    return lms2rgb(chromaticity[np.newaxis][np.newaxis], lms_mat)[0, 0]


def crop_image(img, left, right, top, bot):
    """
    Crop the image (crop the gary card image)
    :param img: Gray card image
    :return: Gary card only
    """
    grey_card = img[top:bot, left:right]
    return grey_card


def gamma_corrections(img, gamma):
    """
    Correct the image by gama corrections
    :param img: The image to correct
    :param gamma: Factor to correct
    :return: The image after apply gama correction
    """
    inv_gamma = 1.0 / gamma
    return np.power(img, inv_gamma)


def main(noflash_path, flash_path, gray_card_path, lms_mat, is_transfer_to_lms, is_flash, wb_func=wb):
    """
    Run thh white balance algorithm
    :param noflash_path: The path of the no flush image
    :param flash_path: The path of the flush image
    :param gray_card_path: The path of the gray card image
    :param lms_mat: The transform matrix to LMS
    :param is_transfer_to_lms: If true transfer the image to the LMS, else don't transfer
    :param is_flash: Is the flash image need to be correct by white balance
    :param wb_func: White balance function (simple or our)
    :return: The correct image
    """
    # Read images
    no_flash_img = read_tiff(noflash_path)
    flash_img = read_tiff(flash_path)
    gray_card_img = read_tiff(gray_card_path)
    points_of_card = 686, 1842, 971, 2324  # for given images

    # Calculate the flash chromaticity
    flash_chromaticity = find_chromaticity_coordinates(gray_card_img, *points_of_card, lms_mat)

    # Transfer the image to the LMS
    if is_transfer_to_lms:
        balanced = lms2rgb(wb_func(rgb2lms(flash_img, lms_mat), rgb2lms(no_flash_img, lms_mat),
                           rgb2lms(flash_chromaticity[np.newaxis][np.newaxis], lms_mat),
                                   is_flash), lms_mat)
    # Work on RGB image
    else:
        balanced = wb_func(flash_img, no_flash_img, flash_chromaticity, is_flash)
    return balanced


def show_result(algo, is_flash, name_algo, noflash_path, flash_path, gray_card_path):
    """
    Show result according to the algorithm
    :param algo: Simple or our algorithm
    :param is_flash: Is the image to correct by white balance is the flush image
    :param name_algo: String that represent the name of the alogrithm
    :param noflash_path: The path of the no flush image
    :param flash_path: The path of the flush image
    :param gray_card_path: The path of the gray card image
    :return:
    """
    if is_flash:
        image_mode = "flash"
    else:
        image_mode = "no flash"

    # goes over all transform matrix to LMS
    for tran_matrix in [XYZ2LMS_VON_KRIES, XYZ2LMS_BRADFORD]:
        if np.all(tran_matrix == XYZ2LMS_VON_KRIES):
            trar_name = "VON KRIES"
        else:
            trar_name = "BRADFORD"
        # show the result with LMS transform and without
        for is_transfer_to_lms in [True, False]:

            if is_transfer_to_lms:
                # correct the image by white balance algorithm
                balanced_img = main(noflash_path, flash_path, gray_card_path, tran_matrix,
                                    is_transfer_to_lms, is_flash, algo)
                show_img(gamma_corrections(balanced_img, 2.4), name_algo+" white balance for "
                                                                         ""+image_mode+" image "
                                                               "with gama correction correction\n Use " +trar_name+" matrix to transfer to lms")
            else:
                # correct the image by white balance algorithm
                balanced_img = main(noflash_path, flash_path, gray_card_path, tran_matrix,
                                    is_transfer_to_lms, is_flash, algo)
                show_img(gamma_corrections(balanced_img, 2.4),
                         name_algo+" white balance for "+ image_mode +" image with gama correction \n Use "
                         +trar_name+" matrix to transfer to lms only for chromaticity")


if __name__ == '__main__':
    noflash_path = 'input-tiff/noflash.tiff'
    flash_path = 'input-tiff/withflash.tiff'
    gray_card_path = 'input-tiff/graycard.tiff'
    # show_result(simple_wb, False, "simple", noflash_path, flash_path, gray_card_path)
    # show_result(simple_wb, True, "simple", noflash_path, flash_path, gray_card_path)
    show_result(wb, False, "Our", noflash_path, flash_path, gray_card_path)

