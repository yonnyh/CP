from imageio import imread
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

GAUSS_BASE = np.array([[1, 1]], dtype="float64")


def read_image(filename: str, representation: int):
    """
    Gets path of an image, and read it with two options of representations:
    RGB or grayscale style.
    :param filename: String of image path.
    :param representation: Int - 1 for grayscale, 2 for RGB.
    :return: An np.float64 image of the required representation.
    """
    im = imread(filename)
    im = im.astype(np.float64)
    im /= 255

    if representation == 1:
        im = rgb2gray(im)

    return im


def __create_one_dimension_gaussian(size):
    """
    Create gaussian with one dimension
    :param size: int, size of gaussian
    :return: numpy array, the gaussian
    """
    gaussian = np.array([[1]], dtype="float64")
    for i in range(size - 1):
        gaussian = convolve2d(gaussian, GAUSS_BASE)
    coefficient = np.sum(gaussian)

    return gaussian / coefficient


def __extension(small_im, kernel):
    """
    Extend given image by 2
    :param small_im: given image
    :param kernel: kernel for blurring
    :return: big image
    """
    extended = np.zeros([small_im.shape[0] * 2, small_im.shape[1] * 2])
    extended[::2, ::2] = small_im
    blurred = convolve(convolve(extended, kernel), np.array(kernel).T)

    return blurred


def __shrink(big_im, kernel):
    """
    Shrink given image by 2
    :param big_im: given image
    :param kernel: kernel for blurring
    :return: small image
    """
    blurred = convolve(convolve(big_im, kernel), np.array(kernel).T)
    sampled = blurred[::2, ::2]
    return sampled


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds gaussian pyramid.
    :param im: grayscale image.
    :param max_levels: max levels of resulting pyramid (include the original
        image).
    :param filter_size: odd int, the gaussian filter size.
    :return: tuple: pyramid as list, kernel
    """
    filter_vec = __create_one_dimension_gaussian(filter_size)
    gpyr = [im]
    for i in range(max_levels - 1):
        s_im = __shrink(gpyr[i], filter_vec)
        if s_im.shape[0] < 16 or s_im.shape[1] < 16:
            break
        gpyr.append(s_im)

    return gpyr, filter_vec


def blur_spatial(im, kernel_size):
    """
    Blur given image by gaussian in given size.
    :param im: float64 dtype.
    :param kernel_size: odd int.
    :return: float64 dtype blurred image.
    """
    gaussian_1d = __create_one_dimension_gaussian(kernel_size)
    return convolve(convolve(im, gaussian_1d), np.array(gaussian_1d).T)
