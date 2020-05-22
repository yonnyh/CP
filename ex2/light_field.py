import numpy as np
import matplotlib.pyplot as plt
from ex2.impro_4 import *
from os import walk
import cv2
from scipy.ndimage.interpolation import shift
from skimage.transform import EuclideanTransform, warp
from skimage.color import rgb2gray


def all_images(dir_path, gray=False):
    """
    Return list of all images form given directory path
    """
    f = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        f.extend(filenames)
        break
    names = sorted(f)

    imgs = []
    for im in names:
        new_im = plt.imread(dir_path + f"{im}")
        if not gray:
            imgs.append(new_im)
        else:
            gray_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
            imgs.append(gray_im)
    return imgs


def homography(curr_img, prev_img, debug=False):
    """
    Find 3x3 homography between prev_img to img
    :param img, prev_img: np.ndarray in dtype of uint8
    :return: np.ndarray with shape (3, 3)
    """
    if prev_img is None:
        return

    # prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    # curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    # src_pts = cv2.goodFeaturesToTrack(prev_gray, 100, 0.1, 3)
    # dst_pts = cv2.goodFeaturesToTrack(curr_gray, 100, 0.1, 3)
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    orb = cv2.ORB_create()
    kpt1, des1 = orb.detectAndCompute(prev_img, None)
    kpt2, des2 = orb.detectAndCompute(curr_img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2)
    dst_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if debug:
        mask = np.ravel(mask) > 0
        src_pts = src_pts[mask].reshape(-1, 2)
        dst_pts = dst_pts[mask].reshape(-1, 2)
        xs = [src_pts[:, 0], dst_pts[:, 0] + prev_img.shape[1]]
        ys = [src_pts[:, 1], dst_pts[:, 1]]
        im_h = cv2.hconcat([prev_img, curr_img])
        plt.imshow(im_h)
        plt.plot(xs, ys, mfc='r', c='b', lw=.2, ms=3, marker='o')
        plt.show()

    return H


def homography2(curr_img, prev_img, debug=False):
    points_and_descriptors = []
    for image in (curr_img, prev_img):
        image = sol4_utils.read_image(image, 1)
        pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
        points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    for i in range(len(points_and_descriptors) - 1):
        points1, points2 = points_and_descriptors[i][0], \
                           points_and_descriptors[i + 1][0]
        desc1, desc2 = points_and_descriptors[i][1], \
                       points_and_descriptors[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = match_features(desc1, desc2, .7)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        H12, inliers = ransac_homography(points1, points2, 100, 6)

        # Uncomment for debugging: display inliers and outliers among
        # matching points.
        # In the submitted code this function should be commented out!
        # display_matches(self.images[i], self.images[i+1], points1 ,
        # points2, inliers)

        return H12


class LightField:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.images_paths = self._get_images_paths()

        # load homographies and transformed images (rotation and y axis only)
        self._load_homographies()
        self._load_images()

    def _get_images_paths(self):
        f = []
        for (dirpath, dirnames, filenames) in walk(self.dir_path):
            f.extend(filenames)
            break
        return sorted(f)

    def _load_homographies(self):
        points_and_descriptors = []
        for im in self.images_paths:
            image = sol4_utils.read_image(self.dir_path + im, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], \
                               points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], \
                           points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6)

            # Uncomment for debugging: display inliers and outliers among
            # matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 ,
            # points2, inliers)

            Hs.append(H12)

        accumulated_homographies = accumulate_homographies(Hs,
                                                           (len(Hs) - 1) // 2)
        Hs = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(
            Hs, minimum_right_translation=5)
        self.Hs = Hs[self.frames_for_panoramas]

    def _load_images(self):
        """
        Load images such that they all rotated and y-axis-aligned related to
            the middle image
        """
        images = []
        for i, num in enumerate(list(self.frames_for_panoramas)):
            img = plt.imread(self.dir_path + self.images_paths[num])
            rot_y_H = np.copy(self.Hs[i])
            rot_y_H[0, -1] = 0
            images.append(warp(img, rot_y_H, output_shape=img.shape))
        self.images = images


class LightFileViewPoint(LightField):
    def __init__(self, dir_path):
        super().__init__(dir_path)

    def calculate_angular_panorama(self, frac):
        # A. Compute panorama canvas size
        # Compute bounding boxes of all warped input images
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.Hs[i], self.w, self.h)
        # Change our reference coordinate system to the panoramas.
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset
        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(
            np.int) + 1

        # B. Compute boundaries between slices
        # center of a frame
        slice_center_2d = np.array([int(self.w * frac), self.h // 2])[None, :]
        # center of all frames (related to middle frame)
        warped_centers = [apply_homography(slice_center_2d, h) for h in
                          self.Hs]
        # get x coordinate of each slice center
        warped_slice_centers = np.array(warped_centers)[:, :, 0].squeeze() - \
                               global_offset[0]
        # boundary between input images in the panorama (and ends of panorama)
        x_strip_boundary = ((warped_slice_centers[
                             :-1] + warped_slice_centers[1:]) / 2)
        x_strip_boundary = np.hstack([0, x_strip_boundary, panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        # C. Put relevant parts from images in slices
        panorama = np.zeros((panorama_size[1], panorama_size[0], 3), dtype=np.uint8)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = plt.imread(self.dir_path + f"{self.images_paths[i]}")
            warped_image = warp_image(image, self.Hs[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            # take strip of warped image and paste to current panorama
            boundaries = x_strip_boundary[i:i + 2]
            image_strip = warped_image[:,
                          boundaries[0] - x_offset: boundaries[1] - x_offset]
            x_end = boundaries[0] + image_strip.shape[1]
            panorama[y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # D. Crop panorama, to get clean rectangle output
        l = int(np.amax(self.bounding_boxes[:, 0, 0]))
        u = int(np.amax(self.bounding_boxes[:, 0, 1]))
        r = int(np.amin(self.bounding_boxes[:, 1, 0]))
        d = int(np.amin(self.bounding_boxes[:, 1, 1]))
        self.panorama = panorama[u:d, l:r, :]

        return self.panorama

    def crop_panorama(self, frame_idx, col_idx):
        pass


class LightFieldRefocus(LightField):
    def __init__(self, dir_path):
        super().__init__(dir_path)

    def refocus(self, shift_size, remove_occ):
        shifted_images = np.zeros((len(self.frames_for_panoramas), self.h, self.w, 3))
        for i in range(len(self.frames_for_panoramas)):
            x_shift = self.Hs[i, 0, -1] * shift_size
            shifted_images[i, :, :, :] = shift(self.images[i], [0, x_shift, 0])
        if remove_occ:
            return np.median(shifted_images, axis=0)
        return np.mean(shifted_images, axis=0)
