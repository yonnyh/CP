import numpy as np
import matplotlib.pyplot as plt
from ex2.impro_4 import *
from os import walk
from scipy.ndimage.interpolation import shift
from skimage.transform import warp
from skimage.draw import line_aa


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

        self.num_of_frames = len(Hs) + 1
        self.Hs = np.stack(accumulate_homographies(Hs, (len(Hs) - 1) // 2))
        self.relative_shifts = [int(w) - int(self.Hs[0, 0, -1]) for w in
                                self.Hs[:, 0, -1]]

        # accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        # Hs = np.stack(accumulated_homographies)
        # self.frames_for_panoramas = filter_homographies_with_translation(
        #     Hs, minimum_right_translation=5)
        # self.Hs = Hs[self.frames_for_panoramas]

    def _load_images(self):
        """
        Load images such that they all rotated and y-axis-aligned related to
            the middle image
        """
        images = []
        for i in range(self.num_of_frames):
            img = plt.imread(self.dir_path + self.images_paths[i])
            rot_y_H = np.copy(self.Hs[i])
            rot_y_H[0, -1] = 0
            images.append(warp(img, rot_y_H, output_shape=img.shape))
        self.images = images


class LightFileViewPoint(LightField):
    def __init__(self, dir_path):
        super().__init__(dir_path)
        self._save_shifted()

    def _save_shifted(self):
        """Assumes scened was taken from left to right (first x-translation is
        the smallest)"""
        self.shifted = np.zeros((self.num_of_frames, self.h,
                                 self.w + self.relative_shifts[-1], 3))
        for i in range(self.num_of_frames):
            x_shift = int(self.relative_shifts[i])
            self.shifted[i, :, x_shift:self.w + x_shift, :] = self.images[i]

    def view_point_by_mask(self, mask):
        _, h, w, c = self.shifted.shape
        canvas = np.zeros((h, w, c))
        for i in range(self.num_of_frames):
            canvas = canvas + self.shifted[i, :, :, :] * mask[i, :][:, None]

        # Crop canvas to remove zeros-pad
        non_zeros_col_ides = np.where(np.sum(canvas, axis=(0, 2)) > 0)[0]
        return canvas[:, non_zeros_col_ides, :]

    def calculate_view_point_by_frames(self, frame1, col1, frame2, col2,
                                       debug=False):
        # Validity check
        for f in [frame1, frame2]:
            if f >= self.num_of_frames:
                raise IndexError(f"There are only {self.num_of_frames} frames,"
                                 f" and {f} is out of range")
        for col in [col1, col2]:
            if col >= self.w:
                raise IndexError(f"Images width is {self.w}, and {col} is "
                                 f"out of range")

        frames, h, w, c = self.shifted.shape
        mask = np.zeros((frames, w))
        rr, cc, val = line_aa(frame1, self.relative_shifts[frame1] + col1,
                              frame2, self.relative_shifts[frame2] + col2)
        mask[rr, cc] = val

        if debug:
            plt.imshow(mask)
            plt.show()

        return self.view_point_by_mask(mask)

    def calculate_view_point_by_angle(self, frame, col, angle_deg, debug=False):
        # Validity check
        if frame >= self.num_of_frames:
            raise IndexError(f"There are only {self.num_of_frames} frames,"
                             f" and {frame} is out of range")
        if col >= self.w:
            raise IndexError(f"Images width is {self.w}, and {col} is "
                             f"out of range")
        if angle_deg > 180 or angle_deg < 0:
            raise ValueError(f"Angle mast be between 0 to 180")

        frames, h, w, c = self.shifted.shape
        mask = np.zeros((frames, w))

        angle_rad = np.deg2rad(angle_deg - 90)
        col2 = self.relative_shifts[frame] + col + \
               (self.num_of_frames - frame) * np.tan(angle_rad)
        rr, cc, val = line_aa(frame, col, self.num_of_frames - 1, int(col2))
        valid = np.where((cc >= 0) & (cc < self.shifted.shape[2]))[0]
        rr_valid, cc_valid, val_valid = rr[valid], cc[valid], val[valid]

        mask[rr_valid, cc_valid] = val_valid

        if debug:
            plt.imshow(mask)
            plt.show()

        return self.view_point_by_mask(mask)


class LightFieldRefocus(LightField):
    def __init__(self, dir_path):
        super().__init__(dir_path)

    def refocus(self, shift_size, remove_occ):
        shifted_images = np.zeros((self.num_of_frames, self.h, self.w, 3))
        for i in range(self.num_of_frames):
            x_shift = self.Hs[i, 0, -1] * shift_size
            shifted_images[i, :, :, :] = shift(self.images[i], [0, x_shift, 0])
        if remove_occ:
            return np.median(shifted_images, axis=0)
        return np.mean(shifted_images, axis=0)
