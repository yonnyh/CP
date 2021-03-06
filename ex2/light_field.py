import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from ex2.impro_4 import *
from os import walk
from scipy.ndimage.interpolation import shift
from skimage.transform import warp
from skimage.draw import line_aa, line
from skimage.filters import sobel
from cv2 import warpAffine, BORDER_TRANSPARENT


class LightField:
    def __init__(self, images):
        self.images = images
        for i, im in enumerate(self.images):
            if i == 0:
                self.h, self.w = im.shape[:2]
            else:
                if self.h != im.shape[0] or self.w != im.shape[1]:
                    raise ValueError("Images Have No Same Shape")
        self.num_of_frames = len(self.images)

        self.default_shift = 1  # in case user doesn't calc homographies
        self.relative_shifts = [self.default_shift *
                                i for i in range(self.num_of_frames)]
        self.left_to_right = True
        self.Hs = None

    def calc_homographies(self, const_move=False):
        points_and_descriptors = []
        for i, im in enumerate(self.images):
            gray_im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])
            pyramid, _ = sol4_utils.build_gaussian_pyramid(gray_im, 3, 7)
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
            Hs.append(H12)

        self.Hs = np.stack(accumulate_homographies(Hs, (len(Hs) - 1) // 2))
        self.relative_shifts = [int(w) - int(self.Hs[0, 0, -1]) for w in
                                self.Hs[:, 0, -1]]

        if self.relative_shifts[-1] < 0:  # frames taken from right to left
            self.left_to_right = False
            for i in range(len(self.relative_shifts)):
                self.relative_shifts[i] -= self.relative_shifts[-1]

        if const_move:
            start, end = self.relative_shifts[0], self.relative_shifts[-1]
            self.relative_shifts = np.linspace(start, end, len(
                self.relative_shifts)).astype(np.int)

    def apply_homographies_on_images(self):
        """
        Load images such that they all rotated and y-axis-aligned related to
            the middle image
        """
        if self.Hs is None:
            self.calc_homographies()

        for i in range(self.num_of_frames):
            rot_y_H = np.copy(self.Hs[i])
            rot_y_H[0, -1] = 0
            self.images[i] = (warp(self.images[i], rot_y_H,
                                   output_shape=self.images[i].shape))


class LightFileViewPoint(LightField):
    def __init__(self, images):
        super().__init__(images)
        self._save_shifted()

    def _save_shifted(self):
        if self.Hs is None:
            self.calc_homographies()

        canvas_w = self.w + max(self.relative_shifts)
        self.shifted = np.zeros((self.num_of_frames, self.h, canvas_w, 3))
        for i in range(self.num_of_frames):
            start = int(self.relative_shifts[i])
            end = start + self.w
            self.shifted[i, :, start:end, :] = self.images[i]

    def _display_debug(self, mask):
        d_mask = np.copy(mask)
        for i in range(self.shifted.shape[0]):  # range of frames
            d_mask[i, np.where(self.shifted[i, 0, :, :] == 0)[0]] = 0.2
            d_mask[i, np.where(mask[i, :] > 0)[0]] = mask[
                i, np.where(mask[i, :] > 0)[0]]
        plt.imshow(d_mask)
        plt.show()

    def view_point_by_mask(self, mask, fast=True, opp_order=False):
        _, h, w, c = self.shifted.shape
        if fast:
            slices = []
            for i in range(self.num_of_frames):
                slices.append(self.shifted[i, :, mask[i] == 1, :])
            if opp_order:
                slices.reverse()
            return np.transpose(np.vstack(slices), (1, 0, 2))

        else:
            canvas = np.zeros((h, w, c))
            for i in range(self.num_of_frames):
                canvas = canvas + self.shifted[i, :, :, :] * mask[i, :][:, None]

            # Crop canvas to remove zeros-pad
            non_zeros_col_ides = np.where(np.sum(canvas, axis=(0, 2)) > 0)[0]
            return canvas[:, non_zeros_col_ides, :]

    def calculate_view_point_by_frames(self, frame1, col1, frame2, col2,
                                       debug=False, by_angle=True, fast=True):
        # Validity check
        for f in [frame1, frame2]:
            if f >= self.num_of_frames:
                raise IndexError(f"There are only {self.num_of_frames} frames,"
                                 f" and {f} is out of range")
        for col in [col1, col2]:
            if col >= self.w:
                raise IndexError(f"Images width is {self.w}, and {col} is "
                                 f"out of range")

        if by_angle:
            angle = self.frames_to_angle(frame1, col1, frame2, col2)
            return self.calculate_view_point_by_angle(frame1, col1, angle,
                                                      debug=debug, fast=fast)

        else:
            frames, h, w, c = self.shifted.shape
            mask = np.zeros((frames, w))

            if not fast:
                rr, cc, val = line_aa(frame1, self.relative_shifts[frame1] + col1,
                                      frame2, self.relative_shifts[frame2] + col2)
                mask[rr, cc] = val
            else:
                rr, cc = line(frame1, self.relative_shifts[frame1] + col1,
                              frame2, self.relative_shifts[frame2] + col2)
                mask[rr, cc] = 1

            if debug:
                self._display_debug(mask)

            opp_order = (cc[0] > cc[-1])

            return self.view_point_by_mask(mask, fast=fast, opp_order=opp_order)

    def frames_to_angle(self, frame1, col1, frame2, col2):
        """Return angle in range [0, 180]"""
        x = frame2 - frame1
        y = self.relative_shifts[frame2] + col2 - \
            (self.relative_shifts[frame1] + col1)
        return np.rad2deg(np.arctan2(y, x)) + 90

    def calculate_view_point_by_angle(self, frame, col, angle_deg, debug=False,
                                      fast=True):
        """
        angle_deg in range [0, 180]
        """
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

        if not fast:
            rr, cc, val = line_aa(frame, col, self.num_of_frames - 1, int(col2))
            valid = np.where((cc >= 0) & (cc < self.shifted.shape[2]))[0]
            rr_valid, cc_valid, val_valid = rr[valid], cc[valid], val[valid]
            mask[rr_valid, cc_valid] = val_valid
        else:
            rr, cc = line(frame, col, self.num_of_frames - 1, int(col2))
            valid = np.where((cc >= 0) & (cc < self.shifted.shape[2]))[0]
            rr_valid, cc_valid = rr[valid], cc[valid]
            mask[rr_valid, cc_valid] = 1

        if debug:
            self._display_debug(mask)

        opp_order = (angle_deg < 90)

        return self.view_point_by_mask(mask, fast=fast, opp_order=opp_order)


class LightFieldRefocus(LightField):
    def __init__(self, dir_path):
        super().__init__(dir_path)

    def refocus_by_shift(self, shift_size, remove_occ=False):
        """Shift frames related to center-frame"""
        centered_shifts = np.array(self.relative_shifts) - \
                          self.relative_shifts[(self.num_of_frames - 1) // 2]
        shifted_images = np.zeros((self.num_of_frames, self.h, self.w, 3))
        for i in range(self.num_of_frames):
            M = np.float32([[1, 0, shift_size * centered_shifts[i]],
                            [0, 1, 0]])
            shifted_images[i] = warpAffine(self.images[i], M, (self.w, self.h),
                                           borderMode=BORDER_TRANSPARENT)
        if remove_occ:
            return np.median(shifted_images, axis=0) * 255
        return np.mean(shifted_images, axis=0) * 255

    def refocus_by_object(self, up, left, down, right, remove_occ=False,
                          range_to_scan=4, num_of_intervals=17, debug=False):
        """Maximise edges in selected area"""
        def evaluate_edges(img):
            edges_img = sobel(img[up:down, left:right])
            return np.sum(edges_img)

        intervals = np.linspace(-range_to_scan, range_to_scan, num_of_intervals)
        eva_list = np.zeros_like(intervals)
        for i in range(num_of_intervals):
            refocused = self.refocus_by_shift(intervals[i], remove_occ)
            gray_refocused = np.dot(refocused[..., :3], [0.2989, 0.5870, 0.1140])
            eva_list[i] = (evaluate_edges(gray_refocused))

        if debug:
            plt.plot(eva_list)
            plt.show()

        self.best_focus = intervals[np.argmax(eva_list)]

        return self.refocus_by_shift(intervals[np.argmax(eva_list)], remove_occ)
