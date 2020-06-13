import cv2
import numpy as np

WINDOW_TITLE = 'Mark focus area (press Enter to complete, or Esc to reset)'
ENTER_KEY = 13
ESC_KEY = 27


def convert_to_uint8(image):
    image = image.astype(np.float)
    min_value = np.min(image)
    image = image - min_value
    value_range = np.max(image)
    normalized_image = image / value_range
    uint8_image = np.round(255 * normalized_image).astype(np.uint8)
    return uint8_image


class AreaMarker:
    def __init__(self, image):
        self.crop_area = []
        self.cropping = False
        self.last_mouse_location = None
        self.clone = None

        self.image = image
        self.full_size_image = self.image.copy()
        self.shrink_ratio = self.resize_image(width=min(self.image.shape[1], 1500))
        self.image = convert_to_uint8(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    def resize_image(self, width=None, height=None):
        """Resize image while maintaining aspect ratio."""
        (h, w) = self.image.shape[:2]

        if width is None:
            r = height / float(h)
            new_dimensions = (int(w * r), height)
        else:
            r = width / float(w)
            new_dimensions = (width, int(h * r))

        self.image = cv2.resize(self.image, new_dimensions, interpolation=cv2.INTER_AREA)
        return w / width

    def convert_coordinates_to_big_image(self):
        self.crop_area = list((np.array(self.crop_area) * self.shrink_ratio).astype(int))

    def click_and_mark(self, event, x, y, flags, param):
        # if left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_area = [(x, y)]
            self.cropping = True
            self.last_mouse_location = (x, y)

        # check if left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record ending (x, y) coordinates and indicate that cropping is finished
            self.crop_area.append((x, y))
            self.cropping = False

        # check if cropping is still in progress in order to show marking region in real-time
        elif self.cropping and event == cv2.EVENT_MOUSEMOVE:
            if (x, y) != self.last_mouse_location:
                self.image = self.clone.copy()
                # start_coordinates = self.crop_area[0]
                # axes_length = int(abs(x - start_coordinates[0]) / 2), int(abs(y - start_coordinates[1]) / 2)
                # center_coordinates = axes_length[0] + min(start_coordinates[0], x), axes_length[1] + min(start_coordinates[1], y)

                # cv2.ellipse(self.image, center_coordinates, axes_length, angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=2)

                cv2.rectangle(self.image, self.crop_area[0], (x, y), (0, 255, 0), 2)
                cv2.imshow(WINDOW_TITLE, self.image)
                self.last_mouse_location = (x, y)

    def get_square_from_user(self):
        self.clone = self.image.copy()
        cv2.namedWindow(WINDOW_TITLE)

        cv2.setMouseCallback(WINDOW_TITLE, self.click_and_mark)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow(WINDOW_TITLE, self.image)
            key = cv2.waitKey(1) & 0xFF

            # if escape or 'r' are pressed, reset cropping region
            if key == ESC_KEY or key == ord("r"):
                self.image = self.clone.copy()

            # if enter or 'c' are pressed and region was marked, break
            elif key == ENTER_KEY or key == ord("c"):
                if len(self.crop_area) == 2:  # if area was marked
                    self.convert_coordinates_to_big_image()
                    start_x, end_x = sorted((self.crop_area[0][0], self.crop_area[1][0]))
                    start_y, end_y = sorted((self.crop_area[0][1], self.crop_area[1][1]))

                    cv2.destroyAllWindows()
                    return start_x, end_x, start_y, end_y
