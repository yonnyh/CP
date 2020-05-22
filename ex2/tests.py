from ex2.light_field import *


STANFORD = "data/Pebbles-Stanford-1/"
BANANA = "data/Banana/"
CHESS_S = "data/chess-small/"


def test_all_images():
    imgs = all_images(STANFORD)
    for im in imgs:
        plt.imshow(im)
        plt.show()


def test_homography():
    img = plt.imread(STANFORD + "IMG_9246.JPG").astype(np.float32)
    # print(homography(img, img))  # needs to be close to np.eye(3)

    H = np.array([[1, 0, 0], [0, 1, 50], [0, 0, 1]]).astype(np.float32)
    im_dst = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    calc_H = homography(im_dst.astype(np.uint8), img.astype(np.uint8), debug=True)
    print(H - calc_H)  # has to be small in all entries


def test_wrap_img(x, y, theta):
    # img = cv2.imread(STANFORD + "IMG_9246.JPG", cv2.IMREAD_GRAYSCALE)
    img = plt.imread(STANFORD + "/IMG_9246.JPG")
    # img = plt.imread(STANFORD + "/IMG_9246.JPG").astype(np.uint8)
    # img = cv2.colorChange(img, cv2.COLOR_RGB2GRAY)
    theta = np.deg2rad(theta)
    H = np.array([[np.cos(theta),   np.sin(theta),  x],
                  [-np.sin(theta),  np.cos(theta),  y],
                  [0,               0,              1]])
    trans = EuclideanTransform(matrix=H)
    new_img = warp(img, trans)
    # new_img = warp_channel(img, H)
    plt.imshow(new_img)
    plt.show()


def test_panoram():
    p = PanoramicVideoGenerator(BANANA, 'rsz_capture_00', 22)
    p.align_images()
    p.generate_panoramic_images(1)


def test_my_panorama():
    lf = LightFileViewPoint(BANANA)
    out = lf.calculate_angular_panorama(0.5)
    plt.imshow(out)
    plt.show()


def test_shift():
    img = plt.imread(BANANA + "rsz_capture_00004.jpg")[200:300, 200:300, :]
    simg = shift(img, [0, 15, 0], mode='constant')
    conc = np.hstack((img, simg))
    plt.imshow(conc)
    plt.show()


def test_refocus():
    lf = LightFieldRefocus(CHESS_S)
    out = lf.refocus(1)
    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    # test_ordered_images_names()
    # test_all_images()
    # test_homography()
    # test_imgs2homographies()
    # test_wrap_img(0, 0, 0)
    # test_wrap_img(10, 10, 40)
    # test_panoram()
    # test_my_panorama()
    # test_shift()
    test_refocus()
