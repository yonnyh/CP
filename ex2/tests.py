from ex2.light_field import *


STANFORD = "data/Pebbles-Stanford-1/"
BANANA = "data/Banana/"
CHESS_S = "data/chess-small/"
NAHLAOT = "data/nahlaot/"


def test_calculate_view_point_by_frames():
    lf = LightFileViewPoint(BANANA)
    # out = lf.calculate_view_point_by_frames(0, 0, 21, 767)
    # out = lf.calculate_view_point_by_frames(0, 300, 21, 300)
    out = lf.calculate_view_point_by_frames(5, 0, 10, 767)

    # lf = LightFileViewPoint(NAHLAOT)
    # out = lf.calculate_view_point_by_frames(0, 0, 60, 729)

    plt.imshow(out)
    plt.show()


def test_calculate_view_point_by_angle():
    lf = LightFileViewPoint(BANANA)
    out = lf.calculate_view_point_by_angle(0, 0, 179)
    plt.imshow(out)
    plt.show()


def test_shift():
    img = plt.imread(BANANA + "rsz_capture_00004.jpg")[200:300, 200:300, :]
    simg = shift(img, [0, 15, 0], mode='constant')
    conc = np.hstack((img, simg))
    plt.imshow(conc)
    plt.show()


def test_refocus(remove_occ=False):
    # lf = LightFieldRefocus(CHESS_S)
    lf = LightFieldRefocus(BANANA)
    out = lf.refocus(3, remove_occ)
    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    # test_calculate_view_point_by_frames()
    test_calculate_view_point_by_angle()
    # test_shift()
    # test_refocus(remove_occ=False)
    # test_refocus(remove_occ=True)
