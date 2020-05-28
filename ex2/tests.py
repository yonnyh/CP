from ex2.light_field import *


STANFORD = "data/Pebbles-Stanford-1/"
BANANA = "data/Banana/"
CHESS_S = "data/chess-small/"
NAHLAOT = "data/nahlaot/"
SNOW = "data/train-in-snow/"


def test_calculate_view_point_by_frames(apply_hs=True):
    lf = LightFileViewPoint(BANANA, apply_homographies=apply_hs)
    out = lf.calculate_view_point_by_frames(0, 0, 21, 767)
    # out = lf.calculate_view_point_by_frames(0, 300, 21, 300)
    # out = lf.calculate_view_point_by_frames(5, 0, 10, 767)

    # lf = LightFileViewPoint(NAHLAOT)
    # out = lf.calculate_view_point_by_frames(0, 0, 60, 729)

    plt.imshow(out)
    plt.show()


def test_calculate_view_point_by_frames_animate(apply_hs=True):
    def init():
        plot.set_data(first_img)

    def update(i):
        new_im = lf.calculate_view_point_by_frames(0, i * gap_for_animation,
                                                   lf.num_of_frames-1,
                                                   i * gap_for_animation)
        plot.set_data(new_im)

    lf = LightFileViewPoint(BANANA, apply_homographies=apply_hs)
    first_img = lf.calculate_view_point_by_frames(0, 0, lf.num_of_frames-1, 0)
    gap_for_animation = 6

    fig = plt.figure()
    plot = plt.imshow(first_img)

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=lf.num_of_frames, interval=50)

    # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    # FFwriter = FFMpegWriter()
    # anim.save('banana_animation.mp4', writer=FFwriter, fps=30)

    # anim.save('banana_animation.avi')

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
    # test_calculate_view_point_by_frames(apply_hs=True)
    test_calculate_view_point_by_frames(apply_hs=False)
    # test_calculate_view_point_by_frames_animate(apply_hs=True)
    # test_calculate_view_point_by_frames_animate(apply_hs=False)
    # test_calculate_view_point_by_angle()
    # test_shift()
    # test_refocus(remove_occ=False)
    # test_refocus(remove_occ=True)

