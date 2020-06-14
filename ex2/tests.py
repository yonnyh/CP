from ex2.light_field import *


STANFORD = "data/Pebbles-Stanford-1/"
BANANA = "data/Banana/"
BANANA_OP = "data/Banana-Op/"
CHESS_S = "data/chess-small/"
NAHLAOT = "data/nahlaot/"
SNOW = "data/train-in-snow/"
JELLY = "data/pp/"
LEGO = "data/blue-lego-small/"
APPLES = "data/apples/"
OUR = "data/our_images/"


def det_images_from_path(dir_path):
    f = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        f.extend(filenames)
        break
    im_pathes = sorted(f)

    imgs = []
    for i in range(len(im_pathes)):
        imgs.append(plt.imread(dir_path + im_pathes[i]).astype(np.float) / 255)
    return imgs


def test_frames_to_angle():
    lf = LightFileViewPoint(det_images_from_path(BANANA))
    print(lf.frames_to_angle(0, 0, 10, 0))  # ~175
    print(lf.frames_to_angle(0, 200, 10, 0))  # ~5


def test_calculate_view_point_by_frames(scene=BANANA, fast=True, apply_hs=False,
                                        debug=True, by_angle=False):
    lf = LightFileViewPoint(det_images_from_path(scene))
    # lf.calc_homographies()
    if apply_hs:
        lf.apply_homographies_on_images()
    # out = lf.calculate_view_point_by_frames(0, 0, 21, 767, fast=fast)
    # out = lf.calculate_view_point_by_frames(0, 300, 21, 300)
    # out = lf.calculate_view_point_by_frames(5, 0, 10, 767)
    # out = lf.calculate_view_point_by_frames(0, 700, 20, 0)

    start_f, start_c, end_f, end_c = 0, 0, lf.num_of_frames - 1, lf.w -1
    out = lf.calculate_view_point_by_frames(start_f, end_c, end_f, end_c,
                                            by_angle=by_angle, debug=debug, fast=fast)

    # by_frame = lf.calculate_view_point_by_frames(0, 700, 20, 0, by_angle=False)
    # by_angle = lf.calculate_view_point_by_frames(0, 700, 20, 0, by_angle=True)
    # out = np.hstack((by_frame, by_angle))

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

    lf = LightFileViewPoint(BANANA)
    first_img = lf.calculate_view_point_by_frames(0, 0, lf.num_of_frames-1, 0)
    gap_for_animation = 6

    fig = plt.figure()
    plot = plt.imshow(first_img)

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=lf.num_of_frames, interval=50)

    # If problem in saving: https://www.wikihow.com/Install-FFmpeg-on-Windows
    plt.rcParams['animation.ffmpeg_path'] = "C:\\FFmpeg\\bin\\ffmpeg.exe"
    FFwriter = FFMpegWriter()
    anim.save('banana_animation.mp4', writer=FFwriter)

    plt.show()


def test_calculate_view_point_by_angle(scene=BANANA):
    lf = LightFileViewPoint(det_images_from_path(scene))
    out = lf.calculate_view_point_by_angle(0, 0, 179)
    plt.imshow(out)
    plt.show()


def save_veiwpoint(scene, name):
    lf = LightFileViewPoint(det_images_from_path(scene))
    lf.apply_homographies_on_images()
    start_f = 0
    end_f = lf.num_of_frames - 1
    trick_s, trick_e = 4, 8
    start_c = 0
    end_c = lf.w - 1
    for sf, sc, ef, ec in [(start_f, start_c + trick_s, end_f, end_c - trick_e),
                           (start_f, end_c - trick_e, end_f, start_c + trick_s),
                           (start_f, start_c + trick_s, end_f, start_c + trick_s),
                           (start_f, end_c - trick_e, end_f, end_c - trick_e)]:
        for fast in [True, False]:
            out = lf.calculate_view_point_by_frames(sf, sc, ef, ec, fast=fast,
                                                    debug=False)
            text_s = start_c if sc == start_c + trick_s else end_c
            text_e = start_c if ec == start_c + trick_s else end_c
            plt.title(f"start frame: {sf}, start col: {text_s}\n"
                      f"end frame: {ef}, end col: {text_e}")
            plt.imshow(out)
            sig = "fast" if fast else "full"
            plt.savefig(f"outputs_viewpoint/{name}/{name}_{sig}_"
                        f"{sf, text_s, ef, text_e}.png")
            # plt.show()


def test_shift():
    img = plt.imread(BANANA + "rsz_capture_00004.jpg")[200:300, 200:300, :]
    simg = shift(img, [0, 15, 0], mode='constant')
    conc = np.hstack((img, simg))
    plt.imshow(conc)
    plt.show()


def test_refocus_by_shift(remove_occ=False):
    lf = LightFieldRefocus(det_images_from_path(CHESS_S))
    # lf = LightFieldRefocus(det_images_from_path(BANANA))
    out = lf.refocus_by_shift(3, remove_occ)
    plt.imshow(out)
    plt.show()


def save_refocus(scene, name):
    lf = LightFieldRefocus(det_images_from_path(scene))
    for f in np.linspace(0, 1, 7):
        out = lf.refocus_by_shift(f)
        plt.title(f"focus: {f}")
        plt.imshow(out)
        plt.savefig(f"outputs_refocus/{name}_{f}.png")
        plt.show()


def test_refocus_by_object(remove_occ=False, debug=True):
    def display_borders(img, u, l, d, r):
        rect = patches.Rectangle((l, u), r - l, d - u, linewidth=2,
                                 edgecolor='r', facecolor='none')
        plt.imshow(img)
        plt.gca().add_patch(rect)
        plt.show()

    lf = LightFieldRefocus(det_images_from_path(CHESS_S))
    # lf = LightFieldRefocus(det_images_from_path(BANANA))

    # up, left, down, right = 110, 900, 150, 960
    up, left, down, right = 350, 950, 430, 1030
    out = lf.refocus_by_object(up, left, down, right, remove_occ, debug=debug)
    display_borders(out, up, left, down, right)
    # plt.imshow(out)
    # plt.show()


if __name__ == '__main__':
    # test_frames_to_angle()
    # test_calculate_view_point_by_frames(apply_hs=True)
    # test_calculate_view_point_by_frames(scene=SNOW, apply_hs=False,
    #                                     debug=True, fast=True, by_angle=False)
    # test_calculate_view_point_by_angle()
    # save_veiwpoint(APPLES, "apples_h")
    # test_shift()
    # test_refocus_by_shift(remove_occ=True)
    # test_refocus_by_shift(remove_occ=False)
    save_refocus(scene=OUR, name="our")
    # test_refocus_by_object(remove_occ=False, debug=True)

