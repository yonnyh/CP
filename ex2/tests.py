from ex2.light_field import *


STANFORD = "data/Pebbles-Stanford-1"


def test_ordered_images_names():
    got = ordered_images_names(STANFORD)
    excepted = ['IMG_9246.JPG', 'IMG_9247.JPG', 'IMG_9248.JPG', 'IMG_9249.JPG',
                'IMG_9250.JPG', 'IMG_9251.JPG', 'IMG_9252.JPG', 'IMG_9253.JPG',
                'IMG_9254.JPG', 'IMG_9255.JPG', 'IMG_9256.JPG', 'IMG_9257.JPG',
                'IMG_9258.JPG', 'IMG_9259.JPG', 'IMG_9260.JPG', 'IMG_9261.JPG',
                'IMG_9262.JPG']
    for i in range(len(got)):
        if got[i] != excepted[i]:
            print("False")
            break
    else:
        print("True")


if __name__ == '__main__':
    test_ordered_images_names()
