import numpy as np
import matplotlib.pyplot as plt
from os import walk


def ordered_images_names(dir_path):
    f = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        f.extend(filenames)
        break
    return sorted(f)


if __name__ == '__main__':
    pass
