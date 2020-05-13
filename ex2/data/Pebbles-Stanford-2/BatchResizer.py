import cv2
import sys
import glob
sys.path.append('/usr/local/lib/python2.7/site-packages')
dir = sys.argv[1];
images = glob.glob(dir + '/*.JPG');

for fname in images:
    print fname
    img = cv2.imread(fname)
    newImg = cv2.resize(img, (512,341))
    cv2.imwrite(fname,newImg)
