import pydicom
import cv2
import glob
import os
from dataLoader import dicom2array
import multiprocessing


def dicom2jpeg(path):
    base_path, _ = os.path.splitext(path)
    out_path = base_path + '.png'
    if not os.path.exists(out_path):
        cv2.imwrite(base_path + '.png', dicom2array(path))


if __name__ == '__main__':
    base = '/home/philipp/Schreibtisch/fp/siim-covid19-detection/train'
    pool = multiprocessing.Pool(10)
    pool.map(dicom2jpeg, glob.glob(base + '/**/*.dcm', recursive=True))
