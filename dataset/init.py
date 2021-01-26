import os
import config

CLASSIFY = config.CLASSIFY
CLASS_NUMBER_FILE = config.CLASS_NUMBER_FILE
PIC_NUMBEI_FILE = config.PIC_NUMBEI_FILE
FILE_PATH = config.DATASET_PATH


def record():
    # 为类别和图片打标签
    # 首先为每个类别编写一个数字
    with open(CLASS_NUMBER_FILE, "w") as f:
        for index, classify_name in enumerate(CLASSIFY):
            f.write(classify_name + " " + str(index) + "\n")

    # 记录每张图片的类别
    with open(PIC_NUMBEI_FILE, "w") as f:
        for index, classify_name in enumerate(CLASSIFY):
            for name in os.listdir(FILE_PATH + classify_name):
                f.write(name + " " + str(index) + "\n")


if __name__ == '__main__':
    record()
