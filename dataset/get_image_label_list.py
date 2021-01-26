import os
import random

import config
FILE_PATH = config.DATASET_PATH
CLASSIFY = config.CLASSIFY


def get():
    # 读取图片路径
    images = []
    labels = []
    for index, classify_name in enumerate(CLASSIFY):
        path = [FILE_PATH + classify_name + "\\" + i for i in os.listdir(FILE_PATH + classify_name)]
        images += path
        labels += [index] * len(path)

    # 打乱图片顺序
    c = list(zip(images, labels))
    random.shuffle(c)
    images[:], labels[:] = zip(*c)
    return images, labels


if __name__ == '__main__':
    image, label = get()
    print(image, label)
