import config
from PIL import Image
import model_config
import numpy as np
from dataset import get_image_label_list
image_name, image_label = get_image_label_list.get()

RATE = config.RATE
FILE_PATH = config.DATASET_PATH
LENGTH = len(image_name)
train_image_name, train_image_label = image_name[:int(LENGTH*RATE)], image_label[:int(LENGTH*RATE)]
test_image_name, test_image_label = image_name[int(LENGTH*RATE):], image_label[int(LENGTH*RATE):]


def train_reader():
    def reader():
        for name, label in zip(train_image_name, train_image_label):
            img = Image.open(name)
            img = img.resize((model_config.INPUT_SHAPE[1], model_config.INPUT_SHAPE[0]), Image.ANTIALIAS)
            img = np.array(img) / 255.
            img = img.transpose((2, 0, 1))
            yield img, int(label)
    return reader


def test_reader():
    def reader():
        for name, label in zip(test_image_name, test_image_label):
            img = Image.open(name)
            img = img.resize((model_config.INPUT_SHAPE[1], model_config.INPUT_SHAPE[0]), Image.ANTIALIAS)
            img = np.array(img) / 255.
            img = img.transpose((2, 0, 1))
            yield img, int(label)
    return reader


if __name__ == '__main__':
    pass
