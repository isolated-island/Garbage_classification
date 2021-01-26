import tensorflow as tf
import matplotlib.pyplot as plt
import config
import model_config

FILE_PATH = config.DATASET_PATH
CLASSIFY = config.CLASSIFY
RATE = config.RATE
INPUT_SHAPE = model_config.INPUT_SHAPE


def load_garbage(mode, images, labels):
    # mode: 数据集的模式， train，test
    if mode == "train":
        images = images[:int(len(images)*RATE)]
        labels = labels[:int(len(labels)*RATE)]
        # images: string path
        # labels: number
        db = tf.data.Dataset.from_tensor_slices((images, labels))
        db = db.map(preprocess).batch(len(images))
        return db
        
    elif mode == "test":
        images = images[int(len(images)*RATE):]
        labels = labels[int(len(labels)*RATE):]
        # images: string path
        # labels: number
        db = tf.data.Dataset.from_tensor_slices((images, labels))
        db = db.map(preprocess).batch(len(images))
        return db

    else:
        assert mode == "train" or mode == "test"


def preprocess(x, y):
    # x: 图片的路径List，y：图片的数字编码List
    x = tf.io.read_file(x)  # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3)  # 图片解码
    x = tf.image.resize(x, INPUT_SHAPE[:2])  # 图片缩放

    # 数据增强
    # x = tf.image.random_flip_up_down(x)
    # x= tf.image.random_flip_left_right(x) # 左右镜像
    # x = tf.image.random_crop(x, INPUT_SHAPE) # 随机裁剪

    # 转换成张量
    # x: [0,255]=> 0~1 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y)  # 转换成张量

    return x, y


if __name__ == "__main__":
    # import get_image_label_list
    # images, labels = get_image_label_list.get()
    # train_ds = load_garbage("train", images, labels)
    # for x, y in train_ds:
    #     plt.imshow(x[0])
    #     plt.show()
    pass

