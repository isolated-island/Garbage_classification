from tensorflow import keras
import tensorflow as tf
import config
import model_config
MODEL_DIR = config.MODEL_DIR
MODEL_NAME = config.MODEL_NAME
INPUT_SHAPE = model_config.INPUT_SHAPE
# 加载保存的模型
model = keras.models.load_model(MODEL_DIR + "/" + MODEL_NAME)


def test(img_list):
    images = preprocess(img_list)
    res = []
    for img in images:
        pred = model.predict(tf.convert_to_tensor([img]))
        res.append(str(tf.argmax(pred[0]).numpy()))
    return res


def preprocess(img_list):
    images = []
    for name in img_list:
        x = tf.io.read_file("img/" + name)  # 根据路径读取图片
        x = tf.image.decode_jpeg(x, channels=3)  # 图片解码
        x = tf.image.resize(x, INPUT_SHAPE[:2])  # 图片缩放

        # 数据增强
        # x = tf.image.random_flip_up_down(x)
        # x= tf.image.random_flip_left_right(x) # 左右镜像
        # x = tf.image.random_crop(x, INPUT_SHAPE) # 随机裁剪

        # 转换成张量
        # x: [0,255]=> 0~1 归一化
        x = tf.cast(x, dtype=tf.float32) / 255.
        images.append(x)
    return images
