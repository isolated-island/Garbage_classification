import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import model_config
from dataset import garbage
from dataset import get_image_label_list

INPUT_SHAPE = model_config.INPUT_SHAPE
optimizer = model_config.optimizer
loss = model_config.loss
metrics = model_config.metrics
epoch = model_config.epochs
batch_size = model_config.batch_size

MODEL_DIR = config.MODEL_DIR
MODEL_NAME = config.MODEL_NAME
model = config.model

# 加载训练集和测试集
images, labels = get_image_label_list.get()
train_ds = garbage.load_garbage("train", images, labels)
train_ds = iter(train_ds)
train_images, train_labels = next(train_ds)

test_ds = garbage.load_garbage("test", images, labels)
test_ds = iter(test_ds)
test_images, test_labels = next(test_ds)

# 如果模型不存在就训练模型
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.exists(MODEL_DIR + "/" + MODEL_NAME):
    model.summary()
    # 配置优化方法
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    # 训练模型
    history = model.fit(train_images, train_labels, epochs=epoch,
                        validation_data=(test_images, test_labels),
                        batch_size=batch_size)
    # 保存模型
    model.save(MODEL_DIR + "/" + MODEL_NAME)
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

else:
    # 加载保存的模型
    model = keras.models.load_model(MODEL_DIR + "/" + MODEL_NAME)
    # 从测试集中随机抽取10张图片，打印测试结果
    for i in range(10):
        k = np.random.randint(low=0, high=len(test_images))
        pred = model.predict(np.array([test_images[k]]))
        print("预测结果为", tf.argmax(pred[0]).numpy(), "，标签是", test_labels[k].numpy())
