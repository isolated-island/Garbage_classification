from tensorflow.keras import models, layers

import model_config
INPUT_SHAPE = model_config.INPUT_SHAPE

model = models.Sequential([
    layers.Conv2D(filters=6, kernel_size=5,
                  activation='sigmoid', input_shape=INPUT_SHAPE),
    layers.MaxPool2D(pool_size=2, strides=2),
    layers.Conv2D(filters=18, kernel_size=5, activation='sigmoid'),
    layers.MaxPool2D(pool_size=2, strides=2),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(2, activation='sigmoid')
])

if __name__ == '__main__':
    print(INPUT_SHAPE)
