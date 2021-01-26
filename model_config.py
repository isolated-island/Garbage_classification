import tensorflow as tf

# 输入维度
INPUT_SHAPE = [48, 64, 3]
# 优化器
optimizer = "adam"
# 损失
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 记录值
metrics = ['accuracy']

epochs = 50
batch_size = 32

