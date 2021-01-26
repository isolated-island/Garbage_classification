import os
from net_struct import LeNet_5
BASE_DIRS = os.path.dirname(__file__)

# 数据集文件位置
DATASET_PATH = BASE_DIRS + r"\dataset\Data\\"

# 类别
CLASSIFY = ["false", "true"]

# 类别和类别编号对应表
CLASS_NUMBER_FILE = BASE_DIRS + r"\dataset\file\classify_number.txt"
# 图片和类别编号对应表
PIC_NUMBEI_FILE = BASE_DIRS + r"\dataset\file\picture_number.txt"

# 训练集占比
RATE = 0.8

# 选择模型
model = LeNet_5.model

# 模型保存位置
MODEL_DIR = BASE_DIRS + "/model"
MODEL_NAME = "LetNet_5.h5"
