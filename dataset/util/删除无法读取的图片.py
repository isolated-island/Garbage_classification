import os
import matplotlib.image as mp
FILE_PATH = r"C:\Users\crazypig\Desktop\ZSTU_CrazyPig_Study\python\Garbage_classification\dataset\Data\\" # 数据集根路径
CLASSIFY = ["false" , "true"]   # 类别
images = []
labels = []
for index, classify_name in enumerate(CLASSIFY):
    path = [FILE_PATH + classify_name + "\\" + i for i in os.listdir(FILE_PATH + classify_name)]
    images += path
    labels += [index] * len(path)

for index, image_path in enumerate(images):
    try:
        image = mp.imread(image_path)
        mp.imsave(image_path, image)
    except:
        print(index, image_path)
        os.remove(image_path)
