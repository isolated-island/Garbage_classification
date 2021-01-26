import os
PATH_1 = r"C:\Users\crazypig\Desktop\unqualified"
PATH_2 = r"C:\Users\crazypig\Desktop\qualified"
PATH_3 = r"C:\Users\crazypig\Desktop\ZSTU_CrazyPig_Study\python\Garbage_classification\dataset\Data\false"
PATH_4 = r"C:\Users\crazypig\Desktop\ZSTU_CrazyPig_Study\python\Garbage_classification\dataset\Data\true"
path_1_list =  os.listdir(PATH_2)
for img in os.listdir(PATH_4):
    if img in path_1_list:
        os.remove(PATH_4 + "\\" + img)
print(len(os.listdir(PATH_4)))