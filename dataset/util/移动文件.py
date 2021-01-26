import shutil
import os
with open(r"C:\Users\crazypig\Desktop\ZSTU_CrazyPig_Study\python\Garbage_classification\error_img.txt", "r") as f:
    list_error = f.readlines()
    for path in list_error:
        shutil.move(path[:-1], path[:37] + "error\\" + path[38:-1])

