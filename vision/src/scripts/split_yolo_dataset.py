#此脚本用于将YOLO数据集拆分为训练集和验证集，比例为80%训练集，20%验证集
from sklearn.model_selection import train_test_split
import glob
import shutil
import os

# 获取当前脚本文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get all paths to your images files and text files
PATH = current_dir + '/dataset/'
img_paths = glob.glob(PATH + 'images/*.jpg')
txt_paths = glob.glob(PATH + 'labels/*.txt')
    
X_train, X_val, y_train, y_val = train_test_split(img_paths, txt_paths, test_size=0.2, random_state=42)

# move the images and text files to the train and test folders

for i in range(len(X_train)):
    shutil.move(X_train[i], PATH + 'images/train/')
    shutil.move(y_train[i], PATH + 'labels/train/')
    
for i in range(len(X_val)):
    shutil.move(X_val[i], PATH + 'images/val/')
    shutil.move(y_val[i], PATH + 'labels/val/')
    