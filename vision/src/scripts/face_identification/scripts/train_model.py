# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
# 此脚本用于训练面部识别模型
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

import os

# 获取当前脚本文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

#加载面部嵌入数据
print("[INFO] loading face embeddings...")
data = pickle.loads(open(os.path.join(current_dir, "..", "output", "embeddings.pickle"), "rb").read())
#这里加载的是之前脚本生成的面部嵌入数据文件embeddings.pickle

#编码标签
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])#将面部名称转换为数值标签

#接下来是训练模型
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="rbf", probability=True)#c越大，误差越小，容易过拟合；c越小，误差越大，容易欠拟合
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(os.path.join(current_dir, "..", "output", "recognizer"), "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(os.path.join(current_dir, "..", "output", "le.pickle"), "wb")
f.write(pickle.dumps(le))
f.close()