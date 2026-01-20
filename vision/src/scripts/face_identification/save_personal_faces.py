import os
import cv2

def save_faces(name, frames):
    """
    Save the faces in the given `frames` to the dataset folder.

    将给定`frames`中的人脸保存到数据集文件夹中。

    Args:
        name: The name of the person.
        frames: a list of frames containing the face of the person.

        name: 人的名字。
        frames: 包含人脸的帧的列表。
    """
    # 获取当前脚本文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 如果 `dataset/<name>/` 不存在，则创建它
    # 如果存在，则清空文件夹
    if not os.path.exists(current_dir + f"/dataset/{name}/"):
        os.makedirs(current_dir + f"/dataset/{name}/")
    else:
        for file in os.listdir(current_dir + f"/dataset/{name}/"):
            os.remove(current_dir + f"/dataset/{name}/" + file)

    # 保存人脸图像
    for i, frame in enumerate(frames):
        cv2.imwrite(current_dir + f"/dataset/{name}/{str.zfill(str(i), 3)}.jpg", frame)