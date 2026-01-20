#此脚本用于预处理面部识别数据，调用其他脚本进行嵌入提取和模型训练
import os

# 获取当前脚本文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# os.system(f"python {current_dir}/scripts/extract_embeddings.py")
# os.system(f"python {current_dir}/scripts/train_model.py")

# 判断当前系统为Windows系统还是Linux系统
if os.name == 'nt':
    # Windows系统
    # 使用`python`命令运行指定的Python脚本
    os.system(f"python {current_dir}/scripts/extract_embeddings.py")
    os.system(f"python {current_dir}/scripts/train_model.py")
else:
    # Linux系统
    # 使用`python3`命令运行指定的Python脚本
    os.system(f"python3 {current_dir}/scripts/extract_embeddings.py")
    os.system(f"python3 {current_dir}/scripts/train_model.py")
