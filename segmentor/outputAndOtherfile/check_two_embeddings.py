import segment_anything
import torch
from mmengine.config import Config
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

device = torch.device('cuda')
cfg = Config.fromfile(f'/data/hotaru/my_projects/PNS_tmp/segmentor/config/pannuke321_b.py')
transform_configs = cfg.data.get('train').transform
model = getattr(segment_anything, f"build_sam_vit_{cfg.segmentor.type[-1].lower()}")(cfg)
model.to(device)

# print(model)
file_name = '1_1'
image_path = "./datasets/pannuke/Images/"+file_name+".png"


image = Image.open(image_path)

# 将 PIL 图像对象转换为 numpy 数组
image_np = np.array(image)

transform_configs = cfg.data.get('train').transform
transform_operations = [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in transform_configs]
transform_operations.append(ToTensorV2())  # 将图像转换为张量

# 创建转换对象并应用于图像
transform = A.Compose(transform_operations, p=1)
transformed_data = transform(image=image_np)

# 获取转换后的图像
transformed_image = transformed_data['image']

# 将图像转换为张量并发送到设备上
transformed_image = transformed_image.unsqueeze(0).to(device)  # 添加批次维度并发送到设备上

# 模型推理
output = model(transformed_image)

print(output)