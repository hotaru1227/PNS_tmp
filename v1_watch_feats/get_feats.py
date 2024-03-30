import torch
from mmengine.config import Config
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from image_encoder import ImageEncoderViT
from dpa_p2pnet import DPAP2PNet
from dpa_p2pnet import Backbone

device = torch.device('cuda')
cfg = Config.fromfile(f'/data/hotaru/my_projects/PNS_tmp/segmentor/config/pannuke321_b.py')
transform_configs = cfg.data.get('train').transform


# print(model)
file_name = '1_1'
image_path = "/data/hotaru/my_projects/PNS_tmp/segmentor/datasets/pannuke/Images/"+file_name+".png"


image = Image.open(image_path)
image_np = np.array(image)
image_np = image_np

transform_configs = cfg.data.get('train').transform
transform_operations = [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in transform_configs]
transform_operations.append(ToTensorV2())  # 将图像转换为张量
transform = A.Compose(transform_operations, p=1)
transformed_data = transform(image=image_np)

transformed_image = transformed_data['image']
transformed_image = transformed_image.unsqueeze(0).to(device)  # 添加批次维度并发送到设备上


sam得到的没问题了
image_encoder = ImageEncoderViT()
image_encoder = image_encoder.to(device)
image_embedding = image_encoder(transformed_image)
print(image_embedding)
print(image_embedding.shape)
#存图！


cfg = Config.fromfile(f'/data/hotaru/my_projects/PNS_tmp/prompter/config/pannuke321.py')
backbone = Backbone(cfg)
backbone = backbone.to(device)
prompter_output1,prompter_output2 = backbone(transformed_image)
print(prompter_output1)
print(prompter_output1[0].shape)