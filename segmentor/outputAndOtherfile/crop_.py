import os
from PIL import Image
import random
import scipy.io as sio
import numpy as np

def random_crop_image(input_image_path, output_image_path, crop_size=(256, 256)):
    # 打开图片
    img = Image.open(input_image_path)
    
    # 获取图片原始大小
    width, height = img.size
    
    # 计算裁剪的左上角坐标（随机）
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])
    
    # 使用PIL的crop方法裁剪图片
    cropped_img = img.crop((left, top, left + crop_size[0], top + crop_size[1]))
    
    # 保存裁剪后的图片
    cropped_img.save(output_image_path)
    
    return left, top

def crop_label_and_binary(input_mat_path, input_binary_path, output_mat_path, output_binary_path, left, top, crop_size=(256, 256)):
    # 裁剪并保存label（inst_map）
    mat = sio.loadmat(input_mat_path)
    inst_map = mat['inst_map']
    
    # 裁剪label
    cropped_inst_map = inst_map[top:top + crop_size[1], left:left + crop_size[0]]
    
    # 保存裁剪后的label
    sio.savemat(output_mat_path, {'inst_map': cropped_inst_map})
    
    # 打开二进制图像
    binary_img = Image.open(input_binary_path)
    
    # 裁剪二进制图像
    cropped_binary_img = binary_img.crop((left, top, left + crop_size[0], top + crop_size[1]))
    
    # 保存裁剪后的二进制图像
    cropped_binary_img.save(output_binary_path)

# 使用函数
input_image_path = '/data/hotaru/projects/PNS_tmp/segmentor/datasets/cpm17/test/Images/image_04.png'
output_image_path = '/data/hotaru/projects/PNS_tmp/segmentor/datasets/cpm17/reference/cropped_image.jpg'

input_mat_path = '/data/hotaru/projects/PNS_tmp/segmentor/datasets/cpm17/test/Labels/image_04.mat'
output_mat_path = '/data/hotaru/projects/PNS_tmp/segmentor/datasets/cpm17/reference/cropped_label.mat'

input_binary_path = '/data/hotaru/projects/sam-hq/data/cpm17/test/Labels_binary_png/image_04.png'
output_binary_path = '/data/hotaru/projects/PNS_tmp/segmentor/datasets/cpm17/reference/cropped_binary.png'

left, top = random_crop_image(input_image_path, output_image_path, crop_size=(256, 256))
crop_label_and_binary(input_mat_path, input_binary_path, output_mat_path, output_binary_path, left, top, crop_size=(256, 256))

print(f"Cropped image saved to {output_image_path}")
print(f"Cropped label saved to {output_mat_path}")
print(f"Cropped binary image saved to {output_binary_path}")
