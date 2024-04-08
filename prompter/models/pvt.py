import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvt_v2 import pvt_v2_b2
from mmengine.config import Config
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dpa_p2pnet import Backbone

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1




class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PolypPVT(nn.Module):
    def __init__(self, channel=256):
        super(PolypPVT, self).__init__()
        self.l1 = 64
        self.l2 = 128
        self.l3 = 320
        self.l4 = 512

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/data/hotaru/projects/PNS_tmp/prompter/models/pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(self.l1, channel, 1)
        self.Translayer2_1 = BasicConv2d(self.l2, channel, 1)
        self.Translayer3_1 = BasicConv2d(self.l3, channel, 1)
        self.Translayer4_1 = BasicConv2d(self.l4, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM()
        
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        # # # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention


        # # CFM
        x1_t = self.Translayer2_0(x1) #我自己加的
        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4)  
        # cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # # SAM
        # T2 = self.Translayer2_0(cim_feature)
        # T2 = self.down05(T2)
        # sam_feature = self.SAM(cfm_feature, T2)

        # prediction1 = self.out_CFM(cfm_feature)
        # prediction2 = self.out_SAM(sam_feature)

        # prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        # prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        # return prediction1_8, prediction2_8
        return x1_t,x2_t,x3_t,x4_t
        # return cim_feature,x2_t

def convert_to_rgb(feature_map):
    # 将单通道特征图重复三次，创建三通道图像
    rgb_image = torch.cat([feature_map, feature_map, feature_map], dim=1)
    return rgb_image
import torchvision.transforms as transforms
def save_tensor_as_image(tensor, save_path):
    """
    将张量保存为图像文件。
    
    参数：
        tensor (Tensor): 要保存的张量。
        save_path (str): 图像文件保存路径。
    """
    # 将张量转换为 PIL 图像
    to_pil = transforms.ToPILImage()
    
    image = to_pil(tensor.squeeze().cpu())
    
    # 保存图像到指定路径
    image.save(save_path)
    print(f"图像已保存到 {save_path}")
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
if __name__ == '__main__':
    model = PolypPVT().cuda()
    cfg = Config.fromfile('/data/hotaru/projects/PNS_tmp/prompter/config/cpm17.py')
    # backbone = Backbone(cfg).cuda()
    additional_targets = {}
    for i in range(1, cfg.data.num_classes):
        additional_targets.update({'keypoints%d' % i: 'keypoints'})

    transform = A.Compose(
            [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get('train').transform] + [ToTensorV2()],
            p=1, keypoint_params=None,
            additional_targets=additional_targets, is_check_shapes=False
        )
    # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    input_image_path = "/data/hotaru/projects/PNS_tmp/segmentor/datasets/cpm17/train/Images/image_22.png"
    # input_image_path = "/data/hotaru/projects/PNS_tmp/segmentor/datasets/pannuke/Images/1_0.png"
    input_image = Image.open(input_image_path) # 1 3 500 500
    input_numpy = np.array(input_image)
    input_tensor = transform(image = input_numpy)['image'].cuda()
    print(input_tensor.shape) 
    # input_tensor = torch.from_numpy(input_image['image']).cuda()


    # feats,_ = backbone(input_tensor)
    # feats1_map = torch.mean(feats[0], dim=1, keepdim=True)
    # feats1_map = convert_to_rgb(feats1_map)
    # save_tensor_as_image(feats1_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/feats1.png")
    # feats2_map = torch.mean(feats[1], dim=1, keepdim=True)
    # save_tensor_as_image(feats2_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/feats2.png")
    # feats3_map = torch.mean(feats[2], dim=1, keepdim=True)
    # save_tensor_as_image(feats3_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/feats3.png")
    # feats4_map = torch.mean(feats[3], dim=1, keepdim=True)
    # save_tensor_as_image(feats4_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/feats4.png")
    # print(feats[0].shape,feats[1].shape,feats[2].shape,feats[3].shape) 


    # prediction1, prediction2 = model(input_tensor)
    # print(prediction1.size(), prediction2.size()) 
    # save_tensor_as_image(prediction1, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/prediction1.png")
    # save_tensor_as_image(prediction2, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/prediction2.png")
    save_tensor_as_image(input_tensor, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/input_tensor.png")
    print(input_tensor.shape) 
    x1,x2,x3,x4 = model(input_tensor)
    print(x1.shape,x2.shape,x3.shape,x4.shape)
    x1_map = torch.mean(x1, dim=1, keepdim=True)
    save_tensor_as_image(x1_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/x1_256.png")
    x2_map = torch.mean(x2, dim=1, keepdim=True)
    save_tensor_as_image(x2_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/x2_256.png")
    x3_map = torch.mean(x3, dim=1, keepdim=True)
    save_tensor_as_image(x3_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/x3_256.png")
    x4_map = torch.mean(x4, dim=1, keepdim=True)
    save_tensor_as_image(x4_map, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/x4_256.png")


    input_image_resized = F.interpolate(input_tensor, size=(x1_map.size(2), x1_map.size(3)), mode='bilinear', align_corners=False)

    # 将 x1_map 叠加到调整大小后的图像上
    combined_image = input_image_resized.clone()
    combined_image[:, :, :x1_map.size(2), :x1_map.size(3)] += x1_map

    # 保存叠加后的图像
    save_tensor_as_image(combined_image, "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/combined_image_with_x1.png")
    