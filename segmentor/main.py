import sys
import wandb
import math
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageDraw
torch.cuda.set_device(6)
import torch.nn.functional as F
from scipy.io import savemat
from mmengine.config import Config
from criterion import build_criterion

from utils import *
from dataset import DataFolder
from torch.utils.data import DataLoader
import segment_anything

from torch.utils.data.distributed import DistributedSampler
from torchvision.ops.boxes import batched_nms
from stats_utils import (
    remap_label,
    get_fast_pq,
    get_fast_aji,
    get_dice_1,
    get_fast_aji_plus,

)

import argparse

 
def parse_args():
    parser = argparse.ArgumentParser('Cell segmentor')

    parser.add_argument('--config', default='pannuke123.py', help='config file')
    parser.add_argument('--output_path', default='', help='config file')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument("--eval", action='store_true', help='only evaluate')
    parser.add_argument("--overlap", default=64, type=int, help="overlapping pixels")

    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--start-eval", default=5, type=int)

    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--tta', action='store_true')

    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument('--run-name', default=None, type=str, help='wandb run name')
    parser.add_argument('--group-name', default=None, type=str, help='wandb group name')

    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    # * Distributed training
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()

    return opt


def main():
    args = parse_args()
    init_distributed_mode(args)
    set_seed(args)

    print(args)

    cfg = Config.fromfile(f'config/{args.config}')
    if args.output_dir:
        mkdir(f'checkpoint/{args.output_dir}')
        cfg.dump(f'checkpoint/{args.output_dir}/config.py')

    train_dataset = DataFolder(cfg, mode='train')
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=train_sampler is None,
        num_workers=cfg.data.num_workers,
        sampler=train_sampler,
        collate_fn=train_collate_fn,
    )

    try:
        val_dataset = DataFolder(cfg, mode='val')
        val_dataloader = DataLoader(
            val_dataset,  #cpm的npy文件复制了train的
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=test_collate_fn,
        )
    except FileNotFoundError:
        #可以这么干吗？但是cpm不validation 没必要了直接复制了cpm的train
    #     val_dataset = DataFolder(cfg, mode='val')
    #     val_dataloader = DataLoader( 
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=cfg.data.num_workers,
    #     collate_fn=test_collate_fn,
    # )
        pass

    test_dataset = DataFolder(cfg, mode='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=test_collate_fn,
    )

    device = torch.device(args.device)

    model = getattr(segment_anything, f"build_sam_vit_{cfg.segmentor.type[-1].lower()}")(cfg)
    model.to(device)

    for param in model.prompt_encoder.parameters(): #是这样冻结的吗？
        param.requires_grad = False

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = build_criterion(cfg).to(device)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        filter(lambda p: p.requires_grad, model_without_ddp.parameters()),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.type)(
        optimizer,
        milestones=cfg.scheduler.milestones,
        gamma=cfg.scheduler.gamma,
    )

    max_mPQ = 0
    metric_dict = {}
    if args.resume:
        # checkpoint = torch.load(
        #     './checkpoint/%s/%s.pth' % (args.resume, 'best' if args.eval else 'latest'),
        #     map_location="cpu"
        # )

        checkpoint = torch.load(
            args.resume,
            map_location="cpu"
        )

        model_without_ddp.load_state_dict(checkpoint["model"])
        # model_without_ddp.load_state_dict(checkpoint)  #vanillaSAM用的这个

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1  #也就是加上这个可以接着train吧
        if "metrics" in checkpoint:
            print(checkpoint['metrics'], checkpoint['epoch'])
            metric_dict = checkpoint['metrics']
            max_mPQ = checkpoint["metrics"].get('mPQ', 0)

    if args.use_wandb and is_main_process():
        wandb.init(
            project="Segmentor",
            name=args.run_name,
            group=args.group_name,
            config=vars(args)
        )
    
    if args.eval:
        return evaluate(
            args,
            model,
            test_dataloader,
            cfg.data.num_classes,
            cfg.data.post.iou_threshold,
            args.tta,
            device,
            cfg.data,
        )

    for epoch in range(args.start_epoch, args.epochs):

        wandb_log_info = train_on_epoch(
            args,
            model,
            train_dataloader,
            criterion,
            optimizer,
            device,
            epoch
        )
        scheduler.step()

        save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'metrics': metric_dict
            },
            f"checkpoint/{args.output_dir}/latest.pth",
        )

        try:
            if epoch >= args.start_eval:
                metric_dict = evaluate(
                    args,
                    model,
                    val_dataloader,
                    cfg.data.num_classes,
                    cfg.data.post.iou_threshold,
                    args.tta,
                    device
                )

                mPQ = metric_dict['mPQ']
                if max_mPQ < mPQ:
                    max_mPQ = mPQ
                    save_on_master(
                        {
                            'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'metrics': metric_dict
                        },
                        f"checkpoint/{args.output_dir}/best.pth",
                    )

                wandb_log_info.update(metric_dict)
        except NameError:
            pass

        if args.use_wandb and is_main_process():
            wandb.log(
                wandb_log_info,
                step=epoch
            )

    if args.use_wandb and is_main_process():
        wandb.finish()


def train_on_epoch(
        args,
        model,
        train_dataloader,
        criterion,
        optimizer,
        device,
        epoch
):
    if args.distributed:
        train_dataloader.sampler.set_epoch(epoch)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    wandb_log_info = {}
    model.train()

    for data_iter_step, (images, true_masks, prompt_points, prompt_labels, all_points, all_points_types, cell_nums) in (
            enumerate(metric_logger.log_every(train_dataloader, args.print_freq, header))):
        images = images.to(device) #torch.Size([16, 3, 256, 256])
        true_masks = true_masks.to(device) #torch.Size([256, 256])

        prompt_points = prompt_points.to(device) #torch.Size([239, 1, 2]) 239难道是mask嘛？
        prompt_labels = prompt_labels.to(device) #torch.Size([239, 1])
        
        cell_nums = cell_nums.to(device) #torch.Size([16])

        outputs = model(   #here  infer的时候把b重复了cell_num次
            images,
            prompt_points, 
            prompt_labels,
            cell_nums
        )

        loss_dict = criterion(
            outputs,
            true_masks,
        )

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        metric_logger.update(
            loss=losses_reduced,
            lr=optimizer.param_groups[0]["lr"],
            **loss_dict_reduced
        )

        for k, v in loss_dict_reduced.items():
            wandb_log_info[k] = wandb_log_info.get(k, 0) + v.item()

    return wandb_log_info


@torch.inference_mode()
def evaluate(
        args,
        model,
        test_dataloader,
        num_classes,
        iou_threshold,
        tta,
        device,
        data_info,
):
    model.eval()

    tissue_types = {
        'Adrenal_gland': 0,
        'Bile-duct': 1,
        'Bladder': 2,
        'Breast': 3,
        'Cervix': 4,
        'Colon': 5,
        'Esophagus': 6,
        'HeadNeck': 7,
        'Kidney': 8,
        'Liver': 9,
        'Lung': 10,
        'Ovarian': 11,
        'Pancreatic': 12,
        'Prostate': 13,
        'Skin': 14,
        'Stomach': 15,
        'Testis': 16,
        'Thyroid': 17,
        'Uterus': 18
    }

    nuclei_types = {
        'Neoplastic': 1,
        'Inflammatory': 2,
        'Connective': 3,
        'Dead': 4,
        'Epithelial': 5,
    }

    nuclei_pq_scores = []  # image_id, category_id
    nuclei_sq_scores = []
    nuclei_dq_scores = []
    nuclei_dice_scores = []
    nuclei_aji_scores = []
    nuclei_aji_plus_scores = []
    tissue_nuclei_pq_scores = [[] for _ in tissue_types]  # tissue_id, category_id

    binary_pq_scores = []  # image_id
    tissue_binary_pq_scores = [[] for _ in tissue_types]  # tissue_id, score

    binary_dq_scores = []
    binary_sq_scores = []
    binary_aji_scores = []
    binary_aji_plus_scores = []
    binary_dice_scores = []

    aji_scores = []
    dice_scores = []
    aji_plus_scores = []

    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test:"

    excel_info = []
    for data_iter_step, (images, inst_maps, type_maps, prompt_points, prompt_labels, prompt_cell_types,
                         cell_nums, ori_sizes, file_inds) in (
            enumerate(metric_logger.log_every(test_dataloader, args.print_freq, header))):

        if data_iter_step % get_world_size() != get_rank():  # To avoid repetitive calculation
            continue

        images = images.to(device)
        inst_maps = inst_maps.numpy()

        type_maps = F.one_hot(
            type_maps.type(torch.int64),
            num_classes + 1
        ).numpy()

        instance_types_nuclei = type_maps * np.expand_dims(inst_maps, -1)
        instance_types_nuclei = instance_types_nuclei.transpose(0, 3, 1, 2)  # b h w c

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        batch_inds = torch.repeat_interleave(torch.arange(images.shape[0]), cell_nums)
        if 'pannuke' in test_dataloader.dataset.dataset:
            if cell_nums.sum() > 0:
                outputs = model( #infer的第一个outputs
                    images,
                    prompt_points.to(device),
                    prompt_labels.to(device),
                    cell_nums.to(device)
                )
            
            # output_folder = '/data/hotaru/my_projects/PromptNucSeg/segmentor/output/pannuke321_outputmat_test'  # 指定保存的文件夹路径
            # # 获取文件名，并替换后缀为 .mat
            # output_filename = test_dataloader.dataset.files[file_inds].split(".")[0].split("/")[-1] + '.mat'
            # # 拼接完整的文件路径
            # output_file_path = os.path.join(output_folder, output_filename)
            # # 准备要保存的数据
            # data_to_save = {
            #     'pred_masks': outputs['pred_masks'].cpu().numpy(),
            #     'pred_ious': outputs['pred_ious'].cpu().numpy(),
            # }
            # # 使用 savemat() 函数将数据保存为 .mat 文件
            # savemat(output_file_path, data_to_save)

            model_time = time.time() - model_time
            metric_logger.update(model_time=model_time)
 
            for batch_ind, file_ind in enumerate(file_inds): 

                c_inst_map = np.zeros((num_classes, *inst_maps.shape[-2:])) #(5, 256, 256)
                b_inst_map = np.zeros_like(inst_maps[0]) #(256, 256)

                if cell_nums.sum() > 0:

                    mask_data = MaskData(
                        masks=outputs["pred_masks"][batch_inds == batch_ind],
                        iou_preds=outputs["pred_ious"][batch_inds == batch_ind],
                        categories=prompt_cell_types[batch_inds == batch_ind],
                    )

                    # Threshold masks and calculate boxes
                    mask_threshold = 0.0
                    mask_data["masks"] = mask_data["masks"] > mask_threshold  #torch.Size([11, 256, 256])
                    mask_data["boxes"] = batched_mask_to_box(mask_data["masks"]) #torch.Size([11, 4])
                    mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"]) #size 掩码大小,count RLE编码

                    if len(mask_data["masks"]) > 0: #去重，分别把mask映射到class和binary
                        # Remove duplicates within this crop.
                        keep_by_nms = batched_nms(
                            mask_data["boxes"].float(),
                            mask_data["iou_preds"],
                            torch.zeros_like(mask_data["boxes"][:, 0]),  # apply cross categories
                            iou_threshold=iou_threshold
                        ).cpu().numpy()
                        order = keep_by_nms[::-1] #根据box和iou获取了一个顺序？

                        mask_data["masks"] = mask_data["masks"].cpu().numpy()
                        for iid, ind in enumerate(order):
                            c_inst_map[int(mask_data['categories'][ind]), mask_data['masks'][ind]] = iid + 1

                        # b_inst_map = binarize(c_inst_map.transpose(1, 2, 0))
                        for iid, ind in enumerate(order):
                            b_inst_map[mask_data['masks'][ind]] = iid + 1

                    # output_folder = '/data/hotaru/my_projects/PromptNucSeg/segmentor/output/pannuke321_b_inst_map_test'  # 指定保存的文件夹路径
                    # # 获取文件名，并替换后缀为 .mat
                    # import matplotlib.pyplot as plt
                    # # 可视化二值实例地图
                    # plt.imshow(b_inst_map, cmap='gray')
                    # plt.axis('off')  # 不显示坐标轴
                    # output_filename = test_dataloader.dataset.files[file_inds].split(".")[0].split("/")[-1] + '.png' 
                    # # 拼接完整的文件路径
                    # output_file_path = os.path.join(output_folder, output_filename)
                    # plt.savefig(output_file_path)  # 保存为图片文件
                    
                if len(np.unique(inst_maps[batch_ind])) == 1 or len(np.unique(b_inst_map)) == 1:
                    bpq_tmp = np.nan
                    bdq_tmp = np.nan
                    bsq_tmp = np.nan
                    bdice_tmp = np.nan
                    baji_tmp = np.nan
                    baji_plus_tmp = np.nan
                else:
                    [bdq_tmp, bsq_tmp, bpq_tmp], _ = get_fast_pq(
                        remap_label(inst_maps[batch_ind]),  #true
                        remap_label(b_inst_map)
                    )
                    bdice_tmp = get_dice_1(
                        remap_label(inst_maps[batch_ind]),  #true
                        remap_label(b_inst_map)
                    )
                    baji_tmp = get_fast_aji(
                        remap_label(inst_maps[batch_ind]),
                        remap_label(b_inst_map)
                    )
                    baji_plus_tmp = get_fast_aji_plus(
                        remap_label(inst_maps[batch_ind]),
                        remap_label(b_inst_map)
                    )
                # 这里是不带类别的所有的指标，先搞这个 这里不append是单个image的，append完就算完了,但是是for循环结束才完的
                #bqp_tmp是单张
                #出了for循环 binary才存了所有图的结果，再算mean
                binary_pq_scores.append(bpq_tmp)
                binary_dq_scores.append(bdq_tmp)
                binary_sq_scores.append(bsq_tmp)
                binary_aji_scores.append(baji_tmp)
                binary_dice_scores.append(bdice_tmp)
                binary_aji_plus_scores.append(baji_plus_tmp)

                tissue_binary_pq_scores[tissue_types[test_dataloader.dataset.types[file_ind]]].append(bpq_tmp) #tissue的就先算了

                # 单张图片5个类别（单张图片保存的应该是五个的平均吗？）
                nuclei_type_pq = []  
                nuclei_type_dq = []
                nuclei_type_sq = []
                nuclei_type_dice = []
                nuclei_type_aji = []
                nuclei_type_aji_plus = []

                for c in range(num_classes):  # 5 class 按类别的inst_map score
                    pred_nuclei_instance_class = remap_label(
                        c_inst_map[c]
                    )
                    target_nuclei_instance_class = remap_label(
                        instance_types_nuclei[batch_ind][c + 1]
                    )

                    if len(np.unique(target_nuclei_instance_class)) == 1:
                        mpq_tmp = np.nan
                        mdq_tmp = np.nan
                        msq_tmp = np.nan
                        mdice_tmp = np.nan
                        maji_tmp = np.nan
                        maji_plus_tmp = np.nan
                    else:
                        [mdq_tmp, msq_tmp, mpq_tmp], _ = get_fast_pq(
                            pred_nuclei_instance_class,
                            target_nuclei_instance_class
                        )
                        mdice_tmp = get_dice_1(
                            pred_nuclei_instance_class,
                            target_nuclei_instance_class
                        )
                        maji_tmp = get_fast_aji(
                            pred_nuclei_instance_class,
                            target_nuclei_instance_class
                        )
                        maji_plus_tmp = get_fast_aji_plus(
                            pred_nuclei_instance_class,
                            target_nuclei_instance_class
                        )
                    # 这里按照类别计算了所有的指标，照葫芦画瓢按类别计算一下dice和aji，然后看看pq和dq也给他传过去
                    nuclei_type_pq.append(mpq_tmp)
                    nuclei_type_dq.append(mdq_tmp)
                    nuclei_type_sq.append(msq_tmp)
                    nuclei_type_aji.append(maji_tmp)
                    nuclei_type_dice.append(mdice_tmp)
                    nuclei_type_aji_plus.append(maji_plus_tmp)
                # 所有图像的五个类别值
                nuclei_pq_scores.append(nuclei_type_pq)
                nuclei_sq_scores.append(nuclei_type_sq)
                nuclei_dq_scores.append(nuclei_type_dq)
                nuclei_aji_scores.append(nuclei_type_aji)
                nuclei_dice_scores.append(nuclei_type_dice)
                nuclei_aji_plus_scores.append(nuclei_type_aji_plus)

                # 要不全在这里算了然后存csv呢？ 但这个是一张图的结果 但按照这个逻辑就应该按图算指标然后算mean
                tissue_nuclei_pq_scores[tissue_types[test_dataloader.dataset.types[file_ind]]].append(nuclei_type_pq)

                excel_info.append(
                    (test_dataloader.dataset.files[file_ind].split("/")[-1],bdice_tmp,baji_tmp,bdq_tmp,bsq_tmp, bpq_tmp, 
                     np.nanmean(nuclei_dice_scores[-1]),np.nanmean(nuclei_aji_scores[-1]),np.nanmean(nuclei_dq_scores[-1]),np.nanmean(nuclei_sq_scores[-1]),
                     np.nanmean(nuclei_pq_scores[-1]),
                     len(np.unique(inst_maps[batch_ind])) - 1, cell_nums[batch_ind].item()))
            

        else:  # applicable when the resolution of input image is larger than 256
            # 确认一下哪个数据集会用这里
            print("else!over 256")
            assert len(images) == 1, 'batch size must be 1'

            crop_boxes = crop_with_overlap(
                images[0],
                *(model.module.image_encoder.img_size if args.distributed else model.image_encoder.img_size,) * 2,
                args.overlap,
            ).tolist()

            all_masks = []
            all_boxes = []
            all_scores = []
            all_classes = []
            all_inds = []

            inds = torch.arange(len(prompt_points))

            for idx, crop_box in enumerate(crop_boxes):
                x1, y1, x2, y2 = crop_box

                keep = (prompt_points[..., 0] >= x1) & (prompt_points[..., 0] < x2) & \
                       (prompt_points[..., 1] >= y1) & (prompt_points[..., 1] < y2)
                keep = keep.squeeze(1)

                if keep.sum() == 0:
                    continue

                sub_prompt_points = prompt_points[keep] - torch.as_tensor([x1, y1])
                sub_prompt_labels = prompt_labels[keep]

                k = test_dataloader.dataset.num_neg_prompt
                if k > 0:
                    from dataset import add_k_nearest_neg_prompt
                    sub_prompt_points, sub_prompt_labels = add_k_nearest_neg_prompt(
                        sub_prompt_points,
                        torch.arange(len(sub_prompt_points)),
                        sub_prompt_points,
                        k=k
                    )

                    sub_prompt_points1 = sub_prompt_points.clone()
                    sub_prompt_labels1 = sub_prompt_labels.clone()
                    sub_prompt_labels1[..., 1:] = -1

                    sub_prompt_points = torch.cat([sub_prompt_points, sub_prompt_points1]).to(device)
                    sub_prompt_labels = torch.cat([sub_prompt_labels, sub_prompt_labels1]).to(device)

                else:

                    sub_prompt_points = sub_prompt_points.to(device)
                    sub_prompt_labels = sub_prompt_labels.to(device)

                masks = inference(
                    model,
                    images[..., y1:y2, x1:x2],
                    crop_box,
                    ori_sizes[0],
                    sub_prompt_points,
                    sub_prompt_labels,
                    prompt_cell_types[keep] if k == 0 else prompt_cell_types[keep].repeat(k + 1),
                    pred_iou_thresh=0.0,
                    stability_score_thresh=0.0,
                    # min_mask_region_area=70,  # NOTE: very slow processing speed but slightly better performance
                    min_mask_region_area=0,  # NOTE: used in our paper
                    inds=inds[keep] if k == 0 else inds[keep].repeat(k + 1),
                    tta=tta
                )

                for mask_data in masks:
                    all_scores.append(mask_data['predicted_iou'])
                    all_masks.append(mask_data['segmentation'][:ori_sizes[0, 0], :ori_sizes[0, 1]])
                    all_boxes.append(mask_data['bbox'])
                    all_classes.append(mask_data['categories'])

                    all_inds.append(mask_data['inds'])

            model_time = time.time() - model_time
            metric_logger.update(model_time=model_time)

            all_boxes = torch.as_tensor(all_boxes)
            all_scores = torch.as_tensor(all_scores)

            all_inds = np.asarray(all_inds)
            unique_inds, counts = np.unique(all_inds, return_counts=True)

            # first-aspect NMS
            keep_prior = np.ones(len(all_inds), dtype=bool)
            for i in np.where(counts > 1)[0]:
                inds = np.where(all_inds == unique_inds[i])[0]
                inds = np.delete(inds, np.argmax(all_scores[inds]))
                keep_prior[inds] = False
            keep_prior = torch.from_numpy(keep_prior)

            all_boxes = all_boxes[keep_prior]
            all_scores = all_scores[keep_prior]
            all_masks = [all_masks[ind] for ind in np.where(keep_prior)[0]]

            '''
            一些可视化------------------------------------------------------------------
            '''
            output_filename = test_dataloader.dataset.files[file_inds].split(".")[0].split("/")[-1] 
            
            sub_prompt_points[0]
            pred_instance_map = np.zeros_like(all_masks[0], dtype=np.uint8)

            # 把all_mask的array按照实例保存
            for i, mask in enumerate(all_masks, start=1):
                pred_instance_map[mask] = i

            # 着色
            pred_instance_colored = cv2.applyColorMap(pred_instance_map, cv2.COLORMAP_JET)
            # 在图像上标记点
            # for point in sub_prompt_points:好奇怪哦，那sub是干什么的呢
            for point in prompt_points:
                x, y = point[0]
                x, y = int(x.item()), int(y.item())  
                cv2.circle(pred_instance_colored, (x, y), radius=2, color=(255,255,255), thickness=-1) 

            # 将图像转换为PIL格式
            pred_instance_image = Image.fromarray(cv2.cvtColor(pred_instance_colored, cv2.COLOR_BGR2RGB))
            to_pil = transforms.ToPILImage()
            image_save = to_pil(images[0].cpu())#0是batch = 1

            inst_map_tensor = torch.from_numpy(inst_maps)
            # gt_mask = to_pil(inst_map_tensor[0].cpu())
            gt_mask_colored = cv2.applyColorMap(inst_map_tensor[0].cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
            for point in prompt_points:
                x, y = point[0]
                x, y = int(x.item()), int(y.item())  
                cv2.circle(gt_mask_colored, (x, y), radius=2, color=(255,255,255), thickness=-1) 
            gt_instance_image = Image.fromarray(cv2.cvtColor(gt_mask_colored, cv2.COLOR_BGR2RGB))


            print("images.shape",images.shape)
            print("pred_instance_image.shape",pred_instance_map.shape)
            print("gt_instance_image.shape",inst_map_tensor.shape)
            print("gt point数量：",len(all_masks))
            print("sub_prompt_points:",len(sub_prompt_points))
            print("prompt_points:",len(prompt_points))
            print("实例图已保存为",output_filename)

            width = max(image_save.width, gt_instance_image.width, pred_instance_image.width)
            height = image_save.height + gt_instance_image.height + pred_instance_image.height
            combined_image = Image.new("RGB", (width, height))
            combined_image.paste(image_save, (0, 0))
            combined_image.paste(gt_instance_image, (0, image_save.height))
            combined_image.paste(pred_instance_image, (0, image_save.height + gt_instance_image.height))
            os.makedirs(args.output_path, exist_ok=True)
            combined_image.save(args.output_path + output_filename + '.png')


            # -----------------------------------------------------------------------
            # -----------------------------------------------------------------------

            # second-aspect NMS
            keep_by_nms = batched_nms(
                all_boxes.float(),
                all_scores,
                torch.zeros_like(all_boxes[:, 0]),  # apply cross categories
                iou_threshold=iou_threshold
            ).numpy()
            order = keep_by_nms[::-1]
            b_inst_map = np.zeros_like(inst_maps[0], dtype=int)
            for iid, ind in enumerate(order):
                b_inst_map[all_masks[ind]] = iid + 1

            if len(np.unique(inst_maps[0])) == 1 or len(b_inst_map)==1 or len(b_inst_map)==0 or len(np.unique(inst_maps[0])) == 0:
                bpq_tmp = np.nan
                bdq_tmp = np.nan
                bsq_tmp = np.nan 
                baji_tmp = np.nan 
                baji_plus_tmp  = np.nan 
                bdice_tmp  = np.nan 
            else:
                [bdq_tmp, bsq_tmp, bpq_tmp], _ = get_fast_pq(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )
                bdice_tmp = get_dice_1(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )
                baji_plus_tmp = get_fast_aji_plus(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )
                # baji_tmp = get_fast_aji(
                #     remap_label(inst_maps[0]),
                #     remap_label(b_inst_map)
                # )

            # aji_score = get_fast_aji(
            #     remap_label(inst_maps[0]),
            #     remap_label(b_inst_map)
            # )
            dice_score = get_dice_1(
                remap_label(inst_maps[0]),
                remap_label(b_inst_map)
            )
            aji_plus_score = get_fast_aji_plus(
                remap_label(inst_maps[0]),
                remap_label(b_inst_map)
            )
            binary_dq_scores.append(bdq_tmp)
            binary_sq_scores.append(bsq_tmp)

            binary_pq_scores.append(bpq_tmp)
            binary_aji_plus_scores.append(baji_plus_tmp)
            binary_dice_scores.append(bdice_tmp)
            # binary_aji_scores.append(baji_tmp)


            # aji_scores.append(aji_score)
            dice_scores.append(dice_score)
            aji_plus_scores.append(aji_plus_score)
            excel_info.append(
                (test_dataloader.dataset.files[file_inds[0]].split("/")[-1],
                 bpq_tmp,
                #  aji_score,
                 len(np.unique(inst_maps[batch_inds[0]])) - 1,
                 cell_nums[batch_inds[0]].item())
            )

    # 到这里循环完了所有的图↑
    print("binary指标："+"*"*10)
    # print("检查一下长度：",len(binary_aji_scores))
    # print("检查一下形状：",binary_dice_scores.shape)
    print("dice:",np.nanmean(binary_dice_scores))
    print("aji:",np.nanmean(binary_aji_scores))
    print("dq:",np.nanmean(binary_dq_scores))
    print("sq:",np.nanmean(binary_sq_scores))
    print("pq:",np.nanmean(binary_pq_scores))
    print("aji_p:",np.nanmean(binary_aji_plus_scores))
    print("检查一下长度：",len(binary_aji_scores))
    print("*"*20)
    if 'pannuke' in test_dataloader.dataset.dataset:  # PanNuke  tissue

        tissue_mpq_scores = []
        for tid, tissue_type in enumerate(tissue_types):
            tmp = [np.asarray(_).reshape(-1, num_classes) for _ in all_gather(tissue_nuclei_pq_scores[tid])]
            tissue_mpq_scores.append(
                np.nanmean(
                    np.nanmean(np.concatenate(tmp), axis=1)
                ))

        tissue_mPQ = np.nanmean(tissue_mpq_scores)

        tissue_bpq_scores = []
        for tid, tissue_type in enumerate(tissue_types):
            tissue_bpq_scores.append(
                np.nanmean(np.concatenate(all_gather(tissue_binary_pq_scores[tid])))
            )

        tissue_bPQ = np.nanmean(tissue_bpq_scores)

        nuclei_pq_scores = np.concatenate(all_gather(nuclei_pq_scores))
        binary_pq_scores = np.concatenate(all_gather(binary_pq_scores))

        nuclei_pq_scores = np.asarray(nuclei_pq_scores)
        print("values：",np.nanmean(nuclei_pq_scores, axis=0))

        global_mPQ = np.nanmean(np.nanmean(nuclei_pq_scores, axis=1))
        
        binary_pq_scores = np.asarray(binary_pq_scores)
        global_bPQ = np.nanmean(binary_pq_scores) #这个好像就是我print的东西，确实是 作者是发现这样计算会低吗？
        try:
            print("1检查一下tissue_bpq_scores形状：",len(tissue_bpq_scores))
            print("2检查一下tissue_binary_pq_scores形状：",len(tissue_binary_pq_scores))
            print("3检查一下tissue_binary_pq_scores[0]形状：",len(tissue_binary_pq_scores[0]))
            
        except:
            print("ok fine")
        try:
            print("4检查一下concatenate形状：",len(np.concatenate(all_gather(tissue_binary_pq_scores[tid]))))
        except:
            print("ok fine")
        try:
            print("5检查一下concatenate形状：",np.concatenate(all_gather(tissue_binary_pq_scores[tid])).shape )
        except:
            print("ok fine")
        metrics = {
            'mPQ': tissue_mPQ,  # <---
            'bPQ': tissue_bPQ,
            'global_bPQ': global_bPQ,
            'global_mPQ': global_mPQ,
        }

    else:  # CPM-17 and Kumar
        aji_scores = np.concatenate(all_gather(aji_scores))
        AJI = np.nanmean(aji_scores)
        
        dice_scores = np.concatenate(all_gather(dice_scores))
        DICE = np.nanmean(dice_scores)

        pq_scores = np.concatenate(all_gather(binary_pq_scores))
        PQ = np.nanmean(pq_scores)

        dq_scores = np.concatenate(all_gather(binary_dq_scores))
        DQ = np.nanmean(dq_scores)

        sq_scores = np.concatenate(all_gather(binary_sq_scores))
        SQ = np.nanmean(sq_scores)

        aji_plus_scores = np.concatenate(all_gather(binary_aji_plus_scores))
        AJI_P = np.nanmean(aji_plus_scores)
        metrics = {
            'DICE:': DICE,
            'AJI:': AJI,
            'DQ:' : DQ,
            'SQ:' : SQ,
            'PQ': PQ,
            'AJI_P':AJI_P,
        }

    for k, v in metrics.items():
        print(f"{k}: {v}")

    gathered_excel_info = []
    for _ in all_gather(excel_info):
        gathered_excel_info.extend(_)

    if is_main_process():
        print("save!")
        pd.DataFrame(
            data=gathered_excel_info,
            # columns=['Imagee Name', 'bDice', 'bAji','bdq', 'bsQ',  'bPQ', 'mDice', 'mAji','mDQ','mSQ','mPQ','GT Num', 'PD Num']
            columns=['Imagee Name', 'PQ', 'AJI', 'GT Num', 'PD Num']
        ).to_csv(f'{test_dataloader.dataset.dataset}.csv')

    return metrics


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    main()
