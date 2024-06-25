import sys
import math
import itertools
import prettytable as pt
from PIL import Image
from utils import *
from tqdm import tqdm
from eval_map import eval_map
from collections import OrderedDict
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def train_one_epoch(
        args,
        model,
        train_loader,
        criterion,
        optimizer,
        epoch,
        device,
        model_ema=None,
        scaler=None,
        save_middle_path_name='tmp'
):
    model.train()
    criterion.train()

    log_info = dict()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for data_iter_step, (images, masks, points_list, labels_list) in enumerate(
            metric_logger.log_every(train_loader, args.print_freq, header)):
        images = images.to(device) # cpm:torch.Size([8, 3, 512, 512])   pannuke:torch.Size([16, 3, 256, 256])
        masks = masks.to(device)

        targets = {
            'gt_masks': masks,
            'gt_nums': [len(points) for points in points_list],
            'gt_points': [points.view(-1, 2).to(device).float() for points in points_list],
            'gt_labels': [labels.to(device).long() for labels in labels_list],
        }

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs,feats_origin,image_embedding,feats = model(images)
            # 表征就在这儿？
            loss_dict = criterion(outputs, targets, epoch)  
            losses = sum(loss for loss in loss_dict.values())
        
        # '''
        
        # # 一些可视化
        # # '''
        # if epoch%20 == 0 :
        #     save_path = "/data/hotaru/projects/PNS_tmp/prompter/checkpoint/cpm17/feature_map_save/"+save_middle_path_name+"/"+str(epoch)+"/"
        #     os.makedirs(save_path, exist_ok=True)
        #     to_pil = transforms.ToPILImage()

        #     feats_origin_average_feature_map = feats_origin[0][:, 0, :, :]
        #     feats_origin_images = [to_pil(feats_origin_average_feature_map[i].squeeze().cpu()) for i in range(8)]
        #     for i, feats_origin_image in enumerate(feats_origin_images):
        #         save_path1 = f"{data_iter_step}_{i}_feats_origin.png"
        #         feats_origin_image.save(save_path+save_path1)

        #     image_embedding_average_feature_map = torch.mean(image_embedding[0], dim=1, keepdim=True)
        #     image_embedding_images = [to_pil(image_embedding_average_feature_map[i].squeeze().cpu()) for i in range(8)]
        #     for i, image_embedding_image in enumerate(image_embedding_images):
        #         save_path1 = f"{data_iter_step}_{i}_image_embedding.png"
        #         image_embedding_image.save(save_path+save_path1)

        #     feats_average_feature_map = torch.mean(feats[0], dim=1, keepdim=True)
        #     feats_images = [to_pil(feats_average_feature_map[i].squeeze().cpu()) for i in range(8)]
        #     for i, feats_image in enumerate(feats_images):
        #         save_path1 = f"{data_iter_step}_{i}_feats.png"
        #         feats_image.save(save_path+save_path1)

           
        

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        for k, v in loss_dict_reduced.items():
            log_info[k] = log_info.get(k, 0) + v.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if args.clip_grad > 0:  # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        if model_ema and data_iter_step % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return log_info


@torch.inference_mode()
def evaluate(
        cfg,
        model,
        test_loader,
        device,
        epoch=0,
        calc_map=False,
        save_middle_path_name='tmp'
):
    model.eval()
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    cls_predictions = []
    cls_annotations = []

    cls_pn, cls_tn = list(torch.zeros(num_classes).to(device) for _ in range(2))
    cls_rn = torch.zeros(num_classes).to(device)

    det_pn, det_tn = list(torch.zeros(1).to(device) for _ in range(2))
    det_rn = torch.zeros(1).to(device)

    iou_scores = []

    epoch_iterator = tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
                          dynamic_ncols=True, disable=not is_main_process())

    for data_iter_step, (images, gt_points, labels, masks, ori_shape) in enumerate(epoch_iterator):
        assert len(images) == 1, 'batch size must be 1'

        if data_iter_step % get_world_size() != get_rank():  # To avoid duplicate evaluation for some test samples
            continue

        epoch_iterator.set_description(
            "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))

        images = images.to(device)

        pd_points, pd_scores, pd_classes, pd_masks = predict(
            model,
            images,
            ori_shape=ori_shape[0].numpy(),
            filtering=cfg.test.filtering,
            nms_thr=cfg.test.nms_thr,
        )

        if pd_masks is not None:
            masks = masks[0].numpy()
            intersection = (pd_masks * masks).sum()
            union = (pd_masks.sum() + masks.sum() + 1e-7) - intersection
            iou_scores.append(intersection / (union + 1e-7))

        gt_points = gt_points[0].reshape(-1, 2).numpy()
        labels = labels[0].numpy()

        cls_annotations.append({'points': gt_points, 'labels': labels})

        cls_pred_sample = []
        for c in range(cfg.data.num_classes):
            ind = (pd_classes == c)
            category_pd_points = pd_points[ind]
            category_pd_scores = pd_scores[ind]
            category_gt_points = gt_points[labels == c]

            cls_pred_sample.append(np.concatenate([category_pd_points, category_pd_scores[:, None]], axis=-1))

            pred_num, gd_num = len(category_pd_points), len(category_gt_points)
            cls_pn[c] += pred_num
            cls_tn[c] += gd_num

            if pred_num and gd_num:
                cls_right_nums,_ = get_tp(category_pd_points, category_pd_scores, category_gt_points, thr=cfg.test.match_dis)
                cls_rn[c] += torch.tensor(cls_right_nums, device=cls_rn.device)

        cls_predictions.append(cls_pred_sample)

        det_pn += len(pd_points)
        det_tn += len(gt_points)

        if len(pd_points) and len(gt_points):
            det_right_nums, unmatched_points = get_tp(pd_points, pd_scores, gt_points, thr=cfg.test.match_dis)
            det_rn += torch.tensor(det_right_nums, device=det_rn.device)
        # 现在想可视化的内容：
        # 所有点：pd_points，里面不匹配的点：unmatched_points为1的索引gt没有被匹配，gt点：gt_points
        # mask:当前轮预测的mask：pd_masks
        # epoch√ images gt_mask上画点  masks(512,512) vs pd_masks (512,512)上画点 vs gt_mask和pd_mask画一起
        if  data_iter_step%8 == 0:
            save_point_mask_path = '/data/hotaru/projects/PNS_tmp/prompter/outputAndOtherfile/middle_save/'+save_middle_path_name+'/'+str(epoch)+"/"
            os.makedirs(save_point_mask_path,exist_ok=True)
            draw_points_on_image(masks, pd_points, gt_points, unmatched_points ,save_path=save_point_mask_path+str(data_iter_step)+"_gtM.png")
            draw_points_on_image(np.squeeze(images).permute(1, 2, 0).cpu(), pd_points,   gt_points, unmatched_points,save_path=save_point_mask_path+str(data_iter_step)+"_img.png")
            draw_points_on_image(pd_masks, pd_points,  gt_points, unmatched_points,save_path=save_point_mask_path+str(data_iter_step)+"_pdM.png")
            overlay_masks(np.squeeze(images).permute(1, 2, 0).cpu(), masks, pd_masks,alpha=0.5,save_path=save_point_mask_path+str(data_iter_step)+"_img&M.png" )
       

    if get_world_size() > 1:
        dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

        dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        cls_predictions = list(itertools.chain.from_iterable(all_gather(cls_predictions)))
        cls_annotations = list(itertools.chain.from_iterable(all_gather(cls_annotations)))

        iou_scores = np.concatenate(all_gather(iou_scores))

    eps = 1e-7
    det_r = det_rn / (det_tn + eps)
    det_p = det_rn / (det_pn + eps)
    det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

    det_r = det_r.cpu().numpy() * 100
    det_p = det_p.cpu().numpy() * 100
    det_f1 = det_f1.cpu().numpy() * 100

    cls_r = cls_rn / (cls_tn + eps)
    cls_p = cls_rn / (cls_pn + eps)
    cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

    cls_r = cls_r.cpu().numpy() * 100
    cls_p = cls_p.cpu().numpy() * 100
    cls_f1 = cls_f1.cpu().numpy() * 100

    table = pt.PrettyTable()
    table.add_column('CLASS', class_names)

    table.add_column('Precision', cls_p.round(2))
    table.add_column('Recall', cls_r.round(2))
    table.add_column('F1', cls_f1.round(2))

    table.add_row(['---'] * 4)

    det_p, det_r, det_f1 = det_p.round(2)[0], det_r.round(2)[0], det_f1.round(2)[0]
    cls_p, cls_r, cls_f1 = cls_p.mean().round(2), cls_r.mean().round(2), cls_f1.mean().round(2)

    table.add_row(['Det', det_p, det_r, det_f1])
    table.add_row(['Cls', cls_p, cls_r, cls_f1])
    print(table)
    if calc_map:
        mAP = eval_map(cls_predictions, cls_annotations, cfg.test.match_dis)[0]
        print(f'mAP: {round(mAP * 100, 2)}')

    metrics = {'Det': [det_p, det_r, det_f1], 'Cls': [cls_p, cls_r, cls_f1],
               'IoU': (np.mean(iou_scores) * 100).round(2)}
    return metrics, table.get_string()

def draw_points_on_image(image, points,gt_points,unmatch_bool,save_path=None):
    if len(unmatch_bool) != len(gt_points):
        print("维度不匹配！")
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        return 
    # 提取坐标
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    plt.figure()
    plt.imshow(image, cmap='gray')  # 显示图像
    
    unmatch_points = gt_points[unmatch_bool == 1]
    match_points = gt_points[unmatch_bool == 0]
    gt_unmatch_x=unmatch_points[:,0]
    gt_unmatch_y=unmatch_points[:,1]

    gt_match_x = match_points[:,0]
    gt_match_y = match_points[:,1]

    # 绘制点到图像上
    # plt.figure()
    # plt.imshow(image, cmap='gray')  # 显示图像
    plt.scatter(x_coords, y_coords, marker='*', color='blue')  # 绘制点
    plt.scatter(gt_unmatch_x, gt_unmatch_y, marker='x', color='r')  # 绘制gt中没有匹配到的点
    plt.scatter(gt_match_x, gt_match_y, marker='o', color='green')  # 绘制gt中匹配到的点
    plt.axis('off')  # 关闭坐标轴显示

    # 如果指定了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return plt.gcf()

def overlay_masks(image, gt_mask, pred_mask, alpha,save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制原始图像
    ax.imshow(image)

    # 将 GT mask 叠加在原始图像上
    ax.imshow(gt_mask, cmap='jet', alpha=alpha)

    # 将预测 mask 叠加在原始图像上
    ax.imshow(pred_mask, cmap='jet', alpha=alpha)
    ax.axis('off')  # 关闭坐标轴显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        return plt.gcf()
    



def save_four_images_as_grid(image1, image2, image3, image4, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    axes = axes.flatten()  # 将二维数组展平成一维数组

    canvas = image1.canvas
    # 将 Canvas 对象转换为图像数据（RGB像素数组）
    canvas.draw()
    width, height = image1.get_size_inches() * image1.get_dpi()
    image1 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    # 绘制第一张图像
    axes[0].imshow(image1)
    axes[0].axis('off')

    canvas = image2.canvas
    # 将 Canvas 对象转换为图像数据（RGB像素数组）
    canvas.draw()
    width, height = image2.get_size_inches() * image2.get_dpi()
    image2 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)    
    # 绘制第二张图像
    axes[1].imshow(image2)
    axes[1].axis('off')

    canvas = image3.canvas
    # 将 Canvas 对象转换为图像数据（RGB像素数组）
    canvas.draw()
    width, height = image3.get_size_inches() * image3.get_dpi()
    image3 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)    
    # 绘制第三张图像
    axes[2].imshow(image3)
    axes[2].axis('off')

    canvas = image4.canvas
    # 将 Canvas 对象转换为图像数据（RGB像素数组）
    canvas.draw()
    width, height = image4.get_size_inches() * image4.get_dpi()
    image4 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)    
    # 绘制第四张图像
    axes[3].imshow(image4)
    axes[3].axis('off')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)

    # 保存图像
    plt.savefig(save_path)


