import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.io import loadmat
import pandas as pd
import cv2
import os
import argparse
import csv
def parse_args():
    parser = argparse.ArgumentParser('prompt_eval')
    parser.add_argument('--dataset', default='', help='config file')
    parser.add_argument('--prompt_path', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--csv_path', default='', type=str, help='resume from checkpoint')

    opt = parser.parse_args()

    return opt


#return inst_map,type_map,points,points_labels
def load_dataset(dataset,file_name):
    if dataset == 'pannuke':
        point_data = np.load("./prompts/pannuke321/"+file_name+".npy")
        mask_data = np.load("./datasets/pannuke/Masks/"+file_name+".npy", allow_pickle=True).item()  # 加载npy文件并将字典转换为Python对象
        inst_map = mask_data['inst_map']
        type_map = mask_data['type_map']
    elif dataset == 'cpm17':
        point_data = np.load("/data/hotaru/my_projects/PNS_tmp/segmentor/prompts/cpm17/"+file_name+".npy")
        mask_data = loadmat("/data/hotaru/my_projects/PNS_tmp/segmentor/datasets/cpm17/test/Labels/"+file_name+".mat")  # 加载npy文件并将字典转换为Python对象
        inst_map = mask_data['inst_map']
        type_map = np.ones_like(inst_map)
    else:
        print("暂不支持其他数据集")
        return
    # 提取坐标和类别信息
    points = point_data[:, :2]
    points_labels = point_data[:, 2]
    return inst_map,type_map,points,points_labels

#return [num_gt_instances,num_points,true_counts,m_back_count,m_front_count,l_single_count,l_multi_count,cls_false_count,all_false_num],temp_inst_map,semantic_liantong_map
def count_false_points(inst_map,type_map,points,points_labels):
    #初始化所需变量
    temp_inst_map = inst_map.copy()
    temp_inst_map = temp_inst_map.astype(np.int8)  # 转换为 CV_8U
    _ , binary_semantic_map = cv2.connectedComponents(temp_inst_map,connectivity=4) 
    binary_semantic_map = binary_semantic_map.astype(np.int8)

    num_gt_instances = 0 # 0. gt总数
    num_points = 0       # 1. points总数
    true_counts = 0      # 2. 正确总数
    m_back_count = 0     # 3. 背景多检
    m_front_count = 0    # 4. 前景多检
    l_single_count = 0   # 5. 单独漏检
    l_multi_count = 0    # 6. 粘连漏检
    cls_false_count = 0  # 7. 类别错误
    all_false_num = 0    # 8. 错误总数


    num_gt_instances = len(np.unique(inst_map)) - 1  # 0. 减去背景标签
    num_points = len(points)                          # 1.
    '''
        开始计算
        * 2. 正确总数：前景非零&非-1&类别正确/没有类别,更新temp_inst_map&semantic_binary_map
        * 4. 前景多检：temp_inst_map = -1
        * 5. 单独漏检：剩余语义图连通图数目
        * 6. 粘连漏检：剩余实例图数目 - 剩余语义图数目
        * 7. 类别错误：前景非零&非-1&类别错误,更新temp_inst_map&semantic_binary_map
    '''
    
    for point , point_label in zip(points,points_labels):
        x,y = int(point[1]),int(point[0])
        label = point_label
        point_is_correct = False # flag
        if temp_inst_map[x, y] != 0 and  temp_inst_map[x, y] != -1 and label == type_map[x,y]-1 :  # 2.
            true_counts += 1
            point_is_correct = True        
            # 删掉已经定位的inst
            liantong_num, liantong_map = cv2.connectedComponents(binary_semantic_map,connectivity=4) 
            temp_inst_id = temp_inst_map[x, y]  
            temp_liantong_id = liantong_map[x,y]
            for i in range(temp_inst_map.shape[0]):  
                for j in range(temp_inst_map.shape[1]):  
                    if temp_inst_map[i, j] == temp_inst_id:  
                        temp_inst_map[i, j] = -1  
                    if liantong_map[i,j] == temp_liantong_id:
                        binary_semantic_map[i, j] = 0

        if point_is_correct == False and inst_map[x, y] == 0:   #3.
            m_back_count += 1
        
        if point_is_correct == False and temp_inst_map[x, y] == -1 :   #4.
            m_front_count += 1
        if point_is_correct == False and temp_inst_map[x, y] != 0 and  temp_inst_map[x, y] != -1 and label != type_map[x,y]-1:
            cls_false_count += 1
            # 删掉已经定位的inst
            liantong_num, liantong_map = cv2.connectedComponents(binary_semantic_map,connectivity=4) 
            temp_inst_id = temp_inst_map[x, y]  
            temp_liantong_id = liantong_map[x,y]
            for i in range(temp_inst_map.shape[0]):  
                for j in range(temp_inst_map.shape[1]):  
                    if temp_inst_map[i, j] == temp_inst_id:  
                        temp_inst_map[i, j] = -1  
                    if liantong_map[i,j] == temp_liantong_id:
                        binary_semantic_map[i, j] = 0

    semantic_liantong_num, semantic_liantong_map = cv2.connectedComponents(binary_semantic_map) 
    l_single_count = semantic_liantong_num -1   # 5
    l_multi_count = num_gt_instances - true_counts - cls_false_count - l_single_count  # 6
    if l_multi_count < 0 : l_multi_count = 0

    all_false_num = m_back_count + m_front_count + l_single_count + l_multi_count + cls_false_count
    # print("0. gt总数",num_gt_instances)
    # print("1. points总数",num_points)
    # print("2. 正确总数",true_counts)
    # print("3. 背景多检",m_back_count)
    # print("4. 前景多检",m_front_count)
    # print("5. 单独漏检",l_single_count)
    # print("6. 粘连漏检",l_multi_count)
    # print("7. 类别错误",cls_false_count)
    # print("8. 错误总数",all_false_num)

    return [num_gt_instances,num_points,true_counts,m_back_count,m_front_count,l_single_count,l_multi_count,cls_false_count,all_false_num],temp_inst_map,semantic_liantong_map



def main():
    args = parse_args()
    print(args)
    dataset = args.dataset
    folder_path = args.prompt_path
    csv_file = args.csv_path
    file_names = os.listdir(folder_path)
    field_names = ["file_name", "num_gt_instances", "num_points", "true_counts", "m_back_count", "m_front_count", "l_single_count", "l_multi_count", "cls_false_count", "all_false_num"]
    count = 0
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(field_names)
        for file_name in file_names:
            file_name = file_name.split('.')[0]
            inst_map,type_map,points,points_labels = load_dataset(dataset, file_name)
            false_points,temp_inst_map, semantic_liantong_map= count_false_points(inst_map,type_map,points,points_labels)
            row = [file_name] + false_points
            writer.writerow(row)
            count+=1
            print(count , row)

    df = pd.read_csv(csv_file)

    # 计算除第一列之外的其他列的和
    data = df.sum(axis=0)[1:]  # axis=0表示按列求和，[1:]表示从第二列开始求和，因为第一列是文件名


    # 打印结果
    print("正确总数：",data[2])
    print("准确率：",data[2]/data[1])
    print("召回率：",data[2]/data[0])
    print("背景多检：",data[3])
    print("前景多检：",data[4])
    print("单独漏检：",data[5])
    print("粘连漏检：",data[6])
    print("类别错误：",data[7])
    print("错误总数：",data[8])     

      
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    main()


