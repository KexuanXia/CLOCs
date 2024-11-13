"""
    This file aims to calculate how the importance score distributed within/outside bounding boxes
    But the results were bad so didn't show it in thesis or presentation.
"""


import numpy as np
import pickle
import cv2


def read_heatmap(idx):
    # choose which heat map to be read
    # random combination heat map
    # read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/d_rise_heat_map_data/'
    #              f'{str(idx).zfill(6)}_3000.pkl')
    # The heat map obtained by only masking images, not mask lidar points
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_heat_map_data_new/'
                 f'{str(idx).zfill(6)}_3000.pkl')
    with open(read_path, 'rb') as file:
        heat_map = pickle.load(file)
    # print(f"heatmap.shape: {heat_map.shape}")
    # print(f"heatmap: {heat_map}")
    return heat_map


def read_original_dt_results_2d(idx):
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results_original/'
                 f'{str(idx).zfill(6)}.pkl')
    with open(read_path, 'rb') as file:
        data = pickle.load(file)
    pred_boxes = data["bbox"]
    pred_scores = data["scores"]
    pred_labels = data["label_preds"] + 1

    pred_boxes = pred_boxes.cpu().numpy().astype(int)
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    return pred_boxes, pred_labels, pred_scores


def min_max_normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def norm_heat_map(heat_map):
    a, b, c = heat_map.shape
    normalized_heatmap = np.zeros((a, b, c))
    for i in range(a):
        normalized_heatmap[i] = min_max_normalize(heat_map[i])
    # print(f"heatmap.shape: {normalized_heatmap.shape}")
    # print(f"heatmap: {normalized_heatmap}")
    return normalized_heatmap


# calculate mean heat map value within, between and outside the bounding boxes in one scene
def calculate_heatmap_means(heatmap, pred_boxes, scale=1.5):
    num_objects, image_h, image_w = heatmap.shape
    scaled_boxes = scale_boxes(pred_boxes, scale)

    # 存储每个物体框内和框外的平均值
    inside_means = []
    between_means = []
    outside_means = []

    for i in range(num_objects):
        # 获取当前的 heatmap 和检测框
        current_heatmap = heatmap[i]
        x1, y1, x2, y2 = adjust_pred_boxes(pred_boxes[i], image_w, image_h)
        x1_, y1_, x2_, y2_ = adjust_pred_boxes(scaled_boxes[i], image_w, image_h)

        # 提取框内的像素
        inside_pixels = current_heatmap[y1:y2 + 1, x1:x2 + 1]
        inside_mean = np.mean(inside_pixels)

        # 计算两框之间的区域
        mask_scaled_box = np.zeros(current_heatmap.shape, dtype=bool)
        mask_scaled_box[y1_:y2_ + 1, x1_:x2_ + 1] = True  # 标记放大框内区域
        mask_pred_box = np.zeros(current_heatmap.shape, dtype=bool)
        mask_pred_box[y1:y2 + 1, x1:x2 + 1] = True  # 标记原框内区域

        # 取出只在放大框内，但不在原框内的像素
        mask_between = mask_scaled_box & ~mask_pred_box
        heatmap_between_boxes = current_heatmap[mask_between]
        between_mean = np.mean(heatmap_between_boxes)

        # 计算框外像素的平均值
        # 通过掩码操作，将框内区域屏蔽
        mask = np.ones_like(current_heatmap, dtype=bool)
        mask[y1:y2 + 1, x1:x2 + 1] = False
        outside_pixels = current_heatmap[mask]
        outside_mean = np.mean(outside_pixels)

        # 保存结果
        inside_means.append(inside_mean)
        between_means.append(between_mean)
        outside_means.append(outside_mean)

    return inside_means, between_means, outside_means


# average the mean values obtained by last function in multiple scenes
def mean_multiple_scenes(start_idx, end_idx, scale=1.5):
    inside_mean_list, betweem_mean_list, outside_mean_list = [], [], []
    for idx in range(start_idx, end_idx):
        heat_map = read_heatmap(idx)
        if heat_map.shape[0] == 0:
            continue
        normed_heatmap = norm_heat_map(heat_map)
        pred_boxes, _, _ = read_original_dt_results_2d(idx)
        inside_means, betweem_means, outside_means = calculate_heatmap_means(normed_heatmap, pred_boxes, scale)
        inside, between, outside = np.mean(inside_means), np.mean(betweem_means), np.mean(outside_means)
        inside_mean_list.append(inside)
        betweem_mean_list.append(between)
        outside_mean_list.append(outside)
    print(inside_mean_list)
    print(betweem_mean_list)
    print(outside_mean_list)
    print(np.mean(inside_mean_list))
    print(np.mean(betweem_mean_list))
    print(np.mean(outside_mean_list))


def scale_boxes(pred_boxes, scale=1.5):
    # 创建一个存放放大后的检测框的数组
    scaled_boxes = np.zeros_like(pred_boxes)

    for i in range(pred_boxes.shape[0]):
        # 获取当前检测框的左上角和右下角的坐标
        x1, y1, x2, y2 = pred_boxes[i]

        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算宽度和高度
        width = x2 - x1
        height = y2 - y1

        # 计算放大后的宽度和高度
        new_width = width * scale
        new_height = height * scale

        # 计算新的左上角和右下角坐标
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        # 更新放大的检测框
        scaled_boxes[i] = [new_x1, new_y1, new_x2, new_y2]

    return scaled_boxes


def adjust_pred_boxes(pred_boxes, image_w, image_h):
    x1, y1, x2, y2 = pred_boxes

    # 限制框的坐标在图像范围内
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w - 1))
    y2 = max(0, min(y2, image_h - 1))

    return x1, y1, x2, y2


if __name__ == '__main__':
    # read_heatmap 可以调整读取的heat map的来源，来自单模态遮掩或者random combination
    mean_multiple_scenes(0, 10, scale=1.5)
