"""
    This file aims to calculate how the importance score distributed within/outside bounding boxes
    But the results were bad so didn't show it in thesis or presentation.
"""


import numpy as np
import pickle


def read_heatmap(idx):
    # choose which heat map to be read
    # random combination heat map
    # read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/occam_heat_map_data/'
    #              f'{str(idx).zfill(6)}_3000.pkl')

    # The heat map obtained by only masking lidar, not mask image
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/heat_map/'
                 f'{str(idx).zfill(6)}_3000.pkl')
    with open(read_path, 'rb') as file:
        heat_map = pickle.load(file)
    # print(f"heatmap.shape: {heat_map.shape}")
    # print(f"heatmap: {heat_map}")
    return heat_map


def read_original_dt_results_3d(idx):
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/'
                 f'{str(idx).zfill(6)}_original.pkl')
    with open(read_path, 'rb') as file:
        data = pickle.load(file)
    pred_boxes = data["box3d_lidar"]
    pred_scores = data["scores"]
    pred_labels = data["label_preds"] + 1

    pred_boxes = pred_boxes.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    for i in range(pred_boxes.shape[0]):
        pred_boxes[i, 6] = -pred_boxes[i, 6] - np.pi / 2
        pred_boxes[i, 2] = pred_boxes[i, 2] + pred_boxes[i, 5] / 2

    return pred_boxes, pred_labels, pred_scores


def read_input_pc(idx):
    input_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/'
                  f'{str(idx).zfill(6)}.bin')
    input_pc = np.fromfile(input_path, dtype=np.float32)
    input_pc = input_pc.reshape(-1, 4)
    return input_pc


def min_max_normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def norm_heat_map(heat_map):
    num_objects, num_points = heat_map.shape
    normalized_heatmap = np.zeros((num_objects, num_points))
    for i in range(num_objects):
        current_heatmap = heat_map[i]
        all_zeros = np.all(current_heatmap == 0)
        if all_zeros:
            continue
        normalized_heatmap[i] = min_max_normalize(current_heatmap)
    return normalized_heatmap


# calculate mean heat map value within, between and outside the bounding boxes in one scene
def calculate_heatmap_means(input_pc, heatmap, pred_boxes, scale=1.5):
    # 存储每个物体框内和框外的平均值
    inside_means = []
    between_means = []
    outside_means = []

    for i, box in enumerate(pred_boxes):
        # 获取当前的 heatmap 和检测框
        current_heatmap = heatmap[i]
        all_zeros = np.all(current_heatmap == 0)
        if all_zeros:
            continue

        cx, cy, cz, length, width, height, yaw = box
        larged_length, larged_width, larged_height = length * scale, width * scale, height * scale

        # 计算旋转矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # 将点云移动到以检测框中心为原点的坐标系中
        translated_points = input_pc[:, :3] - np.array([cx, cy, cz])

        # 应用旋转矩阵将点云转换到检测框坐标系
        rotated_points = np.dot(translated_points, rotation_matrix.T)

        # 计算检测框的边界
        x_min, x_max = -length / 2, length / 2
        y_min, y_max = -width / 2, width / 2
        z_min, z_max = -height / 2, height / 2

        x_min_, x_max_ = -larged_length / 2, larged_length / 2
        y_min_, y_max_ = -larged_width / 2, larged_width / 2
        z_min_, z_max_ = -larged_width/ 2, larged_width / 2

        # 找到位于该检测框内的点
        inside_mask = (
                (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
                (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
                (rotated_points[:, 2] >= z_min) & (rotated_points[:, 2] <= z_max)
        )

        inside_mask_bigger_box = (
                (rotated_points[:, 0] >= x_min_) & (rotated_points[:, 0] <= x_max_) &
                (rotated_points[:, 1] >= y_min_) & (rotated_points[:, 1] <= y_max_) &
                (rotated_points[:, 2] >= z_min_) & (rotated_points[:, 2] <= z_max_)
        )

        # 提取位于检测框内的点和其对应的重要度
        heatmap_in_box = current_heatmap[inside_mask]
        heatmap_out_box = current_heatmap[~inside_mask]
        heatmap_between_boxes = current_heatmap[np.logical_and(~inside_mask, inside_mask_bigger_box)]

        if heatmap_in_box.shape[0] == 0:
            # print("None points in box")
            inside_mean = 0
        else:
            inside_mean = np.mean(heatmap_in_box)
        if heatmap_between_boxes.shape[0] == 0:
            # print("None points between boxes")
            between_mean = 0
        else:
            between_mean = np.mean(heatmap_between_boxes)
        outside_mean = np.mean(heatmap_out_box)

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
        input_pc = read_input_pc(idx)
        if heat_map.shape[0] == 0 or np.all(heat_map == 0):
            continue

        normed_heatmap = norm_heat_map(heat_map)
        pred_boxes, _, _ = read_original_dt_results_3d(idx)
        inside_means, between_means, outside_means = calculate_heatmap_means(input_pc, normed_heatmap, pred_boxes, scale=1.5)
        inside, between, outside = np.mean(inside_means), np.mean(between_means), np.mean(outside_means)
        inside_mean_list.append(inside)
        betweem_mean_list.append(between)
        outside_mean_list.append(outside)
    # print(inside_mean_list)
    # print(betweem_mean_list)
    # print(outside_mean_list)
    print(np.mean(inside_mean_list))
    print(np.mean(betweem_mean_list))
    print(np.mean(outside_mean_list))


if __name__ == '__main__':
    # read_heatmap 可以调整读取的heat map的来源，来自单模态遮掩或者random combination
    mean_multiple_scenes(0, 300, scale=1.5)
