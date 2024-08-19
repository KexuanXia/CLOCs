import pickle
import numpy as np
import torch
from google.protobuf import text_format

import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.models import fusion
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import open3d as o3d
from scipy.spatial.transform import Rotation

from train import (build_inference_net,
                   example_convert_to_torch,
                   get_inference_input_dict,
                   predict_kitti_to_anno)


def occam_evaluation_save_dt(start_idx, end_idx, it_nr=3000, save_result=False,
                             config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
                             second_model_dir='../model_dir/second_model',
                             fusion_model_dir='../CLOCs_SecCas_pretrained'):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    detection_2d_path = config.train_config.detection_2d_path
    center_limit_range = model_cfg.post_center_limit_range
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    class_names = target_assigner.classes
    net = build_inference_net(config_path, second_model_dir)
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    net.cuda()
    torchplus.train.try_restore_latest_checkpoints(fusion_model_dir, [fusion_layer])
    net.eval()
    fusion_layer.eval()
    for idx in range(start_idx, end_idx):
        idx_str = str(idx).zfill(6)
        input_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/{idx_str}.bin'
        i_path = f'/home/xkx/kitti/training/image_2/{idx_str}.png'
        attr_map_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/heat_map/{idx_str}_{it_nr}.pkl'

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)

        base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results(idx_str)
        with open(attr_map_path, 'rb') as file:
            attr_map = pickle.load(file)
        print(f"attr_map.shape: {attr_map.shape}")
        print(f"base_det_boxes: {base_det_boxes}")

        results = []
        if attr_map.shape[0] == 0:
            print("This scene doesn't contain any target.")
        else:
            sorted_points_desc, sorted_points_random, sorted_points_asc = (
                filter_and_sort_points_by_importance_V2(input_pc, base_det_boxes, attr_map))
            removal_results_desc = progressively_remove_points(input_pc, sorted_points_desc)
            removal_results_random = progressively_remove_points(input_pc, sorted_points_random)
            removal_results_asc = progressively_remove_points(input_pc, sorted_points_asc)
            removal_results = [removal_results_desc, removal_results_random, removal_results_asc]

            # visualize_point_cloud_and_bboxes(removal_results[0][100], base_det_boxes)
            for removal_result in removal_results:
                result = []
                for percentage, remaining_points in removal_result.items():

                    example = get_inference_input_dict(config=config,
                                                       voxel_generator=voxel_generator,
                                                       target_assigner=target_assigner,
                                                       info=info,
                                                       points=remaining_points,
                                                       i_path=i_path)
                    example = example_convert_to_torch(example, torch.float32)

                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input)
                        prediction_dicts = prediction_dicts[0]
                        result.append(prediction_dicts)
                results.append(result)
        # print(f"results: {results}")
        if save_result:
            save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_dropped_dt_results_three_order/{idx_str}.pkl'
            with open(save_path, 'wb') as output_file:
                pickle.dump(results, output_file)
            print(f"detection result of dropped {idx_str}.bin have been saved")


def visualize_point_cloud_and_bboxes(input_pc, base_det_boxes):
    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(input_pc[:, :3])

    # 定义用于存储几何体的列表
    geometries = [point_cloud]

    # 遍历检测框并创建立方体
    for box in base_det_boxes:
        # 创建Open3D的OrientedBoundingBox
        rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
        bb = o3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        bb.color = (1.0, 0.0, 1.0)

        # 将立方体添加到几何体列表
        geometries.append(bb)

    # Create a visualizer and set the background color
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Set the background color
    vis.get_render_option().background_color = np.ones(3) * 0.25
    vis.get_render_option().point_size = 4.0

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def filter_and_sort_points_by_importance_V2(input_pc, base_det_boxes, attr_map):
    """
    筛选出位于旋转检测框内的点，并根据重要度排序。同时返回三组结果：
    1. 按重要度降序排序的点及其索引。
    2. 随机排序的点及其索引。
    3. 按重要度升序排序的点及其索引。

    :param input_pc: numpy array, 形状为 (M, 3)，表示 M 个三维点 (x, y, z)。
    :param base_det_boxes: numpy array, 形状为 (N, 7)，表示 N 个检测框，
                           每个框由中心点 (cx, cy, cz)、尺寸 (length, width, height) 和旋转角度 (yaw) 定义。
    :param attr_map: numpy array, 形状为 (N, M)，表示每个检测框内每个点的重要度。
    :return: tuple of three lists，每个列表包含三种不同排序的结果，分别为按降序、随机和升序排序。
    """
    sorted_points_desc = []
    sorted_points_random = []
    sorted_points_asc = []

    for i, box in enumerate(base_det_boxes):
        cx, cy, cz, length, width, height, yaw = box

        # 计算旋转矩阵
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,       0,       1]
        ])

        # 将点云移动到以检测框中心为原点的坐标系中
        translated_points = input_pc[:, :3] - np.array([cx, cy, cz])

        # 应用旋转矩阵将点云转换到检测框坐标系
        rotated_points = np.dot(translated_points, rotation_matrix)

        # 计算检测框的边界
        x_min, x_max = -length / 2, length / 2
        y_min, y_max = -width / 2, width / 2
        z_min, z_max = -height / 2, height / 2

        # 找到位于该检测框内的点
        inside_mask = (
            (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
            (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
            (rotated_points[:, 2] >= z_min) & (rotated_points[:, 2] <= z_max)
        )

        print(f"number of points in boxes: {sum(inside_mask)}")

        # 获取筛选出来的点的索引
        point_indices = np.where(inside_mask)[0]

        # 提取位于检测框内的点和其对应的重要度
        points_in_box = input_pc[inside_mask]
        importance_in_box = attr_map[i, inside_mask]

        # 按重要度降序排序
        sorted_indices_desc = np.argsort(importance_in_box)[::-1]
        sorted_points_desc.append(
            list(zip(points_in_box[sorted_indices_desc], importance_in_box[sorted_indices_desc]))
        )

        # 随机排序
        random_indices = np.random.permutation(len(importance_in_box))
        sorted_points_random.append(
            list(zip(points_in_box[random_indices], importance_in_box[random_indices]))
        )

        # 按重要度升序排序
        sorted_indices_asc = np.argsort(importance_in_box)
        sorted_points_asc.append(
            list(zip(points_in_box[sorted_indices_asc], importance_in_box[sorted_indices_asc]))
        )

    return sorted_points_desc, sorted_points_random, sorted_points_asc


def progressively_remove_points(input_pc, sorted_points_with_importance):
    """
    逐步移除重要度排在前0%、10%...到100%的点。

    :param input_pc: numpy array, 形状为 (M, 3)，表示 M 个三维点 (x, y, z)。
    :param sorted_points_with_importance: list of lists，每个列表包含位于对应检测框内的点及其重要度，并按重要度降序排序。
    :return: dict, key为移除的百分比，value为剩余点的数组。
    """
    removal_results = {}

    # 计算需要移除的数量
    for percent in range(0, 110, 10):
        remaining_points = input_pc.copy()

        # 从每个检测框开始移除
        for box_index, points_with_importance in enumerate(sorted_points_with_importance):
            points_in_box_count = len(points_with_importance)
            # if there is no point in the bbox, there is no need to remove any points
            if points_in_box_count != 0:
                points_to_remove = int(points_in_box_count * (percent / 100))

                # print(f"box_index: {box_index}")
                # print(f"points with...: {points_with_importance}")
                # print(points_in_box_count)

                sorted_points, _ = zip(*points_with_importance)
                sorted_points = np.array(sorted_points)
                mask = np.ones(len(remaining_points), dtype=bool)

                for point in sorted_points[:points_to_remove]:
                    point_indices = np.where((remaining_points == point).all(axis=1))[0]
                    if point_indices.size > 0:
                        mask[point_indices[0]] = False

                remaining_points = remaining_points[mask]

        removal_results[percent] = remaining_points

    return removal_results


def read_kitti_info_val(idx):
    file_path = "/home/xkx/kitti/kitti_infos_trainval.pkl"
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    for item in data:
        if item.get('image_idx') == idx:
            return item
    return IndexError


def read_original_dt_results(idx_str):
    #read_path = f'/home/xkx/kitti/training/velodyne_masked_dt_results/{source_file_path[-10: -4]}_original.pkl'
    read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/{idx_str}_original.pkl'
    with open(read_path, 'rb') as file:
        data = pickle.load(file)
    pred_boxes = data["box3d_lidar"]
    pred_boxes[:, [3, 4]] = pred_boxes[:, [4, 3]]
    pred_scores = data["scores"]
    pred_labels = data["label_preds"] + 1

    pred_boxes = pred_boxes.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    for i in range(pred_boxes.shape[0]):
        pred_boxes[i, 6] = -pred_boxes[i, 6] - np.pi / 2
        pred_boxes[i, 2] = pred_boxes[i, 2] + pred_boxes[i, 5] / 2

    print(f"Successfully read original detection results from {read_path}")

    return pred_boxes, pred_labels, pred_scores


if __name__ == '__main__':
    occam_evaluation_save_dt(167, 501, save_result=True)
