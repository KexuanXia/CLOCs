import pickle

def read_and_print_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(data)
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
file_path = '/home/xkx/kitti/training/velodyne_masked_dt_results/000002_original.pkl'
read_and_print_pkl(file_path)
file_path = '/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/000002_original.pkl'
read_and_print_pkl(file_path)

