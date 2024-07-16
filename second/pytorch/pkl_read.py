import pickle

def read_and_print_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)[0]
            print(data)
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
file_path = '/home/xkx/kitti/kitti_infos_val.pkl'  # 请将 'your_file.pkl' 替换为你的 .pkl 文件路径
read_and_print_pkl(file_path)
print(1)
