import pickle


def read_and_save_top_n_pkl(input_file_path, output_file_path, n):
    try:
        # 读取pkl文件
        with open(input_file_path, 'rb') as file:
            data = pickle.load(file)

            # 检查data是否为列表或其他可切片的数据结构
            if isinstance(data, (list, tuple)):
                top_n_data = [data[1]]
            else:
                raise ValueError("数据不是列表或元组，无法切片")

            # 将前n个数据保存到另一个pkl文件
            with open(output_file_path, 'wb') as output_file:
                pickle.dump(top_n_data, output_file)

            print(f"前{n}个数据已保存到 {output_file_path}")
    except Exception as e:
        print(f"读取或保存文件时发生错误: {e}")


def main():
    n = 5
    input_file_path = '/home/xkx/kitti/kitti_infos_val.pkl'  # 输入pkl文件路径
    output_file_path = f'/home/xkx/kitti/kitti_infos_val_000002.pkl'  # 输出pkl文件路径
    read_and_save_top_n_pkl(input_file_path, output_file_path, n)


if __name__ == "__main__":
    main()
