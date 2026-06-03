import h5py
import numpy as np

def explore_hdf5(file_path):
    def print_structure(name, obj):
        # 打印层级结构
        indent = '  ' * name.count('/')
        
        if isinstance(obj, h5py.Dataset):
            # 如果是数据集，打印形状和前几个数值
            print(f"{indent}📊 数据集: {name} | 形状: {obj.shape} | 类型: {obj.dtype}")
            # 读取前5个数据看看（防止数据太大卡死）
            data = obj[()] 
            # if data.size > 0:
                # print(f"{indent}   └─ 示例值 (前5个): {data.flatten()[:5]}")
        elif isinstance(obj, h5py.Group):
            # 如果是组（文件夹）
            print(f"{indent}📁 组: {name}")

    print(f"--- 正在读取文件: {file_path} ---")
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)


explore_hdf5("./datasets/simdata/V1/annotated_visuo_dataset.hdf5")
