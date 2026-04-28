import numpy as np
import matplotlib.pyplot as plt
import glob

# 1. 文件名
file1 = 'record_model.txt'
file2 = 'record_mujoco.txt'

# 2. 读取并跳过首行
def load(fname):
    with open(fname) as f:
        next(f)                 # 跳过首行
        return np.loadtxt(f, delimiter=',')

data1 = load(file1)             # shape (N, 6)
data2 = load(file2)             # shape (M, 6)

# 3. 绘图
cols = data1.shape[1]
fig, axes = plt.subplots(nrows=cols, ncols=1, figsize=(8, 1.3*cols), sharex=True)
if cols == 1:                   # 只有 1 列时 plt.subplots 返回一维
    axes = [axes]

lab = ['q0', 'q1', 'q2', 'dq0', 'dq1', 'dq2']
for i in range(cols):
    ax = axes[i]
    ax.plot(data1[:, i], color="r", label=lab[i]+"_model")
    ax.plot(data2[:, i], color="b", label=lab[i]+"_mujoco", linestyle="--")
    ax.legend()
    ax.set_ylabel(lab[i])

plt.xlabel('t/s')
plt.tight_layout()
plt.show()