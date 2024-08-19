import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 定义数据
x_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y_values = np.array([1.0, 0.97109313, 0.94208427, 0.90896111, 0.78664316, 0.79223301,
                     0.80166854, 0.83291585, 0.82682638, 0.78592757, 0.68645542])
y_values = np.array([1., 0.97228827,  0.90951168,  0.90015297, 0.85515135, 0.80585734,
 0.80288429, 0.79998087, 0.79464387, 0.78851276, 0.77830577])

# 使用 make_interp_spline 进行插值
x_new = np.linspace(x_values.min(), x_values.max(), 300)  # 生成更多的 x 值
spline = make_interp_spline(x_values, y_values, k=3)  # k=3 表示三次样条插值
y_smooth = spline(x_new)

# 绘制光滑曲线
plt.figure(figsize=(8, 6))
plt.plot(x_new, y_smooth, color='b', label='Smooth Curve')
plt.scatter(x_values, y_values, color='r', marker='o', label='Data Points')  # 标记原始数据点
plt.title('Smooth Curve Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度
plt.yticks(np.arange(0.6, 1.1, 0.1))  # 设置 y 轴刻度
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 显示图形
plt.show()
