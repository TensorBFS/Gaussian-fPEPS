import numpy as np
import matplotlib.pyplot as plt

# 数据
Nv = np.array([1, 2, 3, 4, 5])
# value = np.array([0.257, 0.066, 0.010, 0.0039, 0.0027])  # 如果缺省最后一个，可以删除
value = np.array([0.155, 0.04, 0.0038, 0.0019, 0.0012])  # 如果缺省最后一个，可以删除

# 创建图像
plt.figure(figsize=(12, 5))

# 半对数图（测试指数函数）
plt.subplot(1, 2, 1)
plt.semilogy(Nv, value, 'o-', label='Data')
plt.xlabel('Nv')
plt.ylabel('Value (log scale)')
plt.title('Semi-log plot (test exponential)')
plt.grid(True)
plt.legend()

# 对数-对数图（测试幂函数）
plt.subplot(1, 2, 2)
plt.loglog(Nv, value, 's-', label='Data')
plt.xlabel('Nv (log scale)')
plt.ylabel('Value (log scale)')
plt.title('Log-log plot (test power law)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
