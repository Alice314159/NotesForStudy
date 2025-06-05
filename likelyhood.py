import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import comb  # 计算组合数 C(n, k)
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
# 固定观测结果
n = 10  # 抛硬币次数
k = 8   # 出现正面次数

# 定义似然函数 L(p) = C(n,k) * p^k * (1-p)^(n-k)
def likelihood(p, n, k):
    return comb(n, k) * (p**k) * ((1 - p)**(n - k))

# 参数范围：p 从 0 到 1（100 个点）
p_values = np.linspace(0, 1, 200)
likelihood_values = likelihood(p_values, n, k)

# 找到最大似然点
max_index = np.argmax(likelihood_values)
p_mle = p_values[max_index]
L_max = likelihood_values[max_index]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(p_values, likelihood_values, label='似然函数 L(p)', color='blue')
plt.axvline(p_mle, color='red', linestyle='--', label=f'最大似然点：p = {p_mle:.2f}')
plt.scatter([0.5], [likelihood(0.5, n, k)], color='green', label='p = 0.5')

# 标注
plt.title('抛10次硬币得到8次正面：似然函数图像', fontsize=14)
plt.xlabel('硬币正面概率 p')
plt.ylabel('似然值 L(p)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
