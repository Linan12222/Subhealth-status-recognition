# 导入相关库
import scipy.io  # 用于加载MAT格式的数据文件
import os  # 用于处理文件路径
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理与表格展示库
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.preprocessing import StandardScaler  # 标准化工具
from scipy.stats import kurtosis, skew  # 计算峰度和偏度
from scipy.spatial.distance import mahalanobis  # 马氏距离计算
import matplotlib.pyplot as plt  # 绘图库

# ==== 参数设置 ====
base_path = os.getcwd()  # 获取当前工作目录
DATASET_PATH = os.path.join(base_path, 'dataset')  # 数据集目录路径
FIGURE_PATH = os.path.join(base_path, 'figures')  # 图像保存路径
os.makedirs(FIGURE_PATH, exist_ok=True)  # 如果图像路径不存在则创建
SEGMENT_LENGTH = 5120  # 每段信号的长度，用于分段处理

# ==== 特征提取函数 ====
def extract_features(segment):
    # 提取一个信号段的6个时域特征
    return [
        np.mean(segment),  # 均值
        np.std(segment),  # 标准差
        np.sqrt(np.mean(segment ** 2)),  # 均方根值
        np.max(np.abs(segment)),  # 峰值
        kurtosis(segment),  # 峰度
        skew(segment)  # 偏度
    ]

# ==== 加载数据并切段 ====
def load_all_segments(file_list, label):
    features, labels, rpms = [], [], []  # 初始化特征、标签、转速数组
    for file in file_list:
        filepath = os.path.join(DATASET_PATH, file)  # 构造完整路径
        mat = scipy.io.loadmat(filepath)  # 读取.mat文件
        key = [k for k in mat if '_DE_time' in k][0]  # 获取关键数据字段
        signal = mat[key].squeeze()  # 获取信号数据
        rpm_key = [k for k in mat if 'RPM' in k]  # 查找转速字段
        rpm = int(mat[rpm_key[0]].squeeze()) if rpm_key else 0  # 获取转速值

        num_segments = len(signal) // SEGMENT_LENGTH  # 计算可以分成多少段
        for i in range(num_segments):
            segment = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]  # 取出一段
            features.append(extract_features(segment))  # 提取特征
            labels.append(label)  # 加入对应标签
            rpms.append(rpm)  # 记录转速
    return np.array(features), np.array(labels), np.array(rpms)  # 返回特征、标签、转速数组

# ==== 分类文件 ====
all_files = os.listdir(DATASET_PATH)  # 获取所有数据文件名
health_files = [f for f in all_files if f.startswith("normal")]  # 正常文件
subhealth_files = [f for f in all_files if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]  # 亚健康文件

# ==== 加载数据 ====
X_health, y_health, rpm_health = load_all_segments(health_files, label=0)  # 加载正常数据
X_subhealth, y_subhealth, rpm_sub = load_all_segments(subhealth_files, label=1)  # 加载亚健康数据

# ==== 健康状态空间绘图函数 ====
def visualize_health_space(X_pca, distances, threshold, rpm):
    fig = plt.figure(figsize=(9, 7))  # 创建图像
    ax = fig.add_subplot(111, projection='3d')  # 3D图

    # 颜色根据马氏距离进行分类
    colors = []
    for d in distances:
        if d <= 0.5 * threshold:
            colors.append('yellow')  # 健康中心
        elif d <= 0.75 * threshold:
            colors.append('red')  # 内边缘健康
        elif d <= threshold:
            colors.append('blue')  # 外边缘健康
        else:
            colors.append('green')  # 异常

    # 绘制散点图
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, s=15, alpha=0.85)

    # 添加图像标题与坐标轴标签
    ax.set_title(f'3D Health State Space - {rpm} RPM', fontsize=14)
    ax.set_xlabel('PC1 axis')
    ax.set_ylabel('PC2 axis')
    ax.set_zlabel('PC3 axis')

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy (Center)', markerfacecolor='yellow', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy (Inner Margin)', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy (Outer Margin)', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Abnormal', markerfacecolor='green', markersize=8)
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()  # 紧凑布局

    # 保存图像
    fig_path = os.path.join(FIGURE_PATH, f"health_space_{rpm}RPM.png")
    plt.savefig(fig_path)
    plt.close()  # 关闭图像避免内存溢出

# ==== 主识别过程 ====
results = []  # 用于存储每种转速下的识别结果
unique_rpms = np.unique(rpm_health)  # 获取所有不同的转速值

for target_rpm in unique_rpms:
    # 提取该转速下的正常与亚健康样本
    idx_health = (rpm_health == target_rpm)
    idx_sub = (rpm_sub == target_rpm)

    if np.sum(idx_health) == 0 or np.sum(idx_sub) == 0:
        continue  # 若该转速下数据不足，则跳过

    X_h = X_health[idx_health]  # 当前转速下的健康数据
    X_s = X_subhealth[idx_sub]  # 当前转速下的亚健康数据

    # 数据标准化与PCA降维
    scaler = StandardScaler()
    X_h_scaled = scaler.fit_transform(X_h)
    pca = PCA(n_components=3)
    X_h_pca = pca.fit_transform(X_h_scaled)

    # 计算健康中心与协方差矩阵的逆
    mu = X_h_pca.mean(axis=0)
    cov = np.cov(X_h_pca.T)
    cov_inv = np.linalg.inv(cov)

    # 计算健康数据的马氏距离
    health_distances = [mahalanobis(x, mu, cov_inv) for x in X_h_pca]
    threshold = np.percentile(health_distances, 999)  # 设置99百分位作为判定阈值

    # 计算亚健康数据的马氏距离并进行识别
    X_s_scaled = scaler.transform(X_s)
    X_s_pca = pca.transform(X_s_scaled)
    distances = [mahalanobis(x, mu, cov_inv) for x in X_s_pca]
    predicted = np.array([1 if d > threshold else 0 for d in distances])  # 判定为异常为1，否则为0

    # 计算识别率与误报率
    recall = np.sum(predicted == 1) / len(predicted)  # 识别率
    health_predicted = np.array([1 if d > threshold else 0 for d in health_distances])  # 健康样本的误报
    false_positive_rate = np.sum(health_predicted == 1) / len(health_predicted)  # 误报率

    # 存储识别结果
    results.append({
        "转速RPM": target_rpm,
        "健康样本数": len(X_h),
        "亚健康样本数": len(X_s),
        "误报率（健康误判为亚健康）": round(false_positive_rate * 100, 2),
        "识别率（正确识别亚健康）": round(recall * 100, 2),
        "马氏距离阈值": round(threshold, 4)
    })

    # 绘制健康状态空间图像
    visualize_health_space(X_h_pca, health_distances, threshold, target_rpm)

# ==== 输出中文识别结果表格 ====
results_df = pd.DataFrame(results)  # 将结果转为表格
print(results_df)  # 打印输出
