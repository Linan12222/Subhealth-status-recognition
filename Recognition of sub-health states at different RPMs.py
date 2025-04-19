import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# ==== 参数设置 ====
base_path = os.getcwd()
DATASET_PATH = os.path.join(base_path, 'dataset')
FIGURE_PATH = os.path.join(base_path, 'figures')
os.makedirs(FIGURE_PATH, exist_ok=True)
SEGMENT_LENGTH = 5120

# ==== 特征提取函数 ====
def extract_features(segment):
    return [
        np.mean(segment),
        np.std(segment),
        np.sqrt(np.mean(segment ** 2)),
        np.max(np.abs(segment)),
        kurtosis(segment),
        skew(segment)
    ]

# ==== 加载数据并切段 ====
def load_all_segments(file_list, label):
    features, labels, rpms = [], [], []
    for file in file_list:
        filepath = os.path.join(DATASET_PATH, file)
        mat = scipy.io.loadmat(filepath)
        key = [k for k in mat if '_DE_time' in k][0]
        signal = mat[key].squeeze()
        rpm_key = [k for k in mat if 'RPM' in k]
        rpm = int(mat[rpm_key[0]].squeeze()) if rpm_key else 0

        num_segments = len(signal) // SEGMENT_LENGTH
        for i in range(num_segments):
            segment = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            features.append(extract_features(segment))
            labels.append(label)
            rpms.append(rpm)
    return np.array(features), np.array(labels), np.array(rpms)

# ==== 分类文件 ====
all_files = os.listdir(DATASET_PATH)
health_files = [f for f in all_files if f.startswith("normal")]
subhealth_files = [f for f in all_files if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]

# ==== 加载数据 ====
X_health, y_health, rpm_health = load_all_segments(health_files, label=0)
X_subhealth, y_subhealth, rpm_sub = load_all_segments(subhealth_files, label=1)

# ==== 健康状态空间绘图 ====
def visualize_health_space(X_pca, distances, threshold, rpm):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 区分四类：中心健康（黄），内边缘健康（红），外边缘健康（蓝），异常（绿）
    colors = []
    for d in distances:
        if d <= 0.5 * threshold:
            colors.append('yellow')  # 健康中心
        elif d <= 0.75 * threshold:
            colors.append('red')     # 内边缘健康
        elif d <= threshold:
            colors.append('blue')    # 外边缘健康
        else:
            colors.append('green')   # 异常

    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, s=15, alpha=0.85)

    ax.set_title(f'3D Health State Space - {rpm} RPM', fontsize=14)
    ax.set_xlabel('PC1 axis')
    ax.set_ylabel('PC2 axis')
    ax.set_zlabel('PC3 axis')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy (Center)', markerfacecolor='yellow', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy (Inner Margin)', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Healthy (Outer Margin)', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Abnormal', markerfacecolor='green', markersize=8)
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()

    # 保存图像
    fig_path = os.path.join(FIGURE_PATH, f"health_space_{rpm}RPM.png")
    plt.savefig(fig_path)
    plt.close()

# ==== 主识别过程 ====
results = []
unique_rpms = np.unique(rpm_health)

for target_rpm in unique_rpms:
    idx_health = (rpm_health == target_rpm)
    idx_sub = (rpm_sub == target_rpm)

    if np.sum(idx_health) == 0 or np.sum(idx_sub) == 0:
        continue

    X_h = X_health[idx_health]
    X_s = X_subhealth[idx_sub]

    scaler = StandardScaler()
    X_h_scaled = scaler.fit_transform(X_h)
    pca = PCA(n_components=3)
    X_h_pca = pca.fit_transform(X_h_scaled)

    mu = X_h_pca.mean(axis=0)
    cov = np.cov(X_h_pca.T)
    cov_inv = np.linalg.inv(cov)
    health_distances = [mahalanobis(x, mu, cov_inv) for x in X_h_pca]
    threshold = np.percentile(health_distances, 97)

    X_s_scaled = scaler.transform(X_s)
    X_s_pca = pca.transform(X_s_scaled)
    distances = [mahalanobis(x, mu, cov_inv) for x in X_s_pca]
    predicted = np.array([1 if d > threshold else 0 for d in distances])

    recall = np.sum(predicted == 1) / len(predicted)
    health_predicted = np.array([1 if d > threshold else 0 for d in health_distances])
    false_positive_rate = np.sum(health_predicted == 1) / len(health_predicted)

    results.append({
        "转速RPM": target_rpm,
        "健康样本数": len(X_h),
        "亚健康样本数": len(X_s),
        "误报率（健康误判为亚健康）": round(false_positive_rate * 100, 2),
        "识别率（正确识别亚健康）": round(recall * 100, 2),
        "马氏距离阈值": round(threshold, 4)
    })

    # 绘图并保存
    visualize_health_space(X_h_pca, health_distances, threshold, target_rpm)

# ==== 输出中文识别结果表格 ====
results_df = pd.DataFrame(results)

print(results_df)