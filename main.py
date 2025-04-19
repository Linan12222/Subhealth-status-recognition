import scipy.io
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import mahalanobis

# 参数设置
base_path = os.getcwd()  # 当前工作目录
DATASET_PATH = os.path.join(base_path, 'dataset')  # 相对路径 dataset/
SEGMENT_LENGTH = 5120  # 每段信号长度
FEATURE_NAMES = ["mean", "std", "rms", "peak", "kurtosis", "skewness"]


# 特征提取函数
def extract_features(segment):
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    rms_val = np.sqrt(np.mean(segment ** 2))
    peak_val = np.max(np.abs(segment))
    kurtosis_val = kurtosis(segment)
    skew_val = skew(segment)
    return [mean_val, std_val, rms_val, peak_val, kurtosis_val, skew_val]


# 加载数据并切段提取特征
def load_and_process_files(files, label):
    feature_list = []
    labels = []

    for file in files:
        filepath = os.path.join(DATASET_PATH, file)
        mat = scipy.io.loadmat(filepath)
        key = [k for k in mat.keys() if "_DE_time" in k][0]
        signal = mat[key].squeeze()
        total_length = len(signal)
        num_segments = total_length // SEGMENT_LENGTH

        for i in range(num_segments):
            segment = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            features = extract_features(segment)
            feature_list.append(features)
            labels.append(label)

    return np.array(feature_list), np.array(labels)


# 分类文件
all_files = os.listdir(DATASET_PATH)
health_files = [f for f in all_files if f.startswith("normal")]
subhealth_files = [f for f in all_files if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]

# 构建数据集
X_health, y_health = load_and_process_files(health_files, label=0)
X_subhealth, y_subhealth = load_and_process_files(subhealth_files, label=1)

# 构建健康状态空间（PCA）
scaler = StandardScaler()
X_health_scaled = scaler.fit_transform(X_health)
pca = PCA(n_components=3)
X_health_pca = pca.fit_transform(X_health_scaled)

mu = X_health_pca.mean(axis=0)
cov = np.cov(X_health_pca.T)
cov_inv = np.linalg.inv(cov)

# 识别亚健康样本
X_subhealth_scaled = scaler.transform(X_subhealth)
X_subhealth_pca = pca.transform(X_subhealth_scaled)
distances = [mahalanobis(x, mu, cov_inv) for x in X_subhealth_pca]

# 设置阈值
health_distances = [mahalanobis(x, mu, cov_inv) for x in X_health_pca]
threshold = np.percentile(health_distances, 96)

# 判断识别结果
subhealth_predicted = np.array([1 if d > threshold else 0 for d in distances])
true_positive = np.sum(subhealth_predicted == 1)
false_negative = np.sum(subhealth_predicted == 0)
recall = true_positive / (true_positive + false_negative)

# 健康样本误报率评估
health_predicted = np.array([1 if d > threshold else 0 for d in health_distances])
false_positive = np.sum(health_predicted == 1)
true_negative = np.sum(health_predicted == 0)
false_positive_rate = false_positive / (false_positive + true_negative)

# 输出评估结果
results = {
    "健康样本数": len(X_health),
    "亚健康样本数": len(X_subhealth),
    "误报率（健康误判为亚健康）": false_positive_rate,
    "识别率（正确识别亚健康）": recall
}
results_df = pd.DataFrame([results])
print(results_df)
