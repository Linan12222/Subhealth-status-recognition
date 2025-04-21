import os
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis

# ==== 固定随机种子 ====
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== 参数设置 ====
SEGMENT_LENGTH = 5120
CNN_OUTPUT_DIM = 64
PCA_DIM = 3
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATASET_PATH = './dataset'
TARGET_RPMS = [1725, 1796]
THRESHOLD_MARGIN = 0.2  # 差值阈值

# ==== CNN 网络结构 ====
class Simple1DCNN(nn.Module):
    def __init__(self, output_dim=CNN_OUTPUT_DIM):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (SEGMENT_LENGTH // 4), output_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# ==== 加载数据 ====
def load_data():
    health_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("normal")]
    subhealth_files = [f for f in os.listdir(DATASET_PATH) if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]

    data, labels, rpms = [], [], []

    for file, label in zip([*health_files, *subhealth_files],
                            [0]*len(health_files) + [1]*len(subhealth_files)):
        filepath = os.path.join(DATASET_PATH, file)
        mat = scipy.io.loadmat(filepath)
        rpm_key = [k for k in mat if "RPM" in k]
        rpm = int(mat[rpm_key[0]].squeeze()) if rpm_key else 0

        key = [k for k in mat if "_DE_time" in k][0]
        signal = mat[key].squeeze()
        num_segments = len(signal) // SEGMENT_LENGTH

        for i in range(num_segments):
            segment = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            data.append(segment)
            labels.append(label)
            rpms.append(rpm)

    return np.array(data), np.array(labels), np.array(rpms)

# ==== CNN 特征提取器 ====
class CNNFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Simple1DCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, X, y):
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1),
                                                 torch.tensor(y, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.model.train()
        for epoch in range(EPOCHS):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch_X), batch_y)
                loss.backward()
                self.optimizer.step()

    def extract(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            return self.model(inputs).cpu().numpy()

# ==== 主流程 ====
raw_data, raw_labels, raw_rpms = load_data()

# === 划分训练集和测试集 ===
X_train, X_test, y_train, y_test, rpm_train, rpm_test = train_test_split(
    raw_data, raw_labels, raw_rpms, test_size=0.3, stratify=raw_labels, random_state=42)

# === 提取特征 ===
extractor = CNNFeatureExtractor()
extractor.train(X_train, y_train)
X_train_feat = extractor.extract(X_train)
X_test_feat = extractor.extract(X_test)

# === PCA降维 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)
pca = PCA(n_components=PCA_DIM)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# === 构建状态空间 ===
X_health = X_train_pca[y_train == 0]
X_subhealth = X_train_pca[y_train == 1]
mu_h = X_health.mean(axis=0)
cov_h_inv = np.linalg.inv(np.cov(X_health.T))
mu_s = X_subhealth.mean(axis=0)
cov_s_inv = np.linalg.inv(np.cov(X_subhealth.T))

# === 不区分转速，直接评估全体测试集 ===
X = X_test_pca
y = y_test

pred = []
for x in X:
    d_h = mahalanobis(x, mu_h, cov_h_inv)
    d_s = mahalanobis(x, mu_s, cov_s_inv)
    if (d_s - d_h) > THRESHOLD_MARGIN:
        pred.append(0)
    else:
        pred.append(1)

y = np.array(y)
pred = np.array(pred)

TP = np.sum((y == 1) & (pred == 1))
FN = np.sum((y == 1) & (pred == 0))
FP = np.sum((y == 0) & (pred == 1))
TN = np.sum((y == 0) & (pred == 0))

recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
fpr = FP / (FP + TN) if (FP + TN) != 0 else 0.0

print(f"[总体评估] TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
print(f"健康样本数: {int(np.sum(y == 0))}, 亚健康样本数: {int(np.sum(y == 1))}")
print(f"误报率（健康误判为异常）: {round(fpr * 100, 2)}%")
print(f"亚健康识别率: {round(recall * 100, 2)}%")
