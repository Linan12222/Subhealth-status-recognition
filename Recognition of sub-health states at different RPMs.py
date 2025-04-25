import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import os
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

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
PCA_DIM = 2
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
    all_files = os.listdir(DATASET_PATH)
    health_files = [f for f in all_files if f.startswith("normal")]
    subhealth_files = [f for f in all_files if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]
    fault_files = [f for f in all_files if any(f.startswith(p) for p in ["B014", "IR014", "OR014", "B021", "IR021", "OR021"])]

    data, labels, rpms = [], [], []

    for file, label in zip([*health_files, *subhealth_files, *fault_files],
                            [0]*len(health_files) + [1]*len(subhealth_files) + [2]*len(fault_files)):
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

# === 三类状态空间建模 ===
spaces = {}
thresholds = {}
for cls in [0, 1, 2]:
    X_cls = X_train_pca[y_train == cls]
    mu = X_cls.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(X_cls.T))
    dists = [mahalanobis(x, mu, cov_inv) for x in X_cls]
    threshold = np.percentile(dists, 97) + THRESHOLD_MARGIN
    spaces[cls] = (mu, cov_inv)
    thresholds[cls] = threshold

# ==== 按转速评估 ====
results = []
for rpm in TARGET_RPMS:
    idx = rpm_test == rpm
    X = X_test_pca[idx]
    y = y_test[idx]
    pred = []

    for x in X:
        d_all = {cls: mahalanobis(x, mu, cov_inv) for cls, (mu, cov_inv) in spaces.items()}
        min_cls = min(d_all, key=d_all.get)
        if d_all[min_cls] <= thresholds[min_cls]:
            pred.append(min_cls)
        else:
            pred.append(2)  # 兜底设为故障类

    y = np.array(y)
    pred = np.array(pred)
    cm = pd.crosstab(pd.Series(y, name='实际'), pd.Series(pred, name='预测'), rownames=['实际'], colnames=['预测'], dropna=False)
    cm = cm.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)

    FP = cm.loc[0, 1] + cm.loc[0, 2]
    TN = cm.loc[0, 0]
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    TP1 = cm.loc[1, 1]
    FN1 = cm.loc[1, 0] + cm.loc[1, 2]
    recall_sub = TP1 / (TP1 + FN1) if (TP1 + FN1) != 0 else 0.0
    TP2 = cm.loc[2, 2]
    FN2 = cm.loc[2, 0] + cm.loc[2, 1]
    recall_fault = TP2 / (TP2 + FN2) if (TP2 + FN2) != 0 else 0.0

    results.append({
        "转速": rpm,
        "健康样本数": int(np.sum(y == 0)),
        "亚健康样本数": int(np.sum(y == 1)),
        "严重故障样本数": int(np.sum(y == 2)),
        "误报率（健康误判为异常）": round(FPR * 100, 5),
        "亚健康识别率": round(recall_sub * 100, 5),
        "故障识别率": round(recall_fault * 100, 5)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 创建 figures 文件夹（如果不存在）
FIGURE_PATH = "./figures"
os.makedirs(FIGURE_PATH, exist_ok=True)

# 三类状态标签对应名称和颜色
label_names = {0: "health", 1: "subhealth", 2: "fault"}
label_colors = {0: "green", 1: "orange", 2: "red"}

# 保存每个转速下的 PCA 可视化图
for rpm in TARGET_RPMS:
    idx = rpm_test == rpm
    X = X_test_pca[idx]
    y = y_test[idx]

    plt.figure(figsize=(8, 6))
    for label in [0, 1, 2]:
        mask = y == label
        if np.any(mask):
            plt.scatter(X[mask, 0], X[mask, 1],
                        c=label_colors[label],
                        label=label_names[label],
                        alpha=0.6,
                        edgecolors='k')

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Visualization at RPM {rpm}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 图片命名：pca_rpm_1725_health_subhealth_fault.png
    filename = f"pca_rpm_{rpm}_{'_'.join(label_names.values())}.png"
    save_path = os.path.join(FIGURE_PATH, filename)
    plt.savefig(save_path)
    plt.close()