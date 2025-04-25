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
from sklearn.metrics import confusion_matrix
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
PCA_DIM = 3
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATASET_PATH = './dataset'
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
    fault_files = [f for f in os.listdir(DATASET_PATH) if any(f.startswith(p) for p in ["B014", "IR014", "OR014", "B021", "IR021", "OR021"])]

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

X_train, X_test, y_train, y_test = train_test_split(
    raw_data, raw_labels, test_size=0.3, stratify=raw_labels, random_state=42)

extractor = CNNFeatureExtractor()
extractor.train(X_train, y_train)
X_train_feat = extractor.extract(X_train)
X_test_feat = extractor.extract(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)
pca = PCA(n_components=PCA_DIM)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# === 三类状态空间建模 + 阈值设置 ===
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

# === 分类评估 ===
y_pred = []
for x in X_test_pca:
    d_all = {cls: mahalanobis(x, mu, cov_inv) for cls, (mu, cov_inv) in spaces.items()}
    min_cls = min(d_all, key=d_all.get)
    if d_all[min_cls] <= thresholds[min_cls]:
        y_pred.append(min_cls)
    else:
        y_pred.append(2)  # 兜底为故障类

# === 输出混淆矩阵 ===
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
df_result = pd.DataFrame(cm, index=['实际_健康', '实际_亚健康', '实际_故障'],
                         columns=['预测_健康', '预测_亚健康', '预测_故障'])
print(df_result)

# === 亚健康类别的二分类评估 ===
y_true_binary = (np.array(y_test) == 1).astype(int)
y_pred_binary = (np.array(y_pred) == 1).astype(int)

TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))

recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
fpr = FP / (FP + TN) if (FP + TN) != 0 else 0.0
precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0

print(f"\n【亚健康识别指标 - 二分类视角】")
print(f"TP（真正）     = {TP}")
print(f"FN（假负）     = {FN}")
print(f"FP（假正）     = {FP}")
print(f"TN（真负）     = {TN}")
print(f"Recall（召回率）= {recall:.4f}")
print(f"FPR（误报率）   = {fpr:.4f}")
print(f"Precision（精确率）= {precision:.4f}")
print(f"F1 Score        = {f1:.4f}")
# === 创建保存目录 ===
FIGURE_PATH = "./figures"
os.makedirs(FIGURE_PATH, exist_ok=True)

# === 标签颜色定义 ===
label_names = {0: "health", 1: "subhealth", 2: "fault"}
label_colors = {0: "green", 1: "orange", 2: "red"}

# === 绘制并保存整体 PCA 图 ===
plt.figure(figsize=(8, 6))
for label in [0, 1, 2]:
    mask = np.array(y_test) == label
    if np.any(mask):
        plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                    c=label_colors[label],
                    label=label_names[label],
                    alpha=0.6,
                    edgecolors='k')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of Health / Subhealth / Fault")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
save_path = os.path.join(FIGURE_PATH, "pca_health_subhealth_fault.png")
plt.savefig(save_path)
plt.close()