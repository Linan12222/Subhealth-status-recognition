import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import confusion_matrix

# ==== 参数设置 ====
SEGMENT_LENGTH = 5120
CNN_OUTPUT_DIM = 64
PCA_DIM = 3
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DATASET_PATH = './dataset'

# ==== 1D-CNN 网络结构 ====
class Simple1DCNN(nn.Module):
    def __init__(self, output_dim=CNN_OUTPUT_DIM):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
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
    data, labels = [], []

    for file, label in zip([*health_files, *subhealth_files, *fault_files],
                            [0]*len(health_files) + [1]*len(subhealth_files) + [2]*len(fault_files)):
        filepath = os.path.join(DATASET_PATH, file)
        mat = scipy.io.loadmat(filepath)
        key = [k for k in mat if "_DE_time" in k][0]
        signal = mat[key].squeeze()

        num_segments = len(signal) // SEGMENT_LENGTH
        for i in range(num_segments):
            segment = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            data.append(segment)
            labels.append(label)
    return np.array(data), np.array(labels)

# ==== CNN 训练与提取器 ====
class CNNFeatureExtractor:
    def __init__(self, output_dim=CNN_OUTPUT_DIM):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Simple1DCNN(output_dim=output_dim).to(self.device)
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
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def extract(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            features = self.model(inputs).cpu().numpy()
        return features

# ==== 主流程 ====
raw_data, raw_labels = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    raw_data, raw_labels, test_size=0.3, stratify=raw_labels, random_state=42)

cnn_extractor = CNNFeatureExtractor()
cnn_extractor.train(X_train, y_train)

X_train_feat = cnn_extractor.extract(X_train)
X_test_feat = cnn_extractor.extract(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

pca = PCA(n_components=PCA_DIM)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ==== 构建每类空间中心和阈值 ====
spaces = {}
thresholds = {}
for cls in [0, 1, 2]:
    X_cls = X_train_pca[y_train == cls]
    mu = X_cls.mean(axis=0)
    cov = np.cov(X_cls.T)
    cov_inv = np.linalg.inv(cov)
    dists = [mahalanobis(x, mu, cov_inv) for x in X_cls]
    threshold = np.percentile(dists, 99)
    spaces[cls] = (mu, cov_inv)
    thresholds[cls] = threshold

# ==== 多阈值判定分类 ====
y_pred = []
for x in X_test_pca:
    dists = {cls: mahalanobis(x, mu, cov_inv) for cls, (mu, cov_inv) in spaces.items()}
    if dists[0] <= thresholds[0]:
        y_pred.append(0)
    elif dists[1] <= thresholds[1]:
        y_pred.append(1)
    else:
        y_pred.append(2)

# ==== 混淆矩阵与指标 ====
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
FP = cm[0, 1] + cm[0, 2]
TN = cm[0, 0]
FPR = FP / (FP + TN)
TP = cm[1, 1]
FN = cm[1, 0] + cm[1, 2]
recall_sub = TP / (TP + FN)
TP2 = cm[2, 2]
FN2 = cm[2, 0] + cm[2, 1]
recall_fault = TP2 / (TP2 + FN2)

# ==== 输出结果 ====
results = pd.DataFrame([{
    "健康样本数": int(np.sum(y_test == 0)),
    "亚健康样本数": int(np.sum(y_test == 1)),
    "严重故障样本数": int(np.sum(y_test == 2)),
    "误报率（健康误判为异常）": round(FPR * 100, 2),
    "亚健康识别率": round(recall_sub * 100, 2),
    "故障识别率": round(recall_fault * 100, 2)
}])

print(results)