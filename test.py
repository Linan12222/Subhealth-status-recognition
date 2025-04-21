import os
import scipy.io
import numpy as np
import pandas as pd

# 假设数据集路径
DATASET_PATH = './dataset'
SEGMENT_LENGTH = 5120

# 分类文件
health_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("normal")]
subhealth_files = [f for f in os.listdir(DATASET_PATH) if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]
fault_files = [f for f in os.listdir(DATASET_PATH) if any(f.startswith(p) for p in ["B014", "IR014", "OR014", "B021", "IR021", "OR021"])]

rpm_class_count = {}

def process_file(file, label):
    filepath = os.path.join(DATASET_PATH, file)
    mat = scipy.io.loadmat(filepath)
    rpm_key = [k for k in mat if "RPM" in k]
    rpm = int(mat[rpm_key[0]].squeeze()) if rpm_key else 0
    signal_key = [k for k in mat if '_DE_time' in k][0]
    signal = mat[signal_key].squeeze()
    num_segments = len(signal) // SEGMENT_LENGTH
    if rpm not in rpm_class_count:
        rpm_class_count[rpm] = [0, 0, 0]
    rpm_class_count[rpm][label] += num_segments

# 处理所有文件
for file in health_files:
    process_file(file, label=0)
for file in subhealth_files:
    process_file(file, label=1)
for file in fault_files:
    process_file(file, label=2)

# 整理为DataFrame
rpm_df = pd.DataFrame([
    {"转速RPM": rpm, "健康样本数": counts[0], "亚健康样本数": counts[1], "严重故障样本数": counts[2]}
    for rpm, counts in sorted(rpm_class_count.items())
])

print(rpm_df)

