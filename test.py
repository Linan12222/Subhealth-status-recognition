import os
import scipy.io
import numpy as np
import pandas as pd

DATASET_PATH = './dataset'
SEGMENT_LENGTH = 5120

TARGET_RPM_GROUPS = {
    1720: (1700, 1740),
    1800: (1780, 1820)
}


def load_filtered_data():
    health_files = [f for f in os.listdir(DATASET_PATH) if f.startswith("normal")]
    subhealth_files = [f for f in os.listdir(DATASET_PATH) if any(f.startswith(p) for p in ["B007", "IR007", "OR007"])]
    fault_files = [f for f in os.listdir(DATASET_PATH) if
                   any(f.startswith(p) for p in ["B014", "IR014", "OR014", "B021", "IR021", "OR021"])]

    data, labels, rpms, rpm_groups = [], [], [], []

    for file, label in zip([*health_files, *subhealth_files, *fault_files],
                           [0] * len(health_files) + [1] * len(subhealth_files) + [2] * len(fault_files)):
        mat = scipy.io.loadmat(os.path.join(DATASET_PATH, file))
        key = [k for k in mat if "_DE_time" in k][0]
        signal = mat[key].squeeze()
        rpm_key = [k for k in mat if "RPM" in k]
        rpm = int(mat[rpm_key[0]].squeeze()) if rpm_key else 0

        matched_group = None
        for group, (low, high) in TARGET_RPM_GROUPS.items():
            if low <= rpm < high:
                matched_group = group
                break
        if matched_group is None:
            continue

        num_segments = len(signal) // SEGMENT_LENGTH
        for i in range(num_segments):
            segment = signal[i * SEGMENT_LENGTH:(i + 1) * SEGMENT_LENGTH]
            data.append(segment)
            labels.append(label)
            rpms.append(rpm)
            rpm_groups.append(matched_group)

    return np.array(data), np.array(labels), np.array(rpms), np.array(rpm_groups)


# 检查样本统计信息
raw_data, raw_labels, raw_rpms, raw_rpm_groups = load_filtered_data()
df = pd.DataFrame({
    '转速段': raw_rpm_groups,
    '标签': raw_labels
})
pivot = df.pivot_table(index='转速段', columns='标签', aggfunc='size', fill_value=0)
pivot.columns = ['健康样本数', '亚健康样本数', '故障样本数']
pivot.reset_index(inplace=True)
print(pivot)
