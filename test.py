import scipy.io
import os

# 设置你的数据路径（使用原始字符串避免转义问题）

base_path = os.getcwd()  # 当前工作目录
DATASET_PATH = os.path.join(base_path, 'dataset')  # 相对路径 dataset/
# 获取所有 .mat 文件名称
mat_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".mat")]
print("共找到文件数：", len(mat_files))
print("文件示例：", mat_files[:5])

# 选择一个文件查看内容结构（你也可以改成其他文件）
sample_file = os.path.join(DATASET_PATH, mat_files[0])
print(f"\n正在加载文件: {sample_file}")

# 加载 .mat 文件
mat_data = scipy.io.loadmat(sample_file)

# 查看文件中有哪些变量（键名）
print("\n该文件包含以下变量：")
for key in mat_data.keys():
    if not key.startswith("__"):
        print("  -", key)

# 举例查看某个变量的数据（如 DE_time）
if 'DE_time' in mat_data:
    print("\nDE_time 前10个数据点：")
    print(mat_data['DE_time'].squeeze()[:10])

all_files = os.listdir(DATASET_PATH)
rpm_values = set()

for file in all_files:
    filepath = os.path.join(DATASET_PATH, file)
    mat = scipy.io.loadmat(filepath)
    rpm_key = [k for k in mat if 'RPM' in k]
    if rpm_key:
        rpm = int(mat[rpm_key[0]].squeeze())
        rpm_values.add(rpm)

print("所有包含 RPM 的转速值：", sorted(rpm_values))