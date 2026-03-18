import time

import torch


# 检查昇腾 NPU 是否可用
if not torch.npu.is_available():
    raise RuntimeError("未检测到昇腾 NPU，请确保已正确安装 CANN 软件栈和 torch-npu 库。")

# 设置设备
device_npu = torch.device("npu:0")
device_cpu = torch.device("cpu")

# 定义计算参数：矩阵大小和迭代次数
matrix_size = 4096  # 矩阵维度 (4096x4096)
iterations = 10  # 重复计算次数以取平均值

# ---------------------- CPU 计算 ----------------------
print(f"开始 CPU 计算 (矩阵大小: {matrix_size}x{matrix_size}, 迭代次数: {iterations})...")
cpu_start = time.time()

for _ in range(iterations):
    # 在 CPU 上生成随机矩阵并执行矩阵乘法
    a_cpu = torch.randn(matrix_size, matrix_size, device=device_cpu)
    b_cpu = torch.randn(matrix_size, matrix_size, device=device_cpu)
    c_cpu = torch.matmul(a_cpu, b_cpu)

cpu_time = (time.time() - cpu_start) / iterations
print(f"CPU 平均计算时间: {cpu_time:.4f} 秒\n")

# ---------------------- NPU 计算 ----------------------
print(f"开始 NPU 计算 (矩阵大小: {matrix_size}x{matrix_size}, 迭代次数: {iterations})...")

# 1. 预热（避免首次调用的初始化开销影响计时）
a_warm = torch.randn(matrix_size, matrix_size, device=device_npu)
b_warm = torch.randn(matrix_size, matrix_size, device=device_npu)
_ = torch.matmul(a_warm, b_warm)
torch.npu.synchronize()  # 等待 NPU 完成所有预热计算

# 2. 正式计时
npu_start = time.time()

for _ in range(iterations):
    # 在 NPU 上生成随机矩阵并执行矩阵乘法
    a_npu = torch.randn(matrix_size, matrix_size, device=device_npu)
    b_npu = torch.randn(matrix_size, matrix_size, device=device_npu)
    c_npu = torch.matmul(a_npu, b_npu)

torch.npu.synchronize()  # 等待 NPU 完成所有计算（关键！）
npu_time = (time.time() - npu_start) / iterations
print(f"NPU 平均计算时间: {npu_time:.4f} 秒\n")

# ---------------------- 结果对比 ----------------------
speedup = cpu_time / npu_time
print("===== 性能对比 =====")
print(f"CPU 平均耗时: {cpu_time:.4f} 秒")
print(f"NPU 平均耗时: {npu_time:.4f} 秒")
print(f"NPU 比 CPU 快: {speedup:.2f} 倍")
