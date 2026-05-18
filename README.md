# UniVAM

基于扩散模型的机器人视频预测模型。给定一段机器人操作的短视频（如机械臂抓取物体的前几帧），模型能够预测后续的未来帧画面，用于辅助机器人操作与规划任务。

## 技术架构

整个模型基于 **Wan2.2-TI2V-5B**（5B 参数文本-图像-视频扩散模型）的 Transformer 和 VAE 构建，采用 **Flow Matching** 方法进行训练和推理。

核心流程分为 4 步：

1. **WanVAE（视频压缩）**：用冻结的 VAE 编码器将原始视频帧 `[B, T, 3, H, W]` 压缩到潜在空间 `[B, 48, T', H/16, W/16]`，空间压缩 16 倍，时间压缩约 4 倍（4k+1 帧 → k+1 个潜在帧）
2. **vae_proj（通道投影+Patch切分）**：用 3D 卷积将 VAE 潜变量（48 通道）投影到投影器的隐藏维度（4096），同时做空间 patch 切分（4×4）
3. **Projector（投影器）**：通过 Q-Former 或 MLP 将大量视觉 token（如 128 个）压缩为极少量的条件 token（默认 2 个），作为扩散去噪的条件信号
4. **WanTransformer3D（去噪主干）**：将原始的文字条件模块替换为 **TimeVideoEmbedding**（视频条件嵌入），以视觉条件 token 为引导，通过 Flow Matching 迭代去噪生成未来帧的潜变量，再由 VAE 解码回像素空间

```
输入视频 → [冻结VAE] → 潜变量 [B,48,T',H/16,W/16]
                            ↓
                    [vae_proj Conv3d] → 视觉 token
                            ↓
                      [Projector] → 条件 token (2个)
                            ↓
                      TimeVideoEmbedding
                            ↓
随机噪声 → [WanTransformer3D 去噪] ← 条件信号
                 ↓
            预测潜变量 → [冻结VAE解码] → 预测视频帧
```

### 模块说明

- **TimeVideoEmbedding**：自定义模块，替换了原始 Wan2.2 的 `WanTimeTextImageEmbedding`（文本+图像条件），改为纯视觉条件嵌入，使模型以视觉特征而非文字作为去噪引导
- **vae_proj**：Wan22VisionModel 顶层的 3D 卷积（Conv3d），将 VAE 输出投影到投影器维度，不同于 Transformer 内部的 `patch_embedding`（DiT 的 patch 输入嵌入）
- **Projector**：可选 Q-Former 或 MLP 结构，将大量视觉 token 压缩为极少量条件 token

### 可选动作条件

支持额外接收机器人动作序列作为条件，使用 `ActionEncoder` 将动作编码后融合到去噪过程。

## 训练与评估

- **训练**：输入完整视频片段，VAE 编码后添加噪声，模型预测 Flow Matching 的 velocity field，损失函数为带时间步权重的 MSE
- **评估**：输入视频前几帧，VAE 编码生成条件 token，Transformer 从纯噪声开始迭代去噪（默认 20 步），VAE 解码得到预测视频，计算 PSNR/SSIM 与真实视频对比

## 训练模式

### LoRA 微调（推荐）

在预训练的 Wan2.2 Transformer 上使用 LoRA 低秩适配，大幅降低显存：

```yaml
lora:
  enable: True      # 开关，false 时全量训练
  r: 16             # 低秩维度
  lora_alpha: 32    # 缩放系数（实际 scale = alpha/r）
```

- **LoRA 层**：所有 attention 的 Q/K/V/Out 投影（8 个 Linear × N 层，~32M 参数）
- **全量训练层**（`modules_to_save`）：`condition_embedder`（自定义视频条件嵌入）、`patch_embedding`（Transformer 的 Conv3d 输入）、`proj_out`（输出投影）
- **冻结层**：FFN、LayerNorm、scale_shift 等

| 场景 | 全量训练 | LoRA (r=16) |
|------|----------|-------------|
| 可训练参数 | ~5B | ~48M |
| 单卡 bf16 | 装不下 (~70 GB) | ~12 GB（24 GB 卡可跑） |
| 8 卡 ZeRO-3 | ~12 GB/卡 | ~3 GB/卡 |

### 分布式策略

| 策略 | 配置文件 | 说明 |
|------|----------|------|
| ZeRO-2 | `configs/accelerate/zero2.yaml` | 优化器+梯度分片，参数不分片，推荐 8 卡全量训练 |
| ZeRO-3 | `configs/accelerate/zero3.yaml` + `ds_zero3.json` | 参数+优化器+梯度全部分片，支持单卡 CPU offload |

单卡 CPU offload 时，修改 `ds_zero3.json` 中 `offload_*_device` 从 `"none"` 改为 `"cpu"`。

### 检查点格式

| 文件 | 内容 | 说明 |
|------|------|------|
| `Wan22VM.pth` | Transformer 完整权重（LoRA 模式下为合并后的） | 可独立用于推理 |
| `LoraAdapter.pth` | LoRA 低秩矩阵 + modules_to_save 权重 | 仅 LoRA 模式，用于精确恢复 |
| `Projector.pth` | Projector 权重 | |
| `train_state/` | 优化器 + 调度器 + RNG 状态 | 用于完整恢复训练 |

跨模式 resume 支持：全量训练的 Wan22VM.pth 可直接加载为 LoRA 训练的基础权重（LoRA 部分随机初始化），LoRA 训练的 Wan22VM.pth（合并后）也可加载为全量训练的起点。

## 数据集支持

支持 3 种数据格式，涵盖多个机器人操作数据集：

| 格式 | 实现文件 | 支持的数据集 |
|------|----------|-------------|
| HDF5（JPEG 压缩帧） | `src/univam/utils/dataloaders/hdf5.py` | XVLA, LIBERO, RoboTwin |
| Video（decord 解码） | `src/univam/utils/dataloaders/video.py` | 通用视频文件 |
| LeRobot（HuggingFace） | `src/univam/utils/dataloaders/lerobot_.py` | LIBERO, RoboCasa, RoboTwin |

所有数据加载器统一进行：帧率重采样 → 尺寸缩放 → 归一化到 `[-1, 1]`，训练时支持随机水平翻转。

## 训练基础设施

- **分布式训练**：HuggingFace Accelerate + DeepSpeed ZeRO-2/ZeRO-3，支持单卡至 8 GPU
- **混合精度**：bf16
- **优化器**：AdamW
- **学习率调度**：线性预热后恒定（WarmupLinearConstantLR）
- **采样策略**：支持多数据集混合训练，通过 credit-based 按比例分配每个 batch 中各数据集的样本

## 项目结构

```
UniVAM/
├── train.py                          # 训练入口
├── eval.py                           # 评估入口
├── download_datasets.py              # 下载 LeRobot 数据集
├── download_models.py                # 下载预训练模型
├── configs/
│   ├── debug.yaml                    # 调试配置
│   ├── libero.yaml                   # LIBERO LoRA 训练配置
│   ├── sim.yaml                      # 仿真配置
│   ├── XVLA.yaml                     # XVLA 实验配置
│   ├── mix.yaml                      # 混合数据集配置
│   └── accelerate/
│       ├── zero2.yaml                # DeepSpeed ZeRO-2 配置
│       ├── zero3.yaml                # DeepSpeed ZeRO-3 accelerate 配置
│       └── ds_zero3.json             # DeepSpeed ZeRO-3 原生 JSON 配置
├── jsons/                            # 数据集元数据 JSON 文件
└── src/univam/
    ├── trainer.py                    # 训练循环、检查点保存、评估逻辑
    ├── models/
    │   ├── wanva.py                  # 主模型 Wan22VisionModel + LoRA + VAE 封装
    │   ├── wan.py                    # WanTransformer3D + TimeVideoEmbedding + WanVAE
    │   ├── projector.py              # Qformer / MLP 投影器
    │   ├── scheduler.py              # Flow Matching 调度器
    │   ├── action.py                 # 动作编解码器
    │   └── backbone.py               # 视觉 backbone（独立模块）
    └── utils/
        ├── args.py                   # 配置加载（OmegaConf）
        ├── data.py                   # 数据集加载、采样器、张量工具
        ├── metrics.py                # PSNR、SSIM、Meter、Timer
        ├── optim.py                  # 优化器工厂 + WarmupLinearConstantLR
        ├── files.py                  # 目录工具
        ├── overwatch.py              # 分布式日志
        └── dataloaders/
            ├── hdf5.py               # HDF5 数据集
            ├── video.py              # Decord 视频数据集
            └── lerobot_.py           # LeRobot 数据集封装
```

## 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora.enable` | `True` | 是否启用 LoRA 微调 |
| `lora.r` | 16 | LoRA 低秩维度 |
| `lora.lora_alpha` | 32 | LoRA 缩放系数 |
| `projector.type` | `qformer` | 投影器类型（`mlp` 或 `qformer`） |
| `projector.num_token` | 2 | 压缩后的条件 token 数量 |
| `projector.hidden_dim` | 4096 | 投影器隐藏层维度 |
| `projector.output_align_dim` | 4096 | 输出维度（与 Transformer 对齐） |
| `projector.patch_size` | [1, 2, 2] | 经过 VAE encoder 要接入 projector 的 3D patch 尺寸 |
| `wanva.patch_size` | [1, 2, 2] | Transformer 内部 3D patch 尺寸 |
| `wanva.num_attention_heads` | 24 | Transformer 注意力头数 |
| `data.type` | `video` | 数据集格式（`video` / `hdf5` / `lerobot`） |
| `data.frames` | 5 | 每个片段帧数（4k+1 格式） |
| `data.fps` | 2 | 目标帧率（重采样） |
| `data.image_size` | [256, 256] | 帧分辨率 |
| `train.local_batch_size` | 1 | 每 GPU 的 batch size |
| `train.learning_rate` | 3e-5 | 学习率 |
| `train.warmup_ratio` | 0.01 | 预热比例 |
| `train.decay` | 1e-3 | 权重衰减 |

## 🚀 Quick Start

### 🛠️ Installation

1. **Create and activate the conda environment:**
   ```bash
   conda create -n univam python=3.10 -y
   conda activate univam
   ```

2. **Install the package:**
   ```bash
   cd UniVAM && pip install -e .
   MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation
   ```

   Follow the install guidance to install [torchcodec](https://github.com/meta-pytorch/torchcodec).

3. **Modify Lerobot**

   After installation, please modify the corresponding source file to improve the initialization speed when episodes is specified.

   Replace the original implementation in `/path/to/site-packages/lerobot/datasets/lerobot_dataset.py#L760-L763`:

   ```python
   if self.episodes is not None:
      self._absolute_to_relative_idx = {
         abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
         for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
      }
   ```

   with the optimized version:

   ```python
   if self.episodes is not None:
      indices = self.hf_dataset.data.column("index").to_numpy()
      self._absolute_to_relative_idx = dict(
            zip(indices.tolist(), range(len(indices)))
      )
   ```

   Reference https://github.com/huggingface/lerobot/pull/3279

### 📂 Usage Workflow

#### 1. Environment Setup

Generate the `.env` template and edit it to set correct paths:

```bash
bash scripts/envs/generate_dotenv.sh
# Edit .env to set:
#   PRETRAINED_MODEL_PATH=/path/to/pretrained/models
#   DATASETS_PATH=/path/to/datasets
#   RESUME_PATH=./ckpt/model/        (optional)
#   EVAL_JSON_PATH=./jsons/debug.json (optional)
```

#### 2. Download Pretrained Models

```bash
python download_models.py
```

This downloads `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (VAE + Transformer weights) and `timm/vit_large_patch16_dinov3.lvd1689m` to `$PRETRAINED_MODEL_PATH`.

#### 3. Download Datasets

Choose the method that matches your target data format:

```bash
# LeRobot format (LIBERO, RoboCasa, RoboTwin)
python download_datasets.py

# HDF5 format (XVLA-Soft-Fold)
bash scripts/datasets/download.sh

# RoboTwin2.0 (ZIP format)
python scripts/datasets/download_robotwin.py --output_dir $DATASETS_PATH/robotwin
```

#### 4. Create Dataset JSON Configs

Generate JSON metadata files for training. Choose the script matching your data format:

```bash
# For HDF5 data (e.g., XVLA)
python scripts/datasets/create_hdf5_jsons.py

# For LeRobot data (LIBERO, RoboCasa, RoboTwin)
python scripts/datasets/create_lerobot_jsons.py

# For video files (.mp4, .avi, etc.)
python scripts/datasets/create_video_jsons.py
```

These scripts write JSONL files and train/eval JSON configs to `./jsons/`.

#### 5. Modify the Config File

Edit the YAML config (e.g., `configs/debug.yaml`) to match your setup:

- `data.type` — dataset format (`hdf5`, `lerobot`, or `video`)
- `data.fps` / `data.frames` / `data.image_size` — clip parameters
- `train.eval_step` / `train.save_step` — eval and checkpoint intervals
- `train.learning_rate` / `train.local_batch_size` — training hyperparameters
- `wanva.model_path` — path to pretrained Wan2.2 model (`$PRETRAINED_MODEL_PATH/Wan-AI/Wan2.2-TI2V-5B-Diffusers/`)

The training script you choose in the next step determines which config file is used (e.g., `train_debug.sh` → `configs/debug.yaml`, `train_XVLA.sh` → `configs/XVLA.yaml`).

#### 6. Start Training

```bash
# Quick test with debug config (8 GPUs)
bash scripts/train/train_debug.sh

# Multi-node distributed training
bash scripts/train/train_multi.sh
```

Training checkpoints are saved to `./ckpt/<task_name>/` and logs to `./logs/<task_name>/`.

#### 7. Evaluation

```bash
python eval.py --config_path configs/debug.yaml
```

Evaluation computes PSNR and SSIM and saves side-by-side ground-truth vs. predicted frames.
