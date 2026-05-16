# UniVAM

基于扩散模型的机器人视频预测模型。给定一段机器人操作的短视频（如机械臂抓取物体的前几帧），模型能够预测后续的未来帧画面，用于辅助机器人操作与规划任务。

## 技术架构

整个模型基于 **Wan2.2-TI2V-5B**（5B 参数文本-图像-视频扩散模型）构建，采用 **Flow Matching** 方法进行训练和推理。

核心流程分为 4 步：

1. **WanVAE（视频压缩）**：用冻结的 VAE 编码器将原始视频帧 `[B, T, 3, H, W]` 压缩到潜在空间 `[B, 16, T', H/8, W/8]`，空间压缩 8 倍，时间压缩约 4 倍
2. **Patch Embedding**：用 3D 卷积将 VAE 潜变量进一步切分为视觉 token 序列
3. **Projector（投影器）**：通过 Q-Former 或 MLP 将大量视觉 token 压缩为极少量的条件 token（默认仅 2-8 个），作为扩散去噪的条件信号
4. **WanTransformer3D（去噪主干）**：30 层的 3D 扩散 Transformer，以条件 token 为引导，通过 Flow Matching 迭代去噪生成未来帧的潜变量，再由 VAE 解码回像素空间

```
输入视频 → [冻结VAE] → 潜变量 → [Patch+投影] → 条件token(2-8个)
                                                      ↓
随机噪声 → [WanTransformer3D 去噪] ← 条件token 引导
                 ↓
            预测潜变量 → [冻结VAE解码] → 预测视频帧
```

### 可选投影器结构

- **QformerProjector**：使用可学习的 query token 与视觉 token 做交叉注意力
- **MLPProjector**：通过多层自注意力 + 卷积逐步压缩

### 可选动作条件

支持 `train_with_action=True` 时，模型额外接收机器人动作序列作为条件，使用 `ActionEncoder` 将动作编码后融合到去噪过程中，实现动作条件下的视频预测。

## 训练与评估

- **训练**：输入完整视频片段，VAE 编码后添加噪声，模型预测 Flow Matching 的 velocity field，损失函数为带时间步权重的 MSE
- **评估**：输入视频前几帧，VAE 编码生成条件 token，Transformer 从纯噪声开始迭代去噪（默认 20 步），VAE 解码得到预测视频，计算 PSNR/SSIM 与真实视频对比

## 数据集支持

支持 3 种数据格式，涵盖多个机器人操作数据集：

| 格式 | 实现文件 | 支持的数据集 |
|------|----------|-------------|
| HDF5（JPEG 压缩帧） | `src/univam/utils/dataloaders/hdf5.py` | XVLA, LIBERO, RoboTwin |
| Video（decord 解码） | `src/univam/utils/dataloaders/video.py` | 通用视频文件 |
| LeRobot（HuggingFace） | `src/univam/utils/dataloaders/lerobot_.py` | LIBERO, RoboCasa, RoboTwin |

所有数据加载器统一进行：帧率重采样 → 尺寸缩放 → 归一化到 `[-1, 1]`，训练时支持随机水平翻转。

## 训练基础设施

- **分布式训练**：HuggingFace Accelerate + DeepSpeed ZeRO-2，支持 8 GPU
- **混合精度**：bf16
- **优化器**：AdamW，三组不同学习率（projector 和 condition_embedder 用完整 lr，transformer 其余部分用基础 lr）
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
│   ├── sim.yaml                      # 仿真配置
│   ├── XVLA.yaml                     # XVLA 实验配置
│   ├── XVLA_with_action.yaml         # XVLA + 动作条件配置
│   ├── mix.yaml                      # 混合数据集配置
│   └── accelerate/zero2.yaml         # DeepSpeed ZeRO-2 配置
├── jsons/                            # 数据集元数据 JSON 文件
└── src/univam/
    ├── trainer.py                    # 训练循环、检查点保存、评估逻辑
    ├── models/
    │   ├── wanva.py                  # 主模型 Wan22VisionModel + VAE 封装
    │   ├── wan.py                    # WanTransformer3D（30 层 3D 扩散 Transformer）
    │   ├── projector.py              # Qformer / MLP 投影器
    │   ├── scheduler.py              # Flow Matching 调度器
    │   ├── action.py                 # 动作编解码器
    │   └── backbone.py               # 视觉 backbone / DiT backbone（独立模块）
    └── utils/
        ├── args.py                   # 配置加载（OmegaConf）
        ├── data.py                   # 数据集加载、采样器、张量工具
        ├── metrics.py                # PSNR、SSIM、FID、Meter、Timer
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
| `projector.type` | `qformer` | 投影器类型（`mlp` 或 `qformer`） |
| `projector.num_token` | 2 | 压缩后的条件 token 数量 |
| `projector.hidden_dim` | 4096 | 投影器隐藏层维度 |
| `projector.output_align_dim` | 4096 | 输出维度（与 Transformer video_dim 对齐） |
| `projector.patch_size` | [1, 4, 4] | VAE 潜变量的 3D patch 尺寸 |
| `wanva.patch_size` | [1, 2, 2] | Transformer 的 3D patch 尺寸 |
| `wanva.num_attention_heads` | 24 | Transformer 注意力头数 |
| `data.frames` | 5 | 每个片段帧数 |
| `data.fps` | 10 | 目标帧率 |
| `data.image_size` | [512, 512] | 帧分辨率 |
| `train.learning_rate` | 3e-5 | 基础学习率 |
| `train.warmup_ratio` | 0.01 | 预热比例 |
| `train.decay` | 1e-3 | 权重衰减 |
| `train.gradient_accumulate_steps` | 2 | 梯度累积步数 |

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
