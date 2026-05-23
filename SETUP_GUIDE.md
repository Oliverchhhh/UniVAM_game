# UniVAM 完整部署与训练指南

本文档提供从零开始部署 UniVAM 项目的完整流程，包括环境配置、模型下载、数据集准备、视频提取、数据集划分以及恢复训练。

---

## 步骤 1: 克隆项目

```bash
git clone https://github.com/hello3x3/UniVAM.git
cd UniVAM
```

---

## 步骤 2: 配置环境

### 2.1 创建 Conda 环境

```bash
conda create -n univam python=3.10 -y
conda activate univam
```

### 2.2 安装项目依赖

```bash
pip install -e .
```

### 2.3 安装 flash-attn

```bash
MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation
```

### 2.4 安装 ModelScope 和 HuggingFace CLI

```bash
pip install modelscope
pip install huggingface_hub
```

### 2.5 登录 HuggingFace

下载数据集和预训练权重需要登录 HuggingFace：

```bash
huggingface-cli login
```

按提示输入你的 HuggingFace Access Token（可在 https://huggingface.co/settings/tokens 创建）。

### 2.6 配置 HuggingFace 镜像（国内网络）

如果在国内网络环境下，建议配置 HuggingFace 镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

可以将此行添加到 `~/.bashrc` 或 `~/.zshrc` 中使其永久生效。

### 2.7 生成 .env 配置文件

```bash
bash scripts/envs/generate_dotenv.sh
```

此命令会在项目根目录生成 `.env` 文件，然后编辑该文件设置正确的路径：

```bash
# sdpa | flash_attention_2
ATTN_MODE=sdpa
PRETRAINED_MODEL_PATH=/your/path/to/pretrained_models    # 预训练模型存放路径
DATASETS_PATH=/your/path/to/datasets                      # 数据集存放路径
CHECK_TENSOR=0

# eval part
EVAL_JSON_PATH=./jsons/debug.json
RESUME_PATH=./ckpt/model/
```

> **注意**：`PRETRAINED_MODEL_PATH` 和 `DATASETS_PATH` 需要替换为你本机的实际路径。后续步骤中视频提取脚本会自动从 `.env` 中读取 `DATASETS_PATH`。

---

## 步骤 3: 下载模型

### 3.1 下载 Wan2.2 基础模型（通过 ModelScope）

```bash
python download_models.py
```

此脚本通过 ModelScope 下载 `Wan-AI/Wan2.2-TI2V-5B-Diffusers`（包含 VAE 和 Transformer 权重）到 `$PRETRAINED_MODEL_PATH` 目录下。

### 3.2 下载预训练权重（通过 HuggingFace）

项目提供了三个任务的预训练权重，托管在 HuggingFace 仓库 `STA-I-R/video_stamo` 中。使用 `huggingface-cli` 下载：

```bash
# 下载 libero 任务的预训练权重（110k_2 步）
huggingface-cli download STA-I-R/video_stamo libero/110k_2 \
    --local-dir ./ckpts/libero/110k_2 \
    --repo-type model

# 下载 libero 任务的配置文件
huggingface-cli download STA-I-R/video_stamo libero/config.yaml \
    --local-dir ./ckpts/libero \
    --repo-type model

# 下载 fractal 任务的预训练权重（76000 步）
huggingface-cli download STA-I-R/video_stamo fractal/76000 \
    --local-dir ./ckpts/fractal/76000 \
    --repo-type model

# 下载 fractal 任务的配置文件
huggingface-cli download STA-I-R/video_stamo fractal/config.yaml \
    --local-dir ./ckpts/fractal \
    --repo-type model

# 下载 bridgev2 任务的预训练权重（137k 步）
huggingface-cli download STA-I-R/video_stamo bridgev2/137k \
    --local-dir ./ckpts/bridgev2/137k \
    --repo-type model

# 下载 bridgev2 任务的配置文件
huggingface-cli download STA-I-R/video_stamo bridgev2/config.yaml \
    --local-dir ./ckpts/bridgev2 \
    --repo-type model
```

下载完成后，`ckpts/` 目录结构应如下：

```
ckpts/
├── libero/
│   ├── config.yaml
│   └── 110k_2/
│       ├── Wan22VM.pth        # Transformer 权重
│       ├── LoraAdapter.pth    # LoRA 权重
│       └── Projector.pth      # Projector 权重
├── fractal/
│   ├── config.yaml
│   └── 76000/
│       ├── Wan22VM.pth
│       ├── LoraAdapter.pth
│       └── Projector.pth
└── bridgev2/
    ├── config.yaml
    └── 137k/
        ├── Wan22VM.pth
        ├── LoraAdapter.pth
        └── Projector.pth
```

---

## 步骤 4: 下载数据集

```bash
python download_datasets.py
```

此脚本通过 HuggingFace Hub 下载以下三个数据集到 `$DATASETS_PATH` 目录下：

| 数据集 | HuggingFace Repo ID | 格式 |
|--------|---------------------|------|
| LIBERO | `physical-intelligence/libero` | LeRobot v2.0 |
| Bridge-V2 | `ericonaldo/Bridge-V2` | RLDS (tfrecord) |
| Fractal20220817 | `ucasmichael/fractal20220817_data` | RLDS (tfrecord) |

> 如果设置了 `HF_ENDPOINT=https://hf-mirror.com`，脚本会自动使用镜像站下载。

---

## 步骤 5: 提取视频

下载的数据集为原始格式（LeRobot v2.0 / RLDS），需要分别提取为 MP4 视频文件才能用于训练。以下三个脚本均使用默认参数，自动从 `.env` 中读取 `DATASETS_PATH`。

### 5.1 提取 LIBERO 视频（LeRobot v2.0 格式）

```bash
python scripts/datasets/extract_lerobot_v2_videos.py
```

默认参数：
- 数据集路径：`$DATASETS_PATH/physical-intelligence/libero`
- 输出路径：`$DATASETS_PATH/physical-intelligence/libero_videos`
- 相机 key：`image`
- 帧率：使用数据集原始帧率

### 5.2 提取 Bridge-V2 视频（RLDS 格式）

Bridge-V2 需要分别提取 train 和 eval：

```bash
python scripts/datasets/extract_rlds_bridge_v2_videos.py --split train
python scripts/datasets/extract_rlds_bridge_v2_videos.py --split val
```

默认参数：
- 数据集路径：`$DATASETS_PATH/ericonaldo/Bridge-V2/bridge_v2/0.1.0`
- 输出路径：`$DATASETS_PATH/ericonaldo/bridge_v2_videos`
- 相机 key：`image_0`
- 帧率：5 fps

### 5.3 提取 Fractal 视频（RLDS 格式）

Fractal 只需提取 train：

```bash
python scripts/datasets/extract_rlds_fractal_videos.py
```

默认参数：
- 数据集路径：`$DATASETS_PATH/ucasmichael/fractal20220817_data/fractal20220817_data/0.1.0`
- 输出路径：`$DATASETS_PATH/ucasmichael/fractal_videos`
- 相机 key：`image`
- 帧率：5 fps
- 划分：train

### 5.4 视频提取结果

提取完成后，各数据集的视频目录结构如下：

```
$DATASETS_PATH/physical-intelligence/libero_videos/
└── libero/                          # LeRobot v2.0 输出的子目录名与数据集同名
    ├── episode_000000.mp4
    ├── episode_000001.mp4
    └── ...

$DATASETS_PATH/ericonaldo/bridge_v2_videos/
├── train/
│   ├── episode_000000.mp4
│   └── ...
└── val/
    ├── episode_000000.mp4
    └── ...

$DATASETS_PATH/ucasmichael/fractal_videos/
└── train/
    ├── episode_000000.mp4
    └── ...
```

---

## 步骤 6: 划分数据集并生成 JSON 配置

`create_video_jsons.py` 要求 `eval_video_dirs` 中的目录只包含极少量视频（由 `eval_num` 控制）。由于提取后的 eval 目录包含大量视频，需要先将其裁剪为每个目录仅保留 1 个视频文件，多余的视频放回 train 目录。

### 6.1 准备 eval 目录

对三个数据集分别执行以下操作，每个 eval 目录只保留 1 个视频，多余的移入 train：

```bash
DATASETS_PATH=$(grep DATASETS_PATH .env | cut -d= -f2)

# --- LIBERO ---
# libero 的视频在 libero_videos/libero/ 下，没有 train/eval 子目录
# 需要手动创建 train 和 eval 子目录
mkdir -p $DATASETS_PATH/physical-intelligence/libero_videos/train
mkdir -p $DATASETS_PATH/physical-intelligence/libero_videos/eval

# 将所有视频移到 train，只留 1 个在 eval
cd $DATASETS_PATH/physical-intelligence/libero_videos/libero
ls *.mp4 | tail -n +2 | xargs -I{} mv {} ../train/
mv $(ls *.mp4 | head -1) ../eval/
cd -

# --- Bridge-V2 ---
# 将 val 目录重命名为 eval，只留 1 个视频，多余的移到 train
cd $DATASETS_PATH/ericonaldo/bridge_v2_videos
mv val eval
cd eval
ls *.mp4 | tail -n +2 | xargs -I{} mv {} ../train/
cd -

# --- Fractal ---
# 从 train 中移出 1 个视频作为 eval
mkdir -p $DATASETS_PATH/ucasmichael/fractal_videos/eval
cd $DATASETS_PATH/ucasmichael/fractal_videos/train
mv $(ls *.mp4 | tail -1) ../eval/
cd -
```

处理完成后，目录结构应如下：

```
$DATASETS_PATH/physical-intelligence/libero_videos/
├── train/
│   ├── episode_000001.mp4
│   ├── episode_000002.mp4
│   └── ...
└── eval/
    └── episode_000000.mp4          # 仅 1 个视频

$DATASETS_PATH/ericonaldo/bridge_v2_videos/
├── train/
│   ├── episode_000000.mp4
│   └── ...
└── eval/
    └── episode_000000.mp4          # 仅 1 个视频

$DATASETS_PATH/ucasmichael/fractal_videos/
├── train/
│   ├── episode_000000.mp4
│   └── ...
└── eval/
    └── episode_XXXXXX.mp4          # 仅 1 个视频
```

### 6.2 生成 JSON 配置（每个数据集单独运行）

`create_video_jsons.py` 需要针对每个数据集分别运行。每次运行前修改 `__main__` 部分的目录和数据集名称。

**LIBERO** — 编辑 `scripts/datasets/create_video_jsons.py` 的 `__main__` 部分：

```python
if __name__ == "__main__":
    load_dotenv()
    dataset_path = Path(os.environ.get("DATASETS_PATH", "./datasets"))

    train_video_dirs = [
        dataset_path / "physical-intelligence/libero_videos/train",
    ]
    eval_video_dirs = [
        dataset_path / "physical-intelligence/libero_videos/eval",
    ]
    create_split_jsonl(train_video_dirs, eval_video_dirs, "libero", shared_train_num=1, eval_num=1)
```

```bash
python scripts/datasets/create_video_jsons.py
```

**Bridge-V2** — 修改为：

```python
    train_video_dirs = [
        dataset_path / "ericonaldo/bridge_v2_videos/train",
    ]
    eval_video_dirs = [
        dataset_path / "ericonaldo/bridge_v2_videos/eval",
    ]
    create_split_jsonl(train_video_dirs, eval_video_dirs, "bridgev2", shared_train_num=1, eval_num=1)
```

```bash
python scripts/datasets/create_video_jsons.py
```

**Fractal** — 修改为：

```python
    train_video_dirs = [
        dataset_path / "ucasmichael/fractal_videos/train",
    ]
    eval_video_dirs = [
        dataset_path / "ucasmichael/fractal_videos/eval",
    ]
    create_split_jsonl(train_video_dirs, eval_video_dirs, "fractal", shared_train_num=1, eval_num=1)
```

```bash
python scripts/datasets/create_video_jsons.py
```

### 6.3 生成的文件结构

每次运行后会在 `./jsons/` 目录下生成对应数据集的配置文件。三个数据集全部运行后：

```
jsons/
├── train_libero_part_0.jsonl       # LIBERO 训练集视频路径
├── eval_libero.jsonl               # LIBERO 评估集视频路径
├── train_libero.json               # LIBERO 训练配置
├── eval_libero.json                # LIBERO 评估配置
├── train_bridgev2_part_0.jsonl     # Bridge-V2 训练集视频路径
├── eval_bridgev2.jsonl             # Bridge-V2 评估集视频路径
├── train_bridgev2.json             # Bridge-V2 训练配置
├── eval_bridgev2.json              # Bridge-V2 评估配置
├── train_fractal_part_0.jsonl      # Fractal 训练集视频路径
├── eval_fractal.jsonl              # Fractal 评估集视频路径
├── train_fractal.json              # Fractal 训练配置
└── eval_fractal.json               # Fractal 评估配置
```

JSONL 每行格式：
```json
{"video": "/absolute/path/to/video.mp4"}
```

---

## 步骤 7: 恢复训练

### 7.1 修改训练配置文件

以 libero 为例，编辑 `configs/libero.yaml`，确认以下配置：

```yaml
resume: True
resume_path: ckpts/libero/110k_2    # 指向下载的预训练权重目录

wanva:
  model_path: ${oc.env:PRETRAINED_MODEL_PATH}/Wan-AI/Wan2.2-TI2V-5B-Diffusers/
  # 确保此路径与 .env 中的 PRETRAINED_MODEL_PATH 一致

data:
  type: video
  train_json_path: ./jsons/train_libero.json
  eval_json_path: ./jsons/eval_libero.json
```

其他可调参数：

| 参数 | 说明 | libero 默认值 |
|------|------|--------------|
| `train.learning_rate` | 学习率 | 3e-5 |
| `train.local_batch_size` | 每 GPU batch size | 32 |
| `train.eval_step` | 评估间隔（步） | 1000 |
| `train.save_step` | 保存间隔（步） | 1000 |
| `train.reset_global_step` | resume 时是否重置步数为 0 | False |
| `lora.enable` | 是否使用 LoRA | True |
| `lora.r` | LoRA 低秩维度 | 16 |

三个任务的参考配置：

| 任务 | 配置文件 | resume_path | 学习率 | batch_size | GPU 数 |
|------|---------|-------------|--------|------------|--------|
| LIBERO | `configs/libero.yaml` | `ckpts/libero/110k_2` | 3e-5 | 32 | 2 |
| Fractal | `configs/fractal.yaml` | `ckpts/fractal/76000` | 2e-4 | 32 | 4 |
| Bridge-V2 | `configs/bridgev2.yaml` | `ckpts/bridgev2/137k` | 2e-4 | 32 | 4 |

### 7.2 启动单机训练

```bash
# LIBERO 训练（2 GPU）
bash scripts/train/train_libero.sh

# Fractal 训练（4 GPU）
bash scripts/train/train_fractal.sh

# Bridge-V2 训练（4 GPU）
bash scripts/train/train_bridgev2.sh
```

训练日志会输出到项目根目录（如 `train_libero.log`），检查点保存在 `ckpts/<task_name>/` 下。

### 7.3 启动多机训练

参考 `scripts/train/train_multi.sh`，多机训练需要设置环境变量并使用 `accelerate launch`：

```bash
# 在每台机器上设置以下环境变量
export MASTER_ADDR=<主节点IP>      # 例如 192.168.1.1
export MASTER_PORT=<主节点端口>    # 主节点端口
export WORLD_SIZE=<机器总数>       # 例如 2
export RANK=<当前机器编号>         # 主节点为 0，其他依次为 1, 2, ...

# 启动训练（每台机器上都需要执行）
accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --num_machines $WORLD_SIZE \
    --machine_rank $RANK \
    train.py \
    --config_path configs/libero.yaml \
    2>&1 | tee train_libero_node${RANK}.log
```

> 注意：多机训练时，`--num_processes` 在 `zero2.yaml` 中已配置（默认 8），表示每台机器的 GPU 数。所有机器的配置和数据路径需保持一致。

### 7.4 检查点说明

训练过程中保存的检查点包含以下文件：

| 文件 | 说明 |
|------|------|
| `Wan22VM.pth` | Transformer 完整权重（LoRA 模式下为合并后的权重） |
| `LoraAdapter.pth` | LoRA 低秩矩阵 + modules_to_save 权重 |
| `Projector.pth` | Projector 权重 |

### 7.5 Resume 机制

- 当 `resume: True` 时，训练脚本会从 `resume_path` 加载 `Wan22VM.pth`、`LoraAdapter.pth`、`Projector.pth`
- 如果 `resume_path` 下存在 `train_state/` 子目录，还会恢复优化器和调度器状态
- 设置 `reset_global_step: True` 可以在加载权重后将训练步数重置为 0

---

## 常见问题

### Q: 如何调整 GPU 数量？

修改对应训练脚本中的 `--num_processes` 和 `CUDA_VISIBLE_DEVICES`。例如使用 4 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate/zero2.yaml \
    --num_processes 4 \
    train.py \
    --config_path configs/libero.yaml
```

### Q: 单卡显存不足怎么办？

将训练脚本中的 `configs/accelerate/zero2.yaml` 换成 `configs/accelerate/zero3.yaml`，使用 ZeRO-3 策略分片参数以降低显存占用。例如：

```bash
accelerate launch \
    --config_file configs/accelerate/zero3.yaml \
    --num_processes 1 \
    train.py \
    --config_path configs/libero.yaml
```

### Q: 如何从头训练（不加载预训练权重）？

在配置文件中设置：
```yaml
resume: False
resume_path: null
```

### Q: 如何评估模型？

```bash
python eval.py --config_path configs/libero.yaml
```

评估结果（PSNR/SSIM）会输出到日志，预测帧与真实帧的对比图会保存到指定目录。
