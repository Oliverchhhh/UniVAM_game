# NitroGen + WoG Future Condition：第一阶段训练

## 目标与数据流

第一阶段不训练 YOLO，也不复用 UniVAM/V-JEPA2 特征。每个样本读取当前帧和
`t+4/t+8/t+12/t+17` 四个未来关键帧：冻结的 DINOv2 编码未来帧，冻结的
Wan-VAE 编码 5 帧短视频，Q-Former 用 18 个 query 压缩为 `18x64` condition。
condition 经四个可训练 cross-attention adapter 注入冻结的 NitroGen Action-DiT，
由 `t+1...t+18` 的真实手柄动作提供 masked flow-matching 监督。训练参数只有
Q-Former 和 adapter（约 78.7M）。

## HF 资产内容

`ch1415926/cuphead-action` 包含：

- 4,793 个 chunk 的 `256x256.mp4` 与 `annotation.proto`，按源视频打成 39 个 tar；
- NitroGen `ng.pt`；
- NitroGen 使用的 SigLIP2 本地模型；
- WoG 使用的 DINOv2 与 Wan2.1-VAE 权重；
- SHA-256 校验清单。

原始数据中的 `video.mp4`、`192x192.mp4`、`gamepad.mp4`、UniVAM flow 和
V-JEPA2 特征没有被上传，因为本阶段不读取它们。

## 双 RTX 5090D 快速部署

```bash
git clone -b wog git@github.com:Oliverchhhh/UniVAM_game.git
cd UniVAM_game

# 可选：本地代理
export HTTP_PROXY=http://127.0.0.1:10090
export HTTPS_PROXY=$HTTP_PROXY

bash scripts/setup_stage1_conda.sh
conda activate nitrogen-stage1
hf auth login                         # 私有数据集需要
bash scripts/prepare_cuphead_action_assets.sh
```

先运行双卡真实 forward/backward：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dryrun_stage1_2x5090d.sh
```

dry-run 通过后，在 tmux 中启动带保卡、卡死检测和 checkpoint 自动恢复的训练：

```bash
tmux new -d -s cuphead-stage1 \
  "bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh; \
  conda activate nitrogen-stage1; cd /root/UniVAM_game; \
  PROXY_PORT=10090 PHYSICAL_GPUS=0,1 \
  bash scripts/supervise_stage1_2x5090d.sh'"

tmux attach -t cuphead-stage1
```

若 Miniconda 不在 `/root/miniconda3`，只需把 tmux 命令里的 activation 路径改成
服务器上的实际位置。正式训练默认输出到
`/root/stage1-runs/stage1_future_condition_2x5090d`。

## 监控与恢复

```bash
tail -f /root/stage1-runs/stage1_training_supervised.log
tail -f /root/stage1-runs/stage1_supervisor.log
nvidia-smi
```

监督脚本每 30 秒检查训练，10 分钟没有 metric 更新则重启；普通异常退出也会从
`latest.pt` 自动恢复。连续失败超过 3 次后训练不再循环，但 GPU guard 会继续保卡。
代码已经调整为先写 checkpoint 再执行定期验证。

## 第一阶段验收

```bash
python -m stage1_future_condition.evaluate \
  --config configs/stage1_future_condition_2x5090d.yaml \
  --checkpoint /root/stage1-runs/stage1_future_condition_2x5090d/final.pt \
  --batches 256 \
  --output /root/stage1-runs/stage1_future_condition_2x5090d/eval.json
```

正确未来条件的 loss 应稳定低于 shuffled 和 zero condition。只有满足这一点，才说明
Q-Former 学到了动作相关的未来表征，适合进入第二阶段 UniVAM co-train。

## 5090D 资源预估

原双 4090、micro-batch 1 实测每卡 PyTorch 峰值约 3.84GiB、速度约 1.0 optimizer
step/s。5090D 默认 micro-batch 2、梯度累计 4 次，保持 global effective batch 16；
预计训练本体低于 10GiB/卡。50,000 步粗略预计 8–14 小时，最终以 dry-run 后前
1,000 步的实测速率为准。
