import heapq
import os
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from univam.models.backbone import VisionBackbone
from univam.utils.data import load_multi_datasets_form_json


load_dotenv()
# =========================
# 配置参数
# =========================
args = Namespace(
    frames=5,
    fps=10,
    image_size=[512, 512],
    type="lerobot",
)

print("Configuration:", args)

# =========================
# 1. 初始化 DINO backbone
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "vit_large_patch16_dinov3.lvd1689m"
local_ckpt = Path(os.environ.get("PRETRAINED_MODEL_PATH", "./models")) / "timm" / model_name / "pytorch_model.bin"

backbone = VisionBackbone(
    img_size=args.image_size,
    model_name=model_name,
    local_ckpt=local_ckpt,
    pretrained=False,
).to(device)

backbone.eval()
normalize = backbone.get_transforms()


# =========================
# 2. 数据预处理函数
# =========================
def preprocess_for_dino(videos, normalize):
    """
    将视频数据预处理为 DINO 可接受的格式
    videos: [B, T, C, H, W], 假设范围是 [-1, 1]
    返回: 归一化后的 videos [B, T, C, H, W]
    """
    # 先恢复到 [0, 1]
    videos = (videos + 1) / 2

    B, T, C, H, W = videos.shape
    videos = videos.reshape(B * T, C, H, W)

    # 应用 DINO 的归一化
    videos = torch.stack([normalize(frame) for frame in videos])
    videos = videos.reshape(B, T, C, H, W)

    return videos


# =========================
# 3. 特征编码函数
# =========================
def encode_first_frame(frames, backbone, device):
    """
    编码首帧为 DINO 特征
    frames: [B, C, H, W] 或 [C, H, W]
    返回: 归一化的特征向量 [B, D] 或 [D]
    """
    is_single = False
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)  # [1, C, H, W]
        is_single = True

    with torch.no_grad():
        feat = backbone(frames.to(device).float())

        # 如果输出是 feature map [B, D, H, W]，做 global pooling
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])  # [B, D]
        # 如果输出是 3 维 [B, N, D]，可能是 ViT 的 patch tokens
        elif feat.dim() == 3:
            feat = feat.mean(dim=1)  # [B, D] - 对所有 tokens 做平均

        # 确保 feat 是 2 维的 [B, D]
        if feat.dim() != 2:
            raise ValueError(f"Unexpected feature dimension: {feat.shape}")

        # L2 归一化
        feat = F.normalize(feat, dim=1)

    if is_single:
        return feat.squeeze(0)  # [D]
    else:
        return feat  # [B, D]


# =========================
# 4. Top-K 检索函数
# =========================
def retrieve_topk_dino(
    query_img,
    eval_dataloader,
    backbone,
    k=10,
    device="cuda",
):
    """
    使用 DINO 特征检索 top-k 相似视频

    Args:
        query_img: [T, C, H, W] - 查询视频（已归一化）
        eval_dataloader: 数据加载器
        backbone: DINO backbone
        k: 返回 top-k 结果
        device: 设备

    Returns:
        list of dict: 包含 dataset_index, similarity, video_raw
    """
    backbone.eval()

    # 编码查询图像的首帧
    query_feat = encode_first_frame(query_img[0], backbone, device)
    print(f"Query feature shape: {query_feat.shape}")

    # 确保 query_feat 是 1 维向量
    if query_feat.dim() != 1:
        raise ValueError(f"Query feature should be 1D, got shape: {query_feat.shape}")

    heap = []  # 最小堆，用于维护 top-k
    sample_idx = 0

    print(f"Searching top-{k} similar videos...")
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(eval_dataloader, desc="Retrieving")):
            videos_raw = data["videos"].to(device).float()  # 原始视频数据
            B = videos_raw.shape[0]

            # 预处理视频用于 DINO 编码
            videos_normalized = preprocess_for_dino(videos_raw, normalize)
            first_frames = videos_normalized[:, 0]  # [B, C, H, W]

            # 编码首帧
            feats = encode_first_frame(first_frames, backbone, device)

            # 调试信息（仅第一个批次）
            if batch_idx == 0:
                print("\nDebug info:")
                print(f"  First frames shape: {first_frames.shape}")
                print(f"  Features shape: {feats.shape}")
                print(f"  Query feature shape: {query_feat.shape}")

            # 确保 feats 是 2 维的 [B, D]
            if feats.dim() != 2:
                raise ValueError(f"Features should be 2D [B, D], got shape: {feats.shape}")

            # 计算余弦相似度 [B, D] x [D] -> [B]
            sims = torch.matmul(feats, query_feat)

            # 调试信息（仅第一个批次）
            if batch_idx == 0:
                print(f"  Similarities shape: {sims.shape}")
                print(f"  Sample similarities: {sims[: min(3, B)].tolist()}\n")

            # 更新 top-k 堆
            for i in range(B):
                # 确保 sim 是标量
                sim = sims[i].item() if sims[i].numel() == 1 else sims[i].mean().item()

                item = (
                    sim,
                    sample_idx + i,
                    videos_raw[i].cpu().clone(),  # 保存原始视频数据用于可视化
                )

                if len(heap) < k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)

            sample_idx += B

    # 按相似度降序排列
    results = sorted(heap, reverse=True)

    return [
        {
            "dataset_index": idx,
            "similarity": sim,
            "video_raw": video,  # 原始视频 [-1, 1]
        }
        for sim, idx, video in results
    ]


# =========================
# 5. 可视化函数
# =========================
def visualize_topk_results(query_img, results, save_path="topk_dino_results.png", show_query=True):
    """
    可视化检索结果

    Args:
        query_img: [T, C, H, W] - 查询视频（已归一化）
        results: 检索结果列表
        save_path: 保存路径
        show_query: 是否显示查询图像
    """
    k = len(results)
    n_cols = k + 1 if show_query else k

    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    col_idx = 0

    # 显示查询图像
    if show_query:
        query_frame = query_img[0]  # 首帧
        # 从归一化空间恢复到 [0, 1]
        # 假设 normalize 使用 ImageNet 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        query_frame = query_frame * std + mean
        query_frame = query_frame.clamp(0, 1)

        img_np = query_frame.permute(1, 2, 0).cpu().numpy()
        axes[col_idx].imshow(img_np)
        axes[col_idx].axis("off")
        axes[col_idx].set_title("Query Image", fontsize=12, fontweight="bold")
        axes[col_idx].add_patch(
            plt.Rectangle((0, 0), 1, 1, transform=axes[col_idx].transAxes, fill=False, edgecolor="red", linewidth=3)
        )
        col_idx += 1

    # 显示检索结果
    for i, item in enumerate(results):
        frame = item["video_raw"][0]  # 首帧，范围 [-1, 1]

        # 恢复到 [0, 1]
        frame = (frame + 1) / 2
        frame = frame.clamp(0, 1)

        img_np = frame.permute(1, 2, 0).cpu().numpy()

        axes[col_idx].imshow(img_np)
        axes[col_idx].axis("off")
        axes[col_idx].set_title(
            f"Rank {i + 1}\nIndex: {item['dataset_index']}\nSim: {item['similarity']:.4f}", fontsize=10
        )
        col_idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✓ Visualization saved to: {save_path}")


# =========================
# 6. 主流程
# =========================
def main():
    # 加载查询图像
    # image_path = "logs/sim/videos/397500/17.jpg"
    image_path = "tests/eval/57.jpg"
    print(f"Loading query image from: {image_path}")

    img = Image.open(image_path)
    width, height = img.size
    print(f"Original image size: {width} x {height}")

    # 裁剪图像
    cropped_img = img.crop((0, 512, 512 * 5, 1024))
    print(f"Cropped image size: {cropped_img.size}")

    # 转换为 tensor 并重排为 [T, C, H, W]
    img_tensor = ToTensor()(cropped_img)
    img_tensor = img_tensor.reshape(3, 512, 5, 512).permute(2, 0, 1, 3)
    print(f"Query video shape: {img_tensor.shape}")

    # 应用 DINO 归一化
    img_normalized = torch.stack([normalize(frame) for frame in img_tensor])
    print(f"Normalized query shape: {img_normalized.shape}")

    # 加载评估数据集
    print("Loading evaluation dataset...")
    eval_dataloader = load_multi_datasets_form_json(
        args,
        json_path="jsons/eval_debug.json",
        flip_p=0,
        local_batch_size=32,
        num_workers=16,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        make_single_dataset=True,
    )

    # 执行 top-k 检索
    topk_results = retrieve_topk_dino(
        img_normalized,
        eval_dataloader,
        backbone,
        k=10,
        device=device,
    )

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"Top-{len(topk_results)} Results:")
    print(f"{'=' * 60}")
    for i, result in enumerate(topk_results):
        print(f"Rank {i + 1}: Index={result['dataset_index']}, Similarity={result['similarity']:.4f}")
    print(f"{'=' * 60}\n")

    # 可视化结果
    visualize_topk_results(img_normalized, topk_results, save_path="topk_dino_results.png", show_query=True)


if __name__ == "__main__":
    main()
