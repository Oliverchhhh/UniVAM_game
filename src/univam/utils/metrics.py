import os
from datetime import timedelta
from timeit import default_timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import Normalize, Resize


class InceptionV3Features(nn.Module):
    def __init__(self, weights, device="cpu"):
        super().__init__()
        self.device = device

        # pretrained inception
        self.model = inception_v3(weights=None, transform_input=False, init_weights=True)
        state_dict = torch.load(weights, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Identity()
        self.model.dropout = nn.Identity()
        self.model.eval().to(device)

        # InceptionV3 expects 299x299 RGB
        self.resize = Resize((299, 299))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def forward(self, x):
        """
        x: tensor [B, C, H, W]  ∈ [0,1]
        return: Inception 特征 [B, 2048]
        """
        if x.shape[1] == 1:
            x = x.repeat(3, 1, 1, 1)  # 灰度图转3通道

        x = self.resize(x)
        x = self.normalize(x)

        return self.model(x)


def matrix_sqrt(A, eps=1e-6):
    # Singular Value Decomposition (SVD)
    U, S, V = torch.svd(A)
    S_sqrt = torch.diag(torch.sqrt(S + eps))
    return U @ S_sqrt @ V.T


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    calculate Frechet Distance (FID)
    """
    diff = mu1 - mu2

    covmean = matrix_sqrt(sigma1 @ sigma2, eps=eps)

    # FID
    fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()


def calculate_rfid(pred, target, device="cpu"):
    """
    pred, target: [B, T, C, H, W], 取值范围 [0,1]
    计算 video 重建的 FID
    """

    assert pred.shape == target.shape
    B, T, C, H, W = pred.shape

    # 展平成图像 batch
    pred = pred.reshape(B * T, C, H, W)
    target = target.reshape(B * T, C, H, W)

    weights = os.environ.get("PRETRAINED_MODEL_PATH", "..") + "/inception_v3/inception_v3.pth"
    extractor = InceptionV3Features(weights=weights, device=device)

    dtype = next(extractor.parameters()).dtype

    pred = pred.to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)

    with torch.no_grad():
        feat_pred = extractor(pred)  # [B*T, 2048]
        feat_gt = extractor(target)

    # 计算均值和协方差
    mu1 = torch.mean(feat_pred, dim=0)
    mu2 = torch.mean(feat_gt, dim=0)

    sigma1 = torch.cov(feat_pred.T)
    sigma2 = torch.cov(feat_gt.T)

    fid = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calculate_psnr(pred, target, max_val=1.0):
    """
    pred, target: [B, T, C, H, W]
    """

    assert pred.shape == target.shape
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)

    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=[2, 3, 4])  # 每帧 MSE  -> [B, T]

    psnr = 10 * torch.log10((max_val**2) / (mse + 1e-8))

    return psnr.mean()


def calculate_ssim(pred, target, max_val=1.0, window_size=11, K1=0.01, K2=0.03):
    """
    pred, target: [B, T, C, H, W]
    """

    assert pred.shape == target.shape
    B, T, C, H, W = pred.shape

    # 展平成图像 batch
    pred = pred.reshape(B * T, C, H, W)
    target = target.reshape(B * T, C, H, W)

    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    device = pred.device

    def gaussian_window(window_size, sigma):
        gauss = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        gauss = torch.exp(-(gauss**2) / (2 * sigma**2))
        return gauss / gauss.sum()

    sigma = 1.5
    gauss_1d = gaussian_window(window_size, sigma).unsqueeze(1)
    window_2d = gauss_1d @ gauss_1d.T
    window = window_2d.expand(C, 1, window_size, window_size).to(device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def get_parameters(net: torch.nn.Module):
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    fp32_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and p.requires_grad)
    fp16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and p.requires_grad)
    bf16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.bfloat16 and p.requires_grad)
    fp32_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and not p.requires_grad)
    fp16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and not p.requires_grad)
    bf16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.bfloat16 and not p.requires_grad)
    return {
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_fp32": fp32_trainable_params,
        "trainalbe_fp16": fp16_trainable_params,
        "trainalbe_bf16": bf16_trainable_params,
        "frozen_fp32": fp32_frozen_params,
        "frozen_fp16": fp16_frozen_params,
        "frozen_bf16": bf16_frozen_params,
    }


class Meter:
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n: int = 1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        if isinstance(val, (int, float)):
            self._update_scalar(val, n)
        elif isinstance(val, dict):
            self._update_dict(val, n)
        else:
            raise ValueError(f"Not supported type {type(val)}")

    def _update_scalar(self, val: float, n: int):
        self.val = val
        self.sum = self.sum + val * n if self.sum is not None else val * n
        self.count = self.count + n if self.count is not None else n
        self.avg = self.sum / self.count

    def _update_dict(self, val: dict, n: int):
        # tensor -> item
        val = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in val.items()}

        self.val = {**self.val, **val} if self.val is not None else val

        if self.sum is not None:
            for k, v in val.items():
                self.sum[k] = self.sum.get(k, 0) + v * n
        else:
            self.sum = {k: v * n for k, v in val.items()}

        if self.count is not None:
            for k in val.keys():
                self.count[k] = self.count.get(k, 0) + n
        else:
            self.count = dict.fromkeys(val.keys(), n)

        self.avg = {k: self.sum[k] / self.count[k] for k in self.count.keys()}

    def __str__(self):
        if isinstance(self.avg, dict):
            return str({k: f"{v:.4f}" for k, v in self.avg.items()})
        return "Nan"


class Timer:
    def __init__(self):
        """
        t = Timer()
        time.sleep(1)
        print(t.elapse())
        """
        self.start = default_timer()

    def elapse(self, readable=False):
        seconds = default_timer() - self.start
        if readable:
            seconds = str(timedelta(seconds=seconds))
        return seconds


if __name__ == "__main__":
    pred = torch.rand(8, 3, 256, 256)
    target = torch.rand(8, 3, 256, 256)

    print(f"PSNR: {calculate_psnr(pred, target):.4f}")
    print(f"SSIM: {calculate_ssim(pred, target):.4f}")
    print(f"rFID: {calculate_rfid(pred, target):.4f}")
