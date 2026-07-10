import torch
import sys
import os
sys.path.insert(0, './src')

from univam.models.wan import FluxWanVAE
from univam.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

def test_flux_wan_vae():
    # 修改这两个路径为你的实际路径
    flux_model_path = "black-forest-labs/FLUX.1-schnell"
    wan_model_path = os.environ.get("PRETRAINED_MODEL_PATH", ".") + "/Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    overwatch.info("=" * 80)
    overwatch.info("Testing FluxWanVAE")
    overwatch.info("=" * 80)

    overwatch.info("Initializing FluxWanVAE...")
    vae = FluxWanVAE(flux_model_path, wan_model_path, frames=5)
    vae = vae.to("cuda", dtype=torch.bfloat16)

    # 测试视频
    overwatch.info("Creating test video...")
    videos = torch.randn(2, 5, 3, 256, 256).to("cuda", dtype=torch.bfloat16)

    overwatch.info(f"Input video shape: {videos.shape}")

    # 编码
    overwatch.info("Encoding...")
    vae.eval()
    with torch.no_grad():
        latents = vae.encode(videos)
    overwatch.info(f"Latent shape: {latents.shape}")
    overwatch.info(f"Expected shape: [2, 48, 2, 16, 16]")

    # 解码
    overwatch.info("Decoding...")
    with torch.no_grad():
        reconstructed = vae.decode(latents)
    overwatch.info(f"Reconstructed video shape: {reconstructed.shape}")
    overwatch.info(f"Expected shape: [2, 5, 3, 256, 256]")

    # 重建误差
    mse = torch.mean((videos - reconstructed) ** 2)
    overwatch.info(f"Reconstruction MSE: {mse.item():.6f}")

    # 检查可训练参数
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    overwatch.info(f"Total params: {total_params:,}")
    overwatch.info(f"Trainable params (extra_encoder_layers): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # 检查各部分参数
    flux_enc_params = sum(p.numel() for p in vae.flux_encoder.parameters())
    wan_dec_params = sum(p.numel() for p in vae.wan_decoder.parameters())
    extra_params = sum(p.numel() for p in vae.extra_encoder_layers.parameters())

    overwatch.info(f"Flux encoder params (frozen): {flux_enc_params:,}")
    overwatch.info(f"Wan decoder params (frozen): {wan_dec_params:,}")
    overwatch.info(f"Extra encoder layers params (trainable): {extra_params:,}")

    overwatch.info("=" * 80)
    overwatch.info("✅ Test completed successfully!")
    overwatch.info("=" * 80)

if __name__ == "__main__":
    test_flux_wan_vae()
