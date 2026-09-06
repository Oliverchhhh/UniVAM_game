"""Evaluate a Stage-I checkpoint with correct/shuffled/zero future conditions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import CupheadFutureIterableDataset
from .model import Stage1FutureConditionModel
from .train import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Defaults to config stored in checkpoint")
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = yaml.safe_load(os.path.expandvars(Path(args.config).read_text())) if args.config else state["config"]
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    data_cfg = config["data"]
    dataset = CupheadFutureIterableDataset(
        data_cfg["root"], "val", stride=int(data_cfg.get("val_stride", 17)),
        video_name=data_cfg.get("video_name", "256x256.mp4"), seed=int(config.get("seed", 43)), repeat=True,
    )
    loader = DataLoader(
        dataset, batch_size=int(config["train"].get("micro_batch_size", 1)),
        num_workers=int(data_cfg.get("workers", 2)), pin_memory=True,
    )
    model_cfg = config["model"]
    model = Stage1FutureConditionModel(
        nitrogen_checkpoint=model_cfg["nitrogen_checkpoint"], nitrogen_root=model_cfg["nitrogen_root"],
        siglip_path=model_cfg["siglip_path"], wog_root=model_cfg["wog_root"], device=device,
        dtype=model_cfg.get("dtype", "bfloat16"), condition_dim=int(model_cfg.get("condition_dim", 64)),
        qformer_hidden=int(model_cfg.get("qformer_hidden", 1024)), qformer_layers=int(model_cfg.get("qformer_layers", 6)),
        adapter_dim=int(model_cfg.get("adapter_dim", 256)), condition_dropout=0.0, gradient_checkpointing=False,
    )
    model.load_trainable_state_dict(state["model"])
    metrics = validate(model, loader, device, args.batches)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": int(state.get("step", -1)),
        **metrics,
        "gain_vs_zero": metrics["zero"] - metrics["correct"],
        "gain_vs_shuffled": metrics["shuffled"] - metrics["correct"],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
