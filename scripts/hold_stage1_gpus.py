#!/usr/bin/env python3
"""Keep the Stage-I GPUs reserved while the supervised training job is alive."""

from __future__ import annotations

import argparse
import signal
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib-per-gpu", type=float, default=16.0)
    parser.add_argument("--heartbeat-seconds", type=int, default=300)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit("No visible CUDA device; GPU guard was not started")

    keep_running = True

    def stop(_signum, _frame) -> None:
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    allocations = []
    requested = int(args.gib_per_gpu * 2**30)
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        # Leave at least 6 GiB for the actual Stage-I rank on each 4090.
        reserve = min(requested, max(0, free - 6 * 2**30))
        if reserve < 1 * 2**30:
            raise SystemExit(
                f"cuda:{index} has only {free / 2**30:.2f} GiB free; refusing to start guard"
            )
        allocation = torch.empty(reserve, dtype=torch.uint8, device=f"cuda:{index}")
        allocations.append(allocation)
        print(
            f"reserved {reserve / 2**30:.2f} GiB on cuda:{index}; "
            f"total={total / 2**30:.2f} GiB",
            flush=True,
        )

    while keep_running:
        print(f"GPU guard heartbeat: {len(allocations)} devices reserved", flush=True)
        time.sleep(args.heartbeat_seconds)


if __name__ == "__main__":
    main()
