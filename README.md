# UniVAM

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
