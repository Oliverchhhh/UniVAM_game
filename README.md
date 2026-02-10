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

3. **Install the torchcodec**

follow the installation instructions provided in the official [torchcodec repository](https://github.com/meta-pytorch/torchcodec/tree/main?tab=readme-ov-file#installing-torchcodec)

