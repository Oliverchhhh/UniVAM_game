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
   ```

3. **Test for NPU**:
   ```bash
   python scripts/test_for_npu.py
   ```

4. **Install the decord**
   ```bash
   #!/bin/bash

   # build tools
   yum install -y autoconf automake bzip2 bzip2-devel freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel cmake

   # workspace
   mkdir ~/ffmpeg_sources

   # nasm
   yum install -y nasm

   # yasm
   cd ~/ffmpeg_sources
   curl -O -L https://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
   tar xzf yasm-1.3.0.tar.gz
   cd yasm-1.3.0
   ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin"
   make -j$(nproc)
   make install

   # libx264
   yum install x264 x264-devel -y

   # libvpx
   yum install libvpx libvpx-devel -y

   # ffmpeg
   yum install -y ffmpeg ffmpeg-devel

   # build libs
   ls ~/ffmpeg_build/lib

   # decord
   git clone --recursive https://github.com/dmlc/decord
   cd decord
   mkdir build && cd build
   cmake .. -DUSE_CUDA=0
   make -j$(nproc)
   cp libdecord.so /usr/local/lib/
   ldconfig

   cd ../python
   python setup.py install
   python -c "import decord; print('✅ successfully import decord')"

   ```
