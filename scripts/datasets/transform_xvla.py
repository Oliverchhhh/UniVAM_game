import io
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
from mmengine import fileio


# =========================
# 配置
# =========================
top_path = "./"
output_root = "../XVLA-Soft-Fold-Videos"
fps = 24
num_workers = max(1, multiprocessing.cpu_count() // 2)
# =========================


def process_one(hdf5_name):
    try:
        input_path = os.path.join(top_path, hdf5_name)

        # 构造输出路径（保持目录结构）
        relative_path = os.path.splitext(hdf5_name)[0] + ".mp4"
        output_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 读取 HDF5
        value = fileio.get(input_path)
        f = io.BytesIO(value)
        h = h5py.File(f, "r")

        images = h["/observations/images/cam_high"]
        ep_len = len(images)

        if ep_len == 0:
            return hdf5_name, 0

        # 读取第一帧确定尺寸
        first_img = cv2.imdecode(images[0], cv2.IMREAD_COLOR)
        height, width = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(ep_len):
            img = cv2.imdecode(images[i], cv2.IMREAD_COLOR)
            video_writer.write(img)

        video_writer.release()
        h.close()

        return hdf5_name, ep_len

    except Exception as e:
        return hdf5_name, f"ERROR: {str(e)}"


def main():
    hdf5_files = list(fileio.list_dir_or_file(top_path, suffix=".hdf5", recursive=True, list_dir=False))

    total = len(hdf5_files)
    print(f"Total HDF5 files found: {total}")
    print(f"Using {num_workers} workers\n")

    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, name) for name in hdf5_files]

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            print(f"[{completed}/{total}] Finished: {result[0]}  ->  {result[1]}")

    print("\n🎉 All videos generated successfully!")


if __name__ == "__main__":
    main()
