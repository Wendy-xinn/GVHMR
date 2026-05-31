import os
import gc
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
from hmr4d.utils.preproc import VitPoseExtractor
from hmr4d.utils.preproc.vitfeat_extractor import get_batch

def process_vitpose(root_dir: Path, preproc_dir: Path, output_dir: Path, chunk_size: int = 256):
    """
    根据第一阶段骤生成的 preprocess_view.pt 文件，提取图像序列的 VitPose 2D 关键点。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Log.info("初始化 ViTPose Extractor...")
    extractor = VitPoseExtractor(tqdm_leave=True)

    # 直接查找第一阶段生成的所有预处理文件 (包含了精确的 imgname 和 bbx_xys)
    pt_files = list(preproc_dir.glob("view*/**/preprocess_view*.pt"))
    Log.info(f"共找到 {len(pt_files)} 个预处理文件待提取...")

    for pt_file in pt_files:
        view_name = pt_file.parent.parent.name  # view1 或 view3
        recording_name = pt_file.parent.name    # recording_20210907_S02_S01_01
        
        Log.info(f"\n========== 开始处理: {view_name} / {recording_name} ==========")
        
        save_dir = output_dir / view_name / recording_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "vitpose_kp2d.pt"
        
        if save_file.exists():
            Log.info(f"文件已存在，跳过处理: {save_file}")
            continue

        # 1. 加载第一步清洗好的数据
        data = torch.load(pt_file)
        img_paths = data["imgname"]
        
        # 确定需要提取哪些边界框 (View3 可能同时有 ego 和 exo)
        bbx_keys = [k for k in data.keys() if k.startswith("bbx_xys_")]
        if not bbx_keys:
            Log.warning(f"警告: {pt_file} 中未找到边界框数据，跳过！")
            continue

        # 存储最终结果的字典
        result_dict = {}

        # 2. 按 Chunk 分块处理以防爆内存
        for bbx_key in bbx_keys:
            Log.info(f"正在提取边界框: {bbx_key} ...")
            bbx_xys = data[bbx_key]
            
            # 兼容 torch.Tensor 或 numpy.ndarray
            if isinstance(bbx_xys, torch.Tensor):
                bbx_xys = bbx_xys.numpy()

            all_kp2d = []
            
            for start in tqdm(range(0, len(img_paths), chunk_size), desc="Chunks"):
                end = min(start + chunk_size, len(img_paths))
                
                imgs = []
                for p in img_paths[start:end]:
                    full_path = p if Path(p).is_absolute() else root_dir / p
                    im = cv2.imread(str(full_path))
                    if im is not None:
                        imgs.append(im[..., ::-1])  # BGR to RGB
                    else:
                        Log.warning(f"无法读取图像: {full_path}")
                        imgs.append(np.zeros((1080, 1920, 3), dtype=np.uint8)) 
                
                # 1. 堆叠为 NumPy 数组
                imgs_np = np.stack(imgs, axis=0)
                
                # 2. 确保边界框是 Tensor 类型
                chunk_bbx = torch.tensor(bbx_xys[start:end], dtype=torch.float32)

                try:
                    # 3. 关键修改：调用 get_batch 将 NumPy 图像和边界框转换为模型需要的 Tensor
                    imgs_t, bbx_ds = get_batch(imgs_np, chunk_bbx, img_ds=1.0, path_type="np")

                    # 4. 传入转换后的 Tensor
                    kp2d_chunk = extractor.extract(video_path=imgs_t, bbx_xys=bbx_ds)
                    all_kp2d.append(kp2d_chunk)
                    
                except Exception as e:
                    Log.error(f"处理 Chunk {start}:{end} 时出错: {e}")
                
                # 显式清理内存
                del imgs_np, chunk_bbx, imgs_t, bbx_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if all_kp2d:
                # 拼接所有的 chunk 并存入对应 key (如 kp2d_ego, kp2d_exo)
                kp2d_key = bbx_key.replace("bbx_xys", "kp2d")
                result_dict[kp2d_key] = torch.cat(all_kp2d, dim=0)

        # 3. 保存结果
        if result_dict:
            torch.save(result_dict, save_file)
            Log.info(f"成功保存关键点字典至: {save_file}")
            for k, v in result_dict.items():
                Log.info(f" - {k} 形状: {v.shape}")

if __name__ == "__main__":
    # 原始数据根目录
    ROOT_DIR = Path("/public/home/wenxin/egobody")
    # 你第一个脚本的输出目录 (包含预处理好的 .pt)
    PREPROC_DIR = Path("/public/home/wenxin/egobody/output") 
    # 本次 VitPose 提取的输出目录
    VITPOSE_OUT_DIR = Path("/public/home/wenxin/egobody/vitpose")

    process_vitpose(ROOT_DIR, PREPROC_DIR, VITPOSE_OUT_DIR, chunk_size=256)