# my_pipelines.py
import os.path as osp
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiFrameImageFromFile:
    """
    自定义数据加载器：
    读取 data_infos['frame_paths'] 中的5个路径，堆叠成 15 通道
    """

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        # 1. 从 Dataset 传来的 results 里获取 5 个路径
        if 'frame_paths' not in results['img_info']:
            raise KeyError("results['img_info'] 中没有 'frame_paths'，请检查 MFDataset 是否正确生成了路径列表。")

        paths = results['img_info']['frame_paths']

        # 2. 依次读取图片
        imgs = []
        for p in paths:
            # 拼接完整路径 (img_prefix + filename)
            # 注意：如果 Dataset 里已经拼好了绝对路径，这里可能需要调整
            # 你的 MFDataset 里用的是 osp.join(dir_name, ...)，通常是相对路径或绝对路径
            # MMDetection 会自动把 img_prefix 和 filename 结合，这里我们直接用 Dataset 给的路径试试

            # 如果 results['img_prefix'] 不为空，且路径不是绝对路径，需要拼接
            if results['img_prefix'] is not None and not osp.isabs(p):
                full_path = osp.join(results['img_prefix'], p)
            else:
                full_path = p

            img = mmcv.imread(full_path, self.color_type)

            # 异常检测
            if img is None:
                print(f"警告: 无法读取图片 {full_path}")
                # 可以选择抛出错误或者用全黑图填充（这里选择报错）
                raise FileNotFoundError(f"Cannot find image: {full_path}")

            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        # 3. 堆叠 (H, W, 15)
        stacked_img = np.concatenate(imgs, axis=2)

        # 4. 更新 results 字典
        results['filename'] = paths[-1]  # 使用最新一帧的文件名作为 ID
        results['ori_filename'] = paths[-1]
        results['img'] = stacked_img
        results['img_shape'] = stacked_img.shape
        results['ori_shape'] = stacked_img.shape
        # 关键：告诉后续的 RandomFlip，这个 'img' 字段是图像，需要翻转
        results['img_fields'] = ['img']

        return results