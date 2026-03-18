import os.path as osp
import pickle
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LoadStaticPickleHook(Hook):
    def __init__(self, pkl_path):
        """
        Args:
            pkl_path (str): 之前保存的 .pkl 文件路径 (例如 pseudo_epoch_6.pkl)
        """
        self.pkl_path = pkl_path
        if not osp.exists(pkl_path):
            raise FileNotFoundError(f"找不到指定的伪标签文件: {pkl_path}")

    def get_real_head(self, runner):
        """复用你原有的获取 Head 逻辑"""
        if hasattr(runner.model, 'module') and hasattr(runner.model.module, 'bbox_head'):
            return runner.model.module.bbox_head
        if hasattr(runner.model, 'bbox_head'):
            return runner.model.bbox_head
        return None

    def before_run(self, runner):
        """
        在训练开始前 (IterBased 或 EpochBased 均有效) 执行一次。
        读取 .pkl 并注入 Head。
        """
        runner.logger.info(f"[StaticHook] 正在加载静态伪标签: {self.pkl_path}")

        # 1. 使用 pickle 读取二进制文件 (解决之前的 UnicodeDecodeError)
        try:
            with open(self.pkl_path, 'rb') as f:
                # 保持和你保存时一致的参数
                pseudo_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            runner.logger.error(f"[StaticHook] 加载失败: {e}")
            raise e

        # 2. 获取 Head
        real_head = self.get_real_head(runner)
        if real_head is None:
            runner.logger.warning("[StaticHook] 警告: 未找到 bbox_head，无法注入标签！")
            return

        # 3. 【核心操作】注入数据
        # 直接覆盖 loaded_pseudo_labels，使其变为静态库
        real_head.loaded_pseudo_labels = pseudo_data

        # 4. 【关键步骤】触发索引重建
        # 你的 Head 里有这个方法，必须调用它，否则 get_targets 里的 path_lookup 查不到数据
        if hasattr(real_head, 'build_lookup_table'):
            real_head.build_lookup_table()
        else:
            runner.logger.warning("[StaticHook] Head 缺少 build_lookup_table 方法，伪标签可能无法匹配！")

        runner.logger.info(f"[StaticHook] 成功注入 {len(pseudo_data)} 条静态伪标签，后续将不再更新。")