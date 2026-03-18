import os
import os.path as osp
import pickle
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class PseudoLabelHook(Hook):
    def __init__(self, pseudo_dir, update_interval=1):
        """
        Args:
            pseudo_dir (str): 伪标签保存的根目录
            update_interval (int): 多少个 epoch 更新一次伪标签
        """
        self.pseudo_dir = pseudo_dir
        self.update_interval = update_interval

        # 获取当前进程的 rank
        rank, _ = get_dist_info()

        # 仅主进程负责创建目录
        if rank == 0:
            os.makedirs(self.pseudo_dir, exist_ok=True)
            print(f"[PseudoLabelHook] Initialized directory: {self.pseudo_dir}")

    def get_real_head(self, runner):
        """获取真实的 Head 实例，自动解包 DDP/DataParallel"""
        if hasattr(runner.model, 'module') and hasattr(runner.model.module, 'bbox_head'):
            return runner.model.module.bbox_head
        if hasattr(runner.model, 'bbox_head'):
            return runner.model.bbox_head
        return None

    def before_train_epoch(self, runner):
        """
        每个 epoch 开始前加载伪标签。
        所有卡都需要加载数据（因为所有卡都要训练），这里采用让所有卡读取同一文件的方式。
        """
        # 如果不是更新周期，或者是第0个epoch（还没有伪标签），则跳过
        if runner.epoch == 0 or runner.epoch % self.update_interval != 0:
            return

        # 1. 确保所有进程同步，防止 Rank 0 还没写完，Rank 1 就开始读了
        if dist.is_initialized():
            dist.barrier()

        prev_file = osp.join(self.pseudo_dir, f'pseudo_epoch_{runner.epoch}.pkl')
        real_head = self.get_real_head(runner)

        if real_head is None:
            return

        # 初始化/清空容器
        if hasattr(real_head, 'loaded_pseudo_labels'):
            real_head.loaded_pseudo_labels = {}

        if osp.exists(prev_file):
            try:
                with open(prev_file, 'rb') as f:
                    pseudo_labels = pickle.load(f, encoding='latin1')

                if hasattr(real_head, 'loaded_pseudo_labels'):
                    real_head.loaded_pseudo_labels = pseudo_labels

                    # ===========【核心修复：手动触发重建索引】===========
                    if hasattr(real_head, 'build_lookup_table'):
                        real_head.build_lookup_table()
                    # ================================================

                    rank, _ = get_dist_info()
                    if rank == 0:
                        runner.logger.info(
                            f'[PLHook] Epoch {runner.epoch}: Loaded {len(pseudo_labels)} pseudo labels and updated index.')
            except Exception as e:
                runner.logger.error(f'[PLHook] Failed to load {prev_file}: {e}')
        else:
            rank, _ = get_dist_info()
            if rank == 0:
                runner.logger.warning(f'[PLHook] File not found: {prev_file}')

    def after_train_epoch(self, runner):
        """
        每个 epoch 结束后，收集所有卡的数据并保存。
        """
        # 检查是否需要更新
        if runner.epoch % self.update_interval != 0:
            return

        real_head = self.get_real_head(runner)
        if real_head is None:
            return

        # 1. 获取当前显卡上的局部数据
        # 注意：这里的 generated_pseudo_labels 只有当前 GPU 处理的那部分图片的标签
        local_labels = getattr(real_head, 'generated_pseudo_labels', {})

        # 2. DDP 关键步骤：收集所有显卡的数据
        rank, world_size = get_dist_info()

        if world_size > 1 and dist.is_initialized():
            # 创建一个列表来接收所有卡的数据
            all_labels_list = [None for _ in range(world_size)]
            try:
                # 🌟 集合所有数据：这步会阻塞，直到所有卡都运行到这里
                dist.all_gather_object(all_labels_list, local_labels)
            except Exception as e:
                runner.logger.error(f"[PLHook] Distributed gather failed: {e}")
                # 如果 gather 失败，降级为只保存当前卡的数据（防止程序崩溃）
                all_labels_list = [local_labels]
        else:
            # 单卡模式
            all_labels_list = [local_labels]

        # 3. 仅 Rank 0 负责合并数据和写入磁盘
        if rank == 0:
            merged_labels = {}
            # 合并所有卡的数据字典
            for labels in all_labels_list:
                if labels:
                    merged_labels.update(labels)

            # 执行保存逻辑
            if len(merged_labels) > 0:
                self._save_to_disk(runner, merged_labels, runner.epoch + 1)
            else:
                runner.logger.warning(f'[PLHook] No pseudo labels generated in epoch {runner.epoch}.')

        # 4. 清理内存 (所有卡都要做)
        real_head.generated_pseudo_labels = {}

        # 5. 再次同步，确保 Rank 0 写完文件后，其他卡才能进入下一个 Epoch
        if dist.is_initialized():
            dist.barrier()

    def _save_to_disk(self, runner, data, epoch):
        """辅助函数：安全的写文件操作（仅在 Rank 0 调用）"""
        final_file = osp.join(self.pseudo_dir, f'pseudo_epoch_{epoch}.pkl')
        temp_file = final_file + '.tmp'

        try:
            # 写入临时文件
            with open(temp_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())  # 强制同步到磁盘

            # 原子重命名
            if osp.exists(temp_file):
                os.rename(temp_file, final_file)
                runner.logger.info(
                    f'[PLHook] Successfully saved {len(data)} pseudo labels for next epoch ({epoch}).')
            else:
                runner.logger.error(f"[PLHook] Temp file vanished: {temp_file}")

        except Exception as e:
            runner.logger.error(f'[PLHook] Save failed due to error: {e}')
            if osp.exists(temp_file):
                os.remove(temp_file)