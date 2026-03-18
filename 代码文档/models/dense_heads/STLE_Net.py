# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from mmcv.cnn import build_norm_layer

import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as mcolors


class WindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        # 依然使用标准的 MHA，只是我们喂给它的数据形状变了
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, H, W):
        """
        x: [B, N, C]  (Flattened features)
        H, W: 当前特征图的网格尺寸 (例如 64, 64)
        """
        B, N, C = x.shape
        assert N == H * W, "Input shape mismatch with H, W"

        # 1. 恢复成 2D 图片形状: [B, H, W, C]
        x = x.view(B, H, W, C)

        # 2. 补齐 Padding (如果 H 或 W 不能被 window_size 整除)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        # F.pad 参数顺序: (左, 右, 上, 下, 前, 后...) 对最后几个维度操作
        # 这里是对 H, W 维度 pad，对应的是 dim 1, 2
        # 注意: x 是 [B, H, W, C]，permute 一下方便 pad 或者直接算
        if pad_r > 0 or pad_b > 0:
            # 为了方便 Pad，先转为 [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)  # 转回 [B, H', W', C]

        H_pad, W_pad = x.shape[1], x.shape[2]

        # 3. [核心步骤] Window Partition (切分窗口)
        # 目标: 将 [B, H, W, C] -> [B * num_windows, window_size * window_size, C]
        # 这样 MHA 就会认为 batch 变大了，但每个序列变短了，只在窗口内交互

        # Reshape 为 [B, h_wins, window_size, w_wins, window_size, C]
        x = x.view(B, H_pad // self.window_size, self.window_size,
                   W_pad // self.window_size, self.window_size, C)

        # Permute: [B, h_wins, w_wins, window_size, window_size, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Merge Batch: [B * num_windows, window_area, C]
        # window_area = window_size * window_size (例如 8*8=64)
        x_windows = x.view(-1, self.window_size * self.window_size, C)

        # 4. 计算 Attention
        # 此时 seq_len 只有 64，计算量极小，且没有远处的背景干扰
        attn_out, _ = self.mha(x_windows, x_windows, x_windows)

        # 5. [核心步骤] Window Reverse (还原窗口)
        # 把 batch 拆回来
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)

        # 还原为 [B, h_wins, w_wins, window_size, window_size, C]
        # 注意之前的 view 是把 (B, h_wins, w_wins) 融合了
        attn_out = attn_out.view(B, H_pad // self.window_size, W_pad // self.window_size,
                                 self.window_size, self.window_size, C)

        # Permute 回去: [B, h_wins, window_size, w_wins, window_size, C]
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Flatten 回 2D 图片: [B, H_pad, W_pad, C]
        attn_out = attn_out.view(B, H_pad, W_pad, C)

        # 6. 移除 Padding (如果有)
        if pad_r > 0 or pad_b > 0:
            attn_out = attn_out[:, :H, :W, :]

        # 7. Flatten 回 [B, N, C] 以适配后续网络
        attn_out = attn_out.reshape(B, N, C)

        return attn_out


class PatchAttentionFilter(nn.Module):
    def __init__(self,
                 in_channels=16,
                 patch_size=32,
                 embed_dim=256,
                 num_heads=4,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.patch_size = patch_size
        mid_dim = embed_dim // 2

        if patch_size < 32:
            stride_1 = 4
        else:
            stride_1 = 8

        stride_2 = patch_size // stride_1

        # [分支 1] Max Branch: 寻找小目标
        self.max_branch = nn.Sequential(
            # --- Stage 1 ---
            nn.Conv2d(in_channels, mid_dim, 3, 1, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(),
            nn.MaxPool2d(kernel_size=stride_1, stride=stride_1),

            # --- Stage 2 ---
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(),
            nn.MaxPool2d(kernel_size=stride_2, stride=stride_2)
        )

        # [分支 2] Avg/Context Branch: 识别区域纹理
        self.avg_branch = nn.Sequential(
            # Stage 1
            nn.AvgPool2d(kernel_size=stride_1, stride=stride_1),
            nn.Conv2d(in_channels, mid_dim, 3, 1, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(),
            nn.AvgPool2d(kernel_size=stride_2, stride=stride_2)
        )

        # 融合层 (Concat后 128+128 -> 256)
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # 2. Positional Encoding (CPE), 位置编码
        self.pos_embed_conv = nn.Conv2d(embed_dim, embed_dim,
                                        kernel_size=3,
                                        padding=1,
                                        groups=embed_dim)

        # Context Modeling
        self.mha = WindowAttention(embed_dim, num_heads, window_size=8, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Score Head
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 投影层 (Projection Head)
        self.proj_head = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.score_head[-1].bias, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: 特征提取
        x_max = self.max_branch(x)
        x_avg = self.avg_branch(x)
        x_embed = self.fusion(torch.cat([x_max, x_avg], dim=1))

        # 记录网格尺寸
        current_grid_h, current_grid_w = x_embed.shape[2], x_embed.shape[3]

        # Flatten & MHA
        x_embed = x_embed + self.pos_embed_conv(x_embed)
        x_flat = x_embed.flatten(2).transpose(1, 2)  # [B, N, C]
        attn_out = self.mha(x_flat, current_grid_h, current_grid_w)
        x_flat = self.norm(x_flat + attn_out)

        # Step 2: 预测 Mask (Discriminative)
        patch_scores = self.score_head(x_flat)
        patch_prob = torch.sigmoid(patch_scores)

        # Step 3: 原型提取 (Generative)
        prob_detached = patch_prob.detach()

        # 硬阈值筛选 (依然使用硬阈值)
        threshold = 0.6
        hard_mask = (prob_detached > threshold).float()

        # Gate 机制
        sum_hard = torch.sum(hard_mask, dim=1, keepdim=True)
        has_target_gate = (sum_hard > 0).float()  # [B, 1, 1]

        # 计算原型
        safe_sum = sum_hard + 1e-6

        # 使用投影后的特征计算原型和相似度
        x_proj = self.proj_head(x_flat)
        prototype = torch.sum(x_proj * hard_mask, dim=1, keepdim=True) / safe_sum

        # 计算相似度
        sim_values = F.cosine_similarity(x_proj, prototype, dim=2)  # [B, N]
        sim_values = torch.relu(sim_values).unsqueeze(-1)  # 只取正相关 [B, N, 1]

        # 前景阈值
        bg_threshold = 0.0
        fg_threshold = 0.6  # 高于此值为前景 (Boost)

        raw_coeff = torch.ones_like(sim_values)

        # 3. 处理背景区间 (Sim < 0.15 -> 置 0)，实际改为 0并未抑制背景
        # suppress_mask: [B, N, 1]
        suppress_mask = (sim_values < bg_threshold)
        raw_coeff.masked_fill_(suppress_mask, 0.0)

        # 4. agg
        # boost_mask: [B, N, 1]
        boost_mask = (sim_values > fg_threshold)
        enhancement = 1.0 + 0.5 * sim_values
        raw_coeff = torch.where(boost_mask, enhancement, raw_coeff)
        final_coeff = has_target_gate * raw_coeff + (1.0 - has_target_gate) * 1.0
        coeff_map_low_res = final_coeff.transpose(1, 2).reshape(B, 1, current_grid_h, current_grid_w)
        modulation_map = F.interpolate(coeff_map_low_res, size=(H, W), mode='nearest')
        out = x * modulation_map

        # 返回 logits 给 Loss
        mask_logits_for_loss = patch_scores.transpose(1, 2).reshape(B, 1, current_grid_h, current_grid_w)

        return out, mask_logits_for_loss

@HEADS.register_module()
class Centroid_3D_Attention_Pseudo(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_mask=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 # loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Centroid_3D_Attention_Pseudo, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        # 1. 实例化注意力模块
        self.attn_filter = PatchAttentionFilter(in_channels=in_channel, patch_size=32, embed_dim=256, num_heads=4)

        self.loaded_pseudo_labels = {}  # 存上一轮的数据
        self.generated_pseudo_labels = {}  # 存本轮生成的数据
        self.path_lookup = {}    # 显式初始化索引为空字典，防止 Epoch 0 报错

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_mask = build_loss(loss_mask)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def build_lookup_table(self):
        """
        当 Hook 加载完新的伪标签后，调用此方法重建 O(1) 索引
        """
        if not self.loaded_pseudo_labels:
            self.path_lookup = {}
            return

        print(f"[Head] 正在为 {len(self.loaded_pseudo_labels)} 条伪标签构建加速索引...")
        self.path_lookup = {}
        for full_key in self.loaded_pseudo_labels.keys():
            filename = full_key.split('/')[-1]

            if filename not in self.path_lookup:
                self.path_lookup[filename] = []
            self.path_lookup[filename].append(full_key)
        print(f"[Head] 索引构建完成！")

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        # for head in [self.wh_head, self.offset_head]:
        for head in self.wh_head:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas) # 获取真实信息
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)  # 计算loss
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        """
                Returns:
                    center_heatmap_preds (List[Tensor])
                    wh_preds (List[Tensor])
                    mask_preds (List[Tensor]): [新增] 注意力掩码预测
                """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.
        """

        feat_filtered, mask_low_res = self.attn_filter(feat)
        center_heatmap_pred = self.heatmap_head(feat_filtered).sigmoid()
        # center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        mask_prob = torch.sigmoid(mask_low_res)
        wh_pred = self.wh_head(feat)
        # offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, mask_prob

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'mask_preds'))
    def loss(self,
             heatmap_preds,
             wh_preds,
             mask_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        """
        assert len(heatmap_preds) == len(wh_preds) == 1
        heatmap_pred = heatmap_preds[0]
        wh_pred = wh_preds[0]
        mask_preds = mask_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels, img_metas,
                                                     heatmap_pred)

        heatmap_target = target_result['heatmap_target']
        heatmap_weight = target_result['heatmap_weight']
        wh_target = target_result['wh_target']
        wh_target_weight = target_result['wh_target_weight']
        mask_target = target_result['mask_target']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            heatmap_pred, heatmap_target, heatmap_weight, avg_factor=avg_factor)
        loss_mask = self.loss_mask(mask_preds, mask_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(wh_pred, wh_target, wh_target_weight, avg_factor=avg_factor * 2)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_mask=loss_mask,
            loss_wh=loss_wh)

    def get_targets(self, gt_bboxes, gt_labels, img_metas, heatmap_preds):
        batch_size, _, feat_h, feat_w = heatmap_preds.shape
        device = heatmap_preds.device

        # --- 1. 初始化 Target ---
        final_heatmap_target = torch.zeros(batch_size, self.num_classes, feat_h, feat_w, device=device)
        final_heatmap_weight = torch.ones(batch_size, self.num_classes, feat_h, feat_w, device=device)
        wh_target = torch.zeros([batch_size, 2, feat_h, feat_w], device=device)
        wh_offset_target_weight = torch.zeros([batch_size, 2, feat_h, feat_w], device=device)

        # mask
        mask_target = self.get_attention_mask_target(gt_bboxes, img_metas, patch_size=32)

        for b in range(batch_size):
            raw_filename = img_metas[b]['filename']
            img_key = raw_filename.replace('\\', '/')

            prev_pseudo_info = self.loaded_pseudo_labels.get(img_key, None)

            if prev_pseudo_info is None:
                fname = img_key.split('/')[-1]  # 提取文件名 "000097.jpg"

                candidates = self.path_lookup.get(fname, [])

                for cand_key in candidates:

                    if img_key.endswith(cand_key) or cand_key.endswith(img_key):
                        prev_pseudo_info = self.loaded_pseudo_labels[cand_key]
                        break
            feature_b = heatmap_preds[b:b + 1]


            img_h, img_w = img_metas[b]['pad_shape'][:2]
            width_ratio = feat_w / img_w
            height_ratio = feat_h / img_h
            gt_bbox = gt_bboxes[b]

            gt_centroids = []
            gt_wh_dict = {}

            for i in range(len(gt_bbox)):
                w = (gt_bbox[i, 2] - gt_bbox[i, 0]) * width_ratio
                h = (gt_bbox[i, 3] - gt_bbox[i, 1]) * height_ratio
                cx = (gt_bbox[i, 0] + gt_bbox[i, 2]) * width_ratio / 2
                cy = (gt_bbox[i, 1] + gt_bbox[i, 3]) * height_ratio / 2
                cx_int = int(max(0, min(cx, feat_w - 1)))
                cy_int = int(max(0, min(cy, feat_h - 1)))
                gt_centroids.append([cx_int, cy_int])
                gt_wh_dict[i] = (w, h)

            # ---  生成伪标签 ---
            local_target, local_weight, current_pseudo_info, best_peaks_map = self.set_local_response_target_optimized(
                feature_b,
                gt_centroids,
                prev_pseudo_info,
                img_metas[b]
            )

            final_heatmap_target[b] = local_target
            final_heatmap_weight[b] = local_weight

            # ---  WH 回归 ---
            for gt_idx, (bx, by) in best_peaks_map.items():
                if gt_idx in gt_wh_dict:
                    target_w, target_h = gt_wh_dict[gt_idx]
                    if 0 <= bx < feat_w and 0 <= by < feat_h:
                        wh_target[b, 0, by, bx] = target_w
                        wh_target[b, 1, by, bx] = target_h
                        wh_offset_target_weight[b, :, by, bx] = 1

            # 保存当前伪标签
            cpu_pseudo_info = {}
            for gt_idx, info in current_pseudo_info.items():
                cpu_pseudo_info[gt_idx] = info['coords'].detach().cpu().numpy()
            self.generated_pseudo_labels[img_key] = cpu_pseudo_info

        num_valid_objs = wh_offset_target_weight.eq(1).sum().item() / 2
        avg_factor = max(1, num_valid_objs)

        targets_dict = {
            'heatmap_target': final_heatmap_target,
            'heatmap_weight': final_heatmap_weight,
            'wh_target': wh_target,
            'wh_target_weight': wh_offset_target_weight,
            'mask_target': mask_target,
        }

        return targets_dict, avg_factor

    def set_local_response_target_optimized(self, heatmap_pred, gt_centroids, prev_pseudo_info, img_meta):
        """
        pseudo target
        """
        N, C, H, W = heatmap_pred.shape
        device = heatmap_pred.device

        target_map = torch.zeros(N, C, H, W, device=device)
        weight_map = torch.ones(N, C, H, W, device=device)

        current_pseudo_info_dict = {}
        best_peaks_map = {}

        R = 2  # wh radius
        MIN_PEAK_THRESHOLD = 0.3
        MIN_CANDIDATE_THRESHOLD = 0.1

        is_flipped = img_meta.get('flip', False)
        flip_direction = img_meta.get('flip_direction', 'horizontal')

        def dilate_mask(mask):
            return F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1) > 0.5

        for i, (cx, cy) in enumerate(gt_centroids):
            # 1. 切片窗口
            x1, x2 = max(0, cx - R), min(W, cx + R + 1)
            y1, y2 = max(0, cy - R), min(H, cy + R + 1)

            local_pred = heatmap_pred[0, 0, y1:y2, x1:x2].detach()
            h_win, w_win = local_pred.shape
            if h_win == 0 or w_win == 0: continue

            pos_mask = torch.zeros_like(local_pred, dtype=torch.bool)

            # 2. 锚定 GT (局部坐标)
            l_cx, l_cy = min(max(0, cx - x1), w_win - 1), min(max(0, cy - y1), h_win - 1)
            pos_mask[l_cy, l_cx] = True

            # --- 3. 载入历史 ---
            data = None
            if prev_pseudo_info:
                # 优先尝试整数 Key，备选字符串 Key
                if i in prev_pseudo_info:
                    data = prev_pseudo_info[i]
                elif str(i) in prev_pseudo_info:
                    data = prev_pseudo_info[str(i)]

            if data is not None:
                hist_coords = None

                # 类型兼容处理
                if isinstance(data, dict):
                    raw = data.get('coords', None)
                    if raw is not None:
                        if isinstance(raw, (np.ndarray, list, tuple)):
                            hist_coords = torch.tensor(raw, device=device)
                        elif isinstance(raw, torch.Tensor):
                            hist_coords = raw.to(device)
                elif isinstance(data, (np.ndarray, list, tuple)):
                    hist_coords = torch.tensor(data, device=device)
                elif isinstance(data, torch.Tensor):
                    hist_coords = data.to(device)

                if hist_coords is not None and len(hist_coords) > 0:
                    # 翻转修正
                    if is_flipped and flip_direction == 'horizontal':
                        hist_coords[:, 0] = W - 1 - hist_coords[:, 0]

                    # 距离判定
                    abs_dx = torch.abs(hist_coords[:, 0] - cx)
                    abs_dy = torch.abs(hist_coords[:, 1] - cy)
                    valid_dist_mask = (abs_dx <= R + 1) & (abs_dy <= R + 1)

                    # 转为局部坐标
                    hist_y = hist_coords[:, 1].long() - y1
                    hist_x = hist_coords[:, 0].long() - x1
                    in_window = (hist_x >= 0) & (hist_x < w_win) & (hist_y >= 0) & (hist_y < h_win)

                    # 双重过滤
                    final_valid = in_window & valid_dist_mask

                    # 绘制历史点到 Mask
                    if final_valid.any():
                        pos_mask[hist_y[final_valid], hist_x[final_valid]] = True

            # 4. 寻找 Peak
            best_px, best_py = l_cx, l_cy
            if local_pred.max() > MIN_PEAK_THRESHOLD:
                flat_idx = local_pred.argmax()
                best_py, best_px = flat_idx // w_win, flat_idx % w_win
                pos_mask[best_py, best_px] = True

            best_peaks_map[i] = (int(x1 + best_px), int(y1 + best_py))

            # 5. Top-1 强制生长
            dilated_pos = dilate_mask(pos_mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            neighbor_mask = dilated_pos & (~pos_mask)

            if neighbor_mask.any():
                masked_pred = local_pred.clone()
                masked_pred[~neighbor_mask] = -1.0

                top1_val = masked_pred.max()
                top1_flat = masked_pred.argmax()

                if top1_val > MIN_CANDIDATE_THRESHOLD:
                    t1_y, t1_x = top1_flat // w_win, top1_flat % w_win
                    pos_mask[t1_y, t1_x] = True

            # 6. Buffer & 赋值
            final_dilated = dilate_mask(pos_mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            buffer_mask = final_dilated & (~pos_mask)
            weight_map[0, 0, y1:y2, x1:x2][buffer_mask] = 0.0

            target_window = target_map[0, 0, y1:y2, x1:x2]
            weight_window = weight_map[0, 0, y1:y2, x1:x2]
            target_window[pos_mask] = 0.8
            weight_window[pos_mask] = 1.0
            target_window[best_py, best_px] = 0.9
            target_window[l_cy, l_cx] = 1.0

            # 7. 保存 (存回之前翻转回原始坐标系)
            py, px = torch.where(pos_mask)
            global_x = px + x1
            global_y = py + y1

            if is_flipped and flip_direction == 'horizontal':
                save_x = W - 1 - global_x
            else:
                save_x = global_x

            global_coords = torch.stack([save_x, global_y], dim=1).float()
            current_pseudo_info_dict[i] = {'coords': global_coords}

        return target_map, weight_map, current_pseudo_info_dict, best_peaks_map

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds'))
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   mask_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):

        assert len(center_heatmap_preds) == len(wh_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def get_attention_mask_target(self, gt_bboxes, img_metas, patch_size=64):
        """
        get mask
        """
        # 1. 获取基本信息
        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device if len(gt_bboxes) > 0 else torch.device('cuda')
        img_h, img_w = img_metas[0]['pad_shape'][:2]

        # 计算格网数量
        grid_h = img_h // patch_size
        grid_w = img_w // patch_size

        # 初始化原始掩码 (B, 1, Gh, Gw)
        raw_mask = torch.zeros((batch_size, 1, grid_h, grid_w), device=device)

        for b in range(batch_size):
            bboxes = gt_bboxes[b]
            if bboxes.numel() == 0:
                continue

            # 缩放坐标到 Grid 坐标系
            scaled_bboxes = bboxes / patch_size

            ix1 = torch.floor(scaled_bboxes[:, 0]).long().clamp(0, grid_w - 1)
            iy1 = torch.floor(scaled_bboxes[:, 1]).long().clamp(0, grid_h - 1)
            ix2 = torch.floor(scaled_bboxes[:, 2]).long().clamp(0, grid_w - 1)
            iy2 = torch.floor(scaled_bboxes[:, 3]).long().clamp(0, grid_h - 1)

            for i in range(len(bboxes)):
                x_start, x_end = ix1[i], ix2[i]
                y_start, y_end = iy1[i], iy2[i]

                raw_mask[b, 0, y_start:y_end + 1, x_start:x_end + 1] = 1.0

        mask_target = raw_mask

        return mask_target

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, num_classes, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, corner_offset_channels, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)
        # 没有采取填充，注释，防止测试报错
        # batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                #  [2, 0, 2, 0]]
        # det_bboxes[..., :4] -= batch_border

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        topk_xs = topk_xs
        topk_ys = topk_ys
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels
