"""
此脚本改编自hieum文章中提供的验证思路，即距离为5的度量方式，
原脚本请查看:https://github.com/ChaoXiao12/Moving-object-detection-in-satellite-videos-HiEUM
"""
import os
import json
import numpy as np
from utils.utils_eval import eval_metric

# ====== 配置区域 ======
GT_JSON_PATH = 'C:\\Users\帅比周文亮\Desktop\提交的代码文档\代码文档\data\VISO_test\\test.json'  # GT JSON 文件路径（COCO 格式扁平列表）
DT_JSON_PATH = r'C:\Users\帅比周文亮\Desktop\提交的代码文档\代码文档\work_dirr\result.json'   # DT
# JSON 文件路径（扁平列表，含 score）
CONF_THRESHOLDS = [0.2, 0.25, 0.3]
DIS_TH = 5.0       # 距离阈值（像素）
IOU_TH = 0.05     # IoU 阈值
EVAL_MODE = 'dis' # 'dis' 或 'iou'
# ======================


def load_ground_truth(gt_path):
    """
    加载 GT JSON（dict 格式），返回 dict[image_id] → list of [x1,y1,x2,y2]
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_dict = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        x, y, w, h = ann['bbox']
        box = [float(x), float(y), float(x + w), float(y + h)]
        gt_dict.setdefault(img_id, []).append(box)
    return gt_dict


def load_detections(dt_path):
    """
    加载 DT JSON（list 格式），返回 dict[image_id] → list of (bbox, score)
    bbox 是 [x1,y1,x2,y2]
    """
    with open(dt_path, 'r', encoding='utf-8') as f:
        dets = json.load(f)
    dt_dict = {}
    for ann in dets:
        img_id = ann['image_id']
        x, y, w, h = ann['bbox']
        score = float(ann.get('score', 0))
        box = [float(x), float(y), float(x + w), float(y + h)]
        dt_dict.setdefault(img_id, []).append((box, score))
    return dt_dict


def main():
    gt_dict = load_ground_truth(GT_JSON_PATH)
    dt_dict_full = load_detections(DT_JSON_PATH)
    image_ids = sorted(gt_dict.keys())
    num_images = len(image_ids)

    for conf_thresh in CONF_THRESHOLDS:
        print(f"=== Conf ≥ {conf_thresh:.2f} | dis<{DIS_TH}, IoU≥{IOU_TH} ===")
        # 初始化评测器
        det_metric = eval_metric(dis_th=DIS_TH, iou_th=IOU_TH, eval_mode=EVAL_MODE)
        det_metric.reset()

        # 遍历每帧进行 update
        for img_id in image_ids:
            gt_boxes = np.array(gt_dict[img_id], dtype=float)
            det_list = dt_dict_full.get(img_id, [])
            # 过滤置信度
            det_list = [(b, s) for (b, s) in det_list if s >= conf_thresh]
            if det_list:
                det_arr = np.array([[*b, s] for b, s in det_list], dtype=float)
            else:
                det_arr = np.empty((0, 4))

            det_metric.update(gt_boxes, det_arr)

        # 获取统计结果
        res = det_metric.get_result(img_size=[1024, 1024], seq_len=num_images)
        print(f"TP={res['tp']}, FP={res['fp']}, FN={res['fn']}")
        print(f"Recall={res['recall']:.2f}%, Precision={res['prec']:.2f}%, F1={res['f1']:.2f}%")
        print(f"PD={res.get('pd',0):.3f}, FA={res.get('fa_1',0):.2e}\n")


if __name__ == '__main__':
    main()
