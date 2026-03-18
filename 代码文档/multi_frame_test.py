# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from torch.nn.parallel import _functions as torch_functions
from mmdet.core import encode_mask_results
from mmcv.image import tensor2imgs
from mmcv.parallel import DataContainer as DC

# 引入你的自定义数据集和模型
# 请确保 MYDataset.py 和 models 文件夹在当前目录下或 pythonpath 中
from models import *
# from MYDataset import MYDataset
from Multi_frame_Dataset import MFDataset

def patched_get_stream(device):
    """
    重写 _get_stream 函数以兼容新版 PyTorch
    如果 device 是 int，先转为 torch.device('cuda:x')
    """
    if isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    return torch_functions._get_stream(device)


# 强制替换 mmcv 中的 _get_stream 函数
mmcv.parallel._functions._get_stream = patched_get_stream


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def custom_single_gpu_test(model,
                           data_loader,
                           show=False,
                           out_dir=None,
                           show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)

        # 仅当需要显示或保存图片时才进行繁琐的解包操作
        if show or out_dir:
            # ================== 修复逻辑开始 ==================
            # 1. 获取 img_list (data['img'] 是一个列表，对应 multi-scale，通常只有一个)
            img_container_list = data['img']
            img_container = img_container_list[0]

            # 2. 解包 DataContainer (.data)
            if isinstance(img_container, DC):
                img_data = img_container.data
            else:
                img_data = img_container

            # 3. [关键修复] 二次解包：如果取出的 data 还是一个 list (batch_size=1 时常见)，则取第一个元素
            if isinstance(img_data, list):
                img_tensor = img_data[0]
            else:
                img_tensor = img_data

            # 此时 img_tensor 必须是 Tensor
            if not isinstance(img_tensor, torch.Tensor):
                # 最后的防线：如果还不是 Tensor，说明数据管道有问题，跳过可视化防止崩溃
                warnings.warn(f"Warning: Extracted image is {type(img_tensor)}, not Tensor. Skipping visualization.")
                img_vis_tensor = None
            else:
                # 4. 多帧处理：取最后一帧
                if img_tensor.dim() == 5:
                    # (N, C, T, H, W) -> 取 T 的最后一帧
                    img_vis_tensor = img_tensor[:, :, -1, :, :]
                elif img_tensor.dim() == 4 and img_tensor.size(1) > 3:
                    # (N, C*T, H, W) -> 取最后 3 个通道
                    img_vis_tensor = img_tensor[:, -3:, :, :]
                else:
                    # 正常 (N, 3, H, W)
                    img_vis_tensor = img_tensor

            # 5. 解包 Meta 信息
            meta_container_list = data['img_metas']
            meta_container = meta_container_list[0]

            if isinstance(meta_container, DC):
                # .data 通常是一个 list (batch)，取第一个 batch 的 list
                img_metas_list = meta_container.data[0]
            else:
                img_metas_list = meta_container

            # 确保 img_metas_list 是列表
            if not isinstance(img_metas_list, list):
                img_metas_list = [img_metas_list]

            # 6. 反归一化并可视化
            if img_vis_tensor is not None:
                # 取第一个样本的 meta 来获取 img_norm_cfg
                first_meta = img_metas_list[0]

                try:
                    imgs = tensor2imgs(img_vis_tensor, **first_meta['img_norm_cfg'])

                    for i, (img, img_meta) in enumerate(zip(imgs, img_metas_list)):
                        h, w, _ = img_meta['img_shape']
                        img_show = img[:h, :w, :]

                        ori_h, ori_w = img_meta['ori_shape'][:2]
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                        if out_dir:
                            out_file = osp.join(out_dir, img_meta['ori_filename'])
                        else:
                            out_file = None

                        model.module.show_result(
                            img_show,
                            result[i],
                            show=show,
                            out_file=out_file,
                            score_thr=show_score_thr)
                except KeyError as e:
                    warnings.warn(f"Missing Key in img_metas: {e}. Skipping visualization.")
            # ================== 修复逻辑结束 ==================

        # Encode mask results if necessary
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    return results


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle', '.json')):
        raise ValueError('The output file must be a pkl/pickle/json file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is None and cfg.get('device', None) == 'npu':
        fp16_cfg = dict(loss_scale='dynamic')
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,} ({total_params / 1e6:.2f}M)')
    print(f'Trainable params: {trainable_params:,} ({trainable_params / 1e6:.2f}M)')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        # 使用自定义的测试函数
        outputs = custom_single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
                                             or cfg.evaluation.get('gpu_collect', False))

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            if args.out.endswith(('.pkl', '.pickle')):
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            elif args.out.endswith('.json'):
                # 导出 COCO JSON 格式
                json_prefix = osp.splitext(args.out)[0]
                result_files = dataset.results2json(outputs, json_prefix)
                print(f'\nCOCO json saved to {result_files.get("bbox", result_files)}')
            else:
                raise ValueError('`--out` must be .pkl/.pickle or .json')
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()