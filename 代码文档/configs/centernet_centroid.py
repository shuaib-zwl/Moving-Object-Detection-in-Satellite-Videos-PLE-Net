# config文件修改整个模型所有的参数
# 所有配置文件打印：python tools/misc/print_configs.py /path/to/config_file。方法-backbone-fpn-1X表示12个epoch,2X是24个epoch，coco是数据集
# 示例 python tools/misc/print_configs.py configs/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_coco.py
# 由于dataset是自定义的加载多帧，在使用mmdet的test进行--show的时候，需要修改misc文件下tensor2imgs，将图像加载从多帧变为单帧
dataset_type = 'MYDataset'
data_root = 'D:/zhou/ZWL/'
classes = ("car",)
num_classes = 1
img_norm_cfg = dict(
    mean=[127.41, 127.41, 127.41],
    std=[21.05, 21.05, 21.05], to_rgb=True)
# 12 0.0002, 24 0.0004, 32 0.0005, 36 0.0006
data = dict(
    samples_per_gpu=2,  # 每个GPU处理的个数，batchsize
    workers_per_gpu=2,  # 每个GPU分配的数据加载工作线程数量
    train=dict(
        type='RepeatDataset',
        times=1,
        sampler='SequentialSampler',  # 设置为顺序采样
        dataset=dict(type=dataset_type,
                     classes=classes,
                     ann_file='/root/autodl-tmp/VISO_train/train.json',
                     img_prefix='/root/autodl-tmp/VISO_train',
                     pipeline=[
                         dict(
                             type='LoadImageFromFile',
                             to_float32=True,
                             color_type='color'),
                         dict(type='LoadAnnotations', with_bbox=True),
                         dict(type='RandomFlip', flip_ratio=0.0),
                         dict(type='DefaultFormatBundle'),
                         dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                     ])),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/root/autodl-tmp/VISO_test/test.json',
        img_prefix='/root/autodl-tmp/VISO_test',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg'),
                        keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='D:\ZWL\data\VISO_test/test.json',
        img_prefix='D:\ZWL\data\VISO_test',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[127.41, 127.41, 127.41],
                        std=[21.05, 21.05, 21.05],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg'),
                        keys=['img'])
                ])
        ]))

# 针对目标测试难的问题（过拟合问题），将weight_decay设置为0.001
# optimizer = dict(type='SGD', lr=0.00125*2, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.00008)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 随机梯度裁剪
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    by_epoch=False,
    step=[4000, 4500],
    gamma=0.1)
evaluation = dict(interval=30000, by_epoch=False, metric='bbox')
val_cfg = dict(type='ValLoop', evaluate_with_loss=True)
# runner = dict(type='EpochBasedRunner', max_epochs=10)
runner = dict(type='IterBasedRunner', max_iters=5000)
# checkpoint_config = dict(interval=1)  # 多少个epoch保存一次权重
checkpoint_config = dict(interval=200, by_epoch=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])  # 多少个epoch，文本记录训练信息
custom_hooks = [dict(type='NumClassCheckHook')]  # 检查类别数目是否正确
dist_params = dict(backend='nccl')  # 使用NCCL作为后端进行多GPU，这里添加时间指定，否则会出现验证时间过长。训练过程终止
log_level = 'INFO'  # 日志级别设置为INFO
load_from = None  # 指定是否从预训练模型加载权重。
resume_from = None  # 指定是否从中断处恢复训练。
find_unused_parameters = True
workflow = [('train', 1)]  # 工作流程设定，主要执行训练阶段，数值1表示这一阶段重复1次，即正常的训练流程。
fp16 = dict(loss_scale='dynamic')
cudnn_benchmark = True  # 放心开，能提速
model = dict(
    type='CenterNet',
    # backbone=dict(type='Backbone3D', in_channels=3, channels=[16, 32, 64]),
    backbone=dict(
        type='DNANet',
        input_channels=3,
        num_blocks=[2, 2, 2, 2],
        nb_filter=[16, 32, 64, 128, 256],
        max_downsample=8,  # <--- 这里控制深度，写 8 或 16
        deep_supervision=False
    ),
    # neck=dict(type='DLANeck', start_level=0, end_level=3, channels=[16, 32, 64], scales=[1, 2, 4]),
    neck=None,
    bbox_head=dict(
        type='CenterNet_agg_centroid',
        num_classes=1,
        in_channel=16,
        feat_channel=16,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1)),
    train_cfg=None,
    test_cfg=dict(
        topk=100,
        local_maximum_kernel=3,
        max_per_img=100,
        nms=dict(
            type='nms',  # 开启 soft nms
            iou_threshold=0.5,  # soft-nms 起作用的 IoU 阈值
            min_score=0.05,  # 最低保留得分（低于这个就抛了）
        )))
work_dir = './work_dirr/compare/centernet_agg_centroid_invest2'
gpu_ids = range(0, 4)

# 设置随机种子
# seed = 42
# cudnn_benchmark = False