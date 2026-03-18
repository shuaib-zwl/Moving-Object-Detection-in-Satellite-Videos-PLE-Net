# config文件修改整个模型所有的参数
# 所有配置文件打印：python tools/misc/print_configs.py /path/to/config_file。方法-backbone-fpn-1X表示12个epoch,2X是24个epoch，coco是数据集
# 示例 python tools/misc/print_configs.py configs/faster_rcnn/faster_rcnn_r50_fpn_iou_1x_coco.py
dataset_type = 'MYDataset'
data_root = 'D:/zhou/ZWL/'
classes = ("car",)
num_classes = 1
img_norm_cfg = dict(
    mean=[127.41, 127.41, 127.41],
    std=[21.05, 21.05, 21.05], to_rgb=True)

data = dict(
    samples_per_gpu=8,  # 每个GPU处理的个数，batchsize
    workers_per_gpu=8,  # 每个GPU分配的数据加载工作线程数量
    train=dict(
        type='RepeatDataset',
        times=1,
        sampler='SequentialSampler',  # 设置为顺序采样
        dataset=dict(type=dataset_type,
                     classes=classes,
                     ann_file='/data/ZWL/ZWL/data/VISO_train/train.json',
                     img_prefix='/data/ZWL/ZWL/data/VISO_train',
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
        ann_file='/data/ZWL/ZWL/data/test.json',
        img_prefix='/data/ZWL/ZWL/data/VISO_test',
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
        ann_file='/data/ZWL/ZWL/data/test.json',
        img_prefix='/data/ZWL/ZWL/data/VISO_test',
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
evaluation = dict(interval=2, metric='bbox')  # 评估频率，interval=1，每个epoch评估一次
# 针对目标测试难的问题（过拟合问题），将weight_decay设置为0.001
# optimizer = dict(type='SGD', lr=0.00125*2, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 随机梯度裁剪
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.0001,
    step=[5, 8],
    gamma=0.1)
val_cfg = dict(type='ValLoop', evaluate_with_loss=True)
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=2)  # 多少个epoch保存一次权重
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])  # 多少个epoch，文本记录训练信息
custom_hooks = [dict(type='NumClassCheckHook')]  # 检查类别数目是否正确
dist_params = dict(backend='nccl')  # 使用NCCL作为后端进行多GPU，这里添加时间指定，否则会出现验证时间过长。训练过程终止
log_level = 'INFO'  # 日志级别设置为INFO
load_from = None  # 指定是否从预训练模型加载权重。
resume_from = None  # 指定是否从中断处恢复训练。
# find_unused_parameters=True
workflow = [('train', 1)]  # 工作流程设定，主要执行训练阶段，数值1表示这一阶段重复1次，即正常的训练流程。
model = dict(
    type='CenterNet',
    backbone=dict(
        type='Backbone3D',  # 使用刚注册的 3DBackbone
        in_channels=3,      # 输入通道数，通常为 RGB 图像的 3 通道
        channels=[16, 32, 64],  # 输出通道数与您的网络设计一致
    ),
    neck=dict(type='DLANeck', start_level=0, end_level=3, channels=[16, 32, 64], scales=[1, 2, 4]),
    #bbox_head=dict(type='MYHead',num_classes=1,in_channel=16,feat_channel=16,loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),loss_xywh=dict(type='NWDLoss', loss_weight=1.0)),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channel=16,
        feat_channel=16,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
        # loss_mask=dict(type='BCEWithLogitsLoss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        topk=300,
        local_maximum_kernel=3,
        max_per_img=300,
        nms=dict(
            type='soft_nms',  # 开启 soft nms
            iou_threshold=0.5,  # soft-nms 起作用的 IoU 阈值
            min_score=0.05,  # 最低保留得分（低于这个就抛了）
            method='gaussian',  # 'gaussian' 或 'linear'
            sigma=0.5  # Gaussian 衰减参数，调大衰减更慢
        )))
work_dir = './work_dir/last/conv3d'
gpu_ids = range(0, 4)

