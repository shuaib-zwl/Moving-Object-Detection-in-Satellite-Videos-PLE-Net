custom_imports = dict(
    imports=['muti_frames_pipelines'],  # 这里写文件名（不带 .py）
    allow_failed_imports=False
)

# 1. 定义数据增强 Pipeline
img_norm_cfg = dict(
    mean=[127.41, 127.41, 127.41] * 5,
    std=[21.05, 21.05, 21.05] * 5,
    to_rgb=False)

train_pipeline = [
    dict(type='LoadMultiFrameImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),

    # --- 图像增强区 ---
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadMultiFrameImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,  # 严禁 Resize
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'MFDataset'
data_root = 'D:/zhou/ZWL/'
classes = ("car",)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',  # 使用 RepeatDataset 增加采样率
        times=1,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            # 注意：路径要写对，确保和 LoadMultiFrameImageFromFile 拼接后是正确的
            ann_file='/root/autodl-tmp/VISO_train/train.json',
            img_prefix='/root/autodl-tmp/VISO_train',
            pipeline=train_pipeline)),  # 注入上面的 pipeline
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/root/autodl-tmp/VISO_test/test.json',
        img_prefix='/root/autodl-tmp/VISO_test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='D:\ZWL\data\VISO_test/test.json',
        img_prefix='D:\ZWL\data\VISO_test',
        pipeline=test_pipeline))


# 针对目标测试难的问题（过拟合问题），将weight_decay设置为0.001
# optimizer = dict(type='SGD', lr=0.00125*2, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.000075)
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
        type='DNANet_FrameStack',
        num_frames=5,
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
work_dir = './work_dir/centroid_invest2D'
gpu_ids = range(0, 4)

# 设置随机种子
# seed = 42
# cudnn_benchmark = False