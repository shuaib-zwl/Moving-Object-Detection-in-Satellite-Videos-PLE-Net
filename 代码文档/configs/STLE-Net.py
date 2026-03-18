custom_imports = dict(
    imports=['muti_frames_pipelines', 'hook.custom', 'hook.loadhookl'],  # 修改点：合并 imports
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
    samples_per_gpu=1,
    workers_per_gpu=1,
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
        ann_file = 'D:\ZWL\data\VISO_test/test.json',
        img_prefix = 'D:\ZWL\data\VISO_test',
        pipeline=test_pipeline))


# 针对目标测试难的问题（过拟合问题），将weight_decay设置为0.001
# optimizer = dict(type='SGD', lr=0.00125*2, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.00015)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))  # 随机梯度裁剪
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    by_epoch=False,
    step=[14000, 14500],
    gamma=0.1)
evaluation = dict(interval=40000, metric='bbox')  # 评估频率，interval=1，每个epoch评估一次
# evaluation = dict(interval=30000, by_epoch=False, metric='bbox')
val_cfg = dict(type='ValLoop', evaluate_with_loss=True)
# runner = dict(type='EpochBasedRunner', max_epochs=10)
runner = dict(type='IterBasedRunner', max_iters=5000)
# checkpoint_config = dict(interval=1)  # 多少个epoch保存一次权重
checkpoint_config = dict(interval=200, by_epoch=False)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])  # 多少个epoch，文本记录训练信息
static_pkl_path = 'work_dirs/pseudo_labels/pseudo_epoch_5.pkl'

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='LoadStaticPickleHook',
        pkl_path=static_pkl_path
    )
    # dict(type='PseudoLabelHook', pseudo_dir='work_dirs/pseudo_labels', update_interval=1)
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/tmp/pycharm_project_46/work_dirr/5_pseudo_without_mask/iter_2000.pth'
resume_from = None
# find_unused_parameters = True
workflow = [('train', 1)]
fp16 = dict(loss_scale='dynamic')
cudnn_benchmark = True
model = dict(
    type='CenterNet',
    # backbone=dict(type='Backbone3D', in_channels=3, channels=[16, 32, 64]),
    backbone=dict(
        type='DNANet_3DCollapse',
        num_frames=5,
        input_channels=3,
        num_blocks=[2, 2, 2, 2],
        nb_filter=[16, 32, 64, 128, 256],
        max_downsample=8,
        deep_supervision=False
    ),
    # neck=dict(type='DLANeck', start_level=0, end_level=3, channels=[16, 32, 64], scales=[1, 2, 4]),
    neck=None,
    bbox_head=dict(
        type='Centroid_3D_Attention_Pseudo',
        num_classes=1,
        in_channel=16,
        feat_channel=16,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_mask=dict(type='GaussianFocalLoss', loss_weight=5.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.3)),
    train_cfg=None,
    test_cfg=dict(
        topk=128,
        local_maximum_kernel=3,
        max_per_img=128,
        nms=dict(
            type='nms',
            iou_threshold=0.5,
            min_score=0.05,
        )))
work_dir = './work_dirr/5_pseudo_with_mask'
gpu_ids = range(0, 4)

