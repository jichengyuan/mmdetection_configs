"""Faster RCNN with ResNet50-FPN, 3x schedule, MS training."""

# model settings
model = dict(
    type="FasterRCNN",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ),
)


"""Dataset settings."""
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[
            (1280, 600),
            (1280, 624),
            (1280, 648),
            (1280, 672),
            (1280, 696),
            (1280, 720),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

dataset_type = 'CocoDataset'
data_root = 'data/annotations/'
classes = ('pedestrian', 'car', 'rider', 'bus', 'truck', 'bicycle', 'motorcycle', 
           'traffic light', 'traffic sign', 'train',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_bdd/bdd_100k_train.json',
        img_prefix='/pinky-data/jicheng_workspace/jicheng_notebook/detectron2/projects/UniDet/datasets/bdd_100k/images/100k/train/',
        pipeline=train_pipeline,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_bdd/bdd_100k_val.json',
        img_prefix='/pinky-data/jicheng_workspace/jicheng_notebook/detectron2/projects/UniDet/datasets/bdd_100k/images/100k/val/',
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_bdd/bdd_100k_val.json',
        img_prefix='/pinky-data/jicheng_workspace/jicheng_notebook/detectron2/projects/UniDet/datasets/bdd_100k/images/100k/val/',
        pipeline=test_pipeline,
        classes=classes,
    ),
)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')

# optimizer
optimizer = dict(type="SGD", lr=0.04, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33],
)
runner = dict(type="EpochBasedRunner", max_epochs=36)

checkpoint_config = dict(interval=1)
# yapf:disable
project = 'eccv_visionkg'
name = "Baseline_frcnn_bdd100k_fatercnn_r50_wt_coco_pretrain"
entity = 'tu-berlin-ods'
notes = 'lr 0.04, warmup_lr 0.0001, bs 4 trying to pretrain on bdd-100k dataset with coco type'
group = 'frcnn_r50'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
             project=project, 
             name=name, 
             entity=entity, 
             notes=notes, 
             group=group))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

runner = dict(type='EpochBasedRunner', max_epochs=36)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './outputs/frcnn/logs_bdd/'
out='./outputs/frcnn/logs_bdd/result/result_test.pkl'
resume_from = None
load_from = None
workflow = [('train', 1),('val', 1)]