"""JZ defect-tile fine-tuning config for WeDetect-Uni Base.

The dataset is intentionally class-agnostic (one COCO category: ``object``),
matching the official Uni downstream fine-tuning setup.
"""

_base_ = ["../config/wedetect_uni_base_finetune.py"]

# Paths are inside the Docker container used for training.
load_from = "/workspace/wedetect/weights/wedetect_base_uni.pth"
work_dir = "/workspace/wedetect/work_dirs/jz_wedetect_uni_finetune"

jz_data_root = "/workspace/jz_uni_coco/"

# Verified on RTX 3060: batch 4 uses about 1.7 GB in the frozen-backbone setup.
train_batch_size_per_gpu = 4
base_lr = 2e-5
max_epochs = 4
close_mosaic_epochs = 2

optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=2,
    persistent_workers=False,
    dataset=dict(
        data_root=jz_data_root,
        ann_file="annotations/coco_instances_objectness.json",
        data_prefix=dict(img="train2017/"),
        metainfo=dict(classes=["object"]),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        data_root=jz_data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        metainfo=dict(classes=["object"]),
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=f"{jz_data_root}annotations/instances_val2017.json",
    metric="bbox",
)
test_evaluator = val_evaluator

# Keep the pretrained ConvNeXt frozen; train prompt, neck and detection head.
model = dict(
    backbone=dict(
        image_model=dict(frozen_modules=["all"]),
    ),
    neck=dict(freeze_all=False),
    bbox_head=dict(
        head_module=dict(freeze_all=False),
    ),
)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2),
)
