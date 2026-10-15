"""Two-iteration smoke test config; inherits the real JZ Uni fine-tuning setup."""

_base_ = ["jz_wedetect_uni_finetune.py"]

train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=2,
    val_interval=100000,
)

# Do not spend the smoke test time on the full validation set.
custom_hooks = []
param_scheduler = []
default_hooks = dict(
    logger=dict(interval=1),
    checkpoint=dict(interval=100000),
)
