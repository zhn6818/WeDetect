# WeDetect-Uni JZ 数据微调记录

## 目标

使用 `/data2/dataset/JZ_balance_data/img_json_split_512/` 的 JZ 缺陷切片数据，
对 `weights/wedetect_base_uni.pth` 进行 WeDetect-Uni Base 微调。

## 数据整理结果

- 原始数据：5388 张 JPG + 5388 个 LabelMe JSON。
- 标注类型：LabelMe polygon。
- 类别：`sulfide_a`、`globular_oxide_d`、`alumina_b`、`silicate_c`、`titanium_d`、`titanium_b`、`spherical_oxide_ds`。
- 目标总数：25304。
- 已转换为 COCO bbox 格式，输出目录：`/data2/dataset/JZ_balance_data/wedetect_uni_coco/`。
- 按原始大图编号分组划分，避免同一大图切片泄漏：训练集 4298 张，验证集 1090 张。
- 当前 Uni 微调采用官方的类无关 objectness 方式：7 个原始类别统一映射为 `object`，COCO `category_id=0`。
- COCO API 已验证：训练集 4298 张/19900 个目标，验证集 1090 张/5404 个目标。



## 当前配置

- 配置文件：`finetune_uni/jz_wedetect_uni_finetune.py`。
- 模型：WeDetect-Uni Base。
- 权重：`weights/wedetect_base_uni.pth`。
- 输入：640x640。
- Backbone：ConvNeXt-Base，冻结。
- 可训练部分：Prompt、Neck、Detection Head。
- RTX 3060 已验证参数：batch size 4，学习率 2e-5，4 epochs，AMP。
- batch size 4 冒烟测试显存约 1.73 GB，无 OOM；因此正式训练使用 batch size 4。



## 问题与处理



### 已解决

1. 原始标注为 LabelMe polygon，项目 Uni 训练读取 COCO bbox；已转换为 COCO。
2. 当前 Uni downstream 配置要求单类 `object`；已将 7 个类别映射为 `object`，并保留 `source_label` 追溯字段。
3. 项目 README 要求 objectness 标注的 `category_id` 为 0；已修正并通过 COCO API 校验。
4. 原始图片由同一大图切片而来；已按 `_rNcM` 前的原图编号分组切分，避免训练/验证泄漏。
5. 当前容器内训练数据和权重路径与独立配置保持一致：`/workspace/jz_uni_coco/`、`/workspace/wedetect/weights/wedetect_base_uni.pth`。
6. 配置继承后 `optim_wrapper.optimizer.lr` 不会自动跟随 `base_lr` 变量，已显式覆盖学习率。
7. 原配置验证器使用 `proposal_fast`，在本数据的可变目标数上触发 NumPy shape 错误；已切换为标准 COCO `bbox` 评估。



### 已验证

- 数据集已被 MMEngine/MMDetection 完整构建。
- 单步前向、反向和 checkpoint 保存正常。
- RTX 3060 使用 batch=4，训练峰值显存约 1.9 GB，无 OOM。
- 完整训练和验证已完成，最终 checkpoint 参数无 NaN/Inf。

## 冒烟测试

### batch=1

- 2 个训练 iteration 成功完成，loss 有限：`loss=14.1618`、`11.7011`。
- 完整验证集 1090 张成功完成。
- `bbox_mAP=0.024`，`bbox_mAP_50=0.072`，`AR@300=0.138`。
- 验证器原先为 `proposal_fast`，出现 `ValueError: setting an array element with a sequence`；改成 `metric='bbox'` 后解决。

### batch=4

- 2 个训练 iteration 成功完成，loss 有限：`loss=25.0638`、`27.9015`。
- 峰值显存约 1727 MB，无 CUDA OOM。
- 完整验证集成功完成：`bbox_mAP=0.025`，`bbox_mAP_50=0.075`，`AR@300=0.138`。
- checkpoint 成功保存：`work_dirs/jz_uni_smoke_b4/epoch_1.pth`。

## 正式微调

运行配置：`finetune_uni/jz_wedetect_uni_finetune.py`。

```bash
docker run --rm --gpus all \
  -v /data2/code/cv/wedetect:/workspace/wedetect \
  -v /data2/dataset/JZ_balance_data/wedetect_uni_coco:/workspace/jz_uni_coco:ro \
  wedetect:cu124 \
  /opt/miniconda3/envs/ai/bin/python train.py \
  finetune_uni/jz_wedetect_uni_finetune.py \
  --amp \
  --work-dir /workspace/wedetect/work_dirs/jz_wedetect_uni_finetune
```

训练结果：

| Epoch | bbox mAP | bbox mAP50 | bbox mAP75 | small mAP | AR@300 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.288 | 0.552 | 0.264 | 0.256 | 0.509 |
| 2 | 0.322 | 0.610 | 0.298 | 0.285 | 0.522 |
| 3 | 0.329 | 0.622 | 0.300 | 0.295 | 0.524 |
| 4 | 0.347 | 0.633 | 0.331 | 0.304 | 0.534 |

最终权重：

- `work_dirs/jz_wedetect_uni_finetune/epoch_4.pth`
- `work_dirs/jz_wedetect_uni_finetune/last_checkpoint`

独立测试命令使用 `test.py` 从 `epoch_4.pth` 重新加载，得到与训练结束时相同的结果：

```text
bbox_mAP=0.347
bbox_mAP_50=0.633
bbox_mAP_75=0.331
bbox_mAP_s=0.304
AR@300=0.534
```

日志文件：

- `finetune_full.log`
- `finetune_test.log`
- `finetune_smoke_b4.log`

## 训练中的非致命警告

- `albumentations` 尝试联网检查版本时超时，不影响训练。
- PyTorch 的 `torch.load(weights_only=False)` 和 `torch.meshgrid` 有版本兼容警告，不影响结果。
- 训练日志中 `grad_norm` 大多数较大，单个 iteration 出现过 `inf`；对应 loss 始终为有限值，后续训练正常，最终 checkpoint 扫描确认 910 个 state tensor 均无 NaN/Inf。配置已有 `clip_grad.max_norm=10.0`。
- 原配置的 `proposal_fast` 评估器在本数据上触发 NumPy 不规则数组错误，已改成标准 COCO `bbox` 评估，训练和独立测试均通过。



## 运行记录

（后续追加命令、日志、错误和修复结果。）
