# AEF-seg：从多源遥感到分割训练的完整流程

本文档说明如何从 Google Earth Engine 下载 L8 / S2 / S1 多时相影像，利用 AlphaEarth Foundations (AEF) 生成 64 维（多时相）embedding，并在给定 annotation 的前提下，用 U-Net 进行下游分割训练。

## 1. 数据输入与整体假设

- 输入 shapefile：
  - 路径示例：`data/your_tiles.shp`（实际路径由你指定）。
  - **每个要素(feature) 是一个 128×128 像素的 patch 多边形**，覆盖同一 AOI 上的规则网格。
  - shapefile 中存在字段 `ID_patch`：
    - 第 i 个 patch 的 `ID_patch` = i（或其他唯一整数），用于在影像与标签之间建立对应关系。

- annotation 文件：
  - 颜色/ID 版：`data/label/ParcelIDs_10000.npy`（原始 10000 parcel/patch 对应标签编码）。
  - 类别配置：
    - `data/label/colormap.txt`：20 行 RGB 颜色 (0–19)。
    - `data/label/label_names.json`：0–19 的类别名称（Background / 作物类型 / Void label 等）。
  - 经过转换脚本后得到的像素级标签：
    - `data/label/ParcelIDs_10000_labels.npz`，包含：
      - `labels`：形状 (H, W) 的 int64，取值为 0–19。

- AEF 多源数据与模型：
  - 下载到 `data/gee_multi/` 下的 `sample_XXXXX.npz`（每个 XXXXX=ID_patch）。
  - 多源时序 embedding 生成在 `outputs_gee_multisource_tiny/embeddings_timeseries/`。

## 2. 从 shapefile 划分 patch 并下载 L8/S2/S1

下载脚本：`data/download_gee_l8_s1_s2.py`

### 2.1 主要改动与行为

- `load_aoi_tiles`：
  - 将 shapefile 读入后：
    - 计算全部几何的 union（`aoi_union`），用于 `filterBounds`；
    - 逐行遍历 feature，构造每个 patch 的 `ee.Geometry`，记为 `tile_geoms`；
    - 若存在字段 `ID_patch`，则读出为该 tile 的 `patch_id`；否则用顺序索引 0,1,2,...；
  - 返回 `(aoi_union, tile_geoms, patch_ids)`。

- 主循环中：
  - 不再随机采样空间点，而是**按 shapefile 中的 patch 顺序逐个 tile 处理**：
    - 对每个 tile：
      - 对一系列时间步（按 `--step_days`）寻找 L8/S2/S1 最近一景；
      - 使用 `sampleRectangle` 在该 tile 的 polygon 上截取 patch；
      - 若某源缺失则填 0；
      - 最多采集 `--max_timesteps` 个时间步；
    - 最终得到：
      - `landsat`：形状 (T, H, W, C_L8)
      - `sentinel2`：形状 (T, H, W, C_S2)
      - `sentinel1`：形状 (T, H, W, C_S1)
      - `timestamps`：形状 (T,) 的 ms 时间戳（网格时间）
      - `timestamps_iso`：可读字符串时间

- 输出命名：
  - 若 shapefile 中有 `ID_patch`，则：
    - 输出文件名为：`sample_{ID_patch:05d}.npz`，例如 `sample_00010.npz`。
    - npz 内额外包含 `patch_id` 字段，便于之后对齐 label。
  - 这样即可与 annotation 中的 patch/parcel 编号建立一一对应。

### 2.2 典型运行命令

在 `src` 目录下（或项目根）运行：

```bash
python -m data.download_gee_l8_s1_s2 \
  --aoi_shapefile D:\ProgramFiles\AEF\alphaearth-foundations\data\roi\metadata.shp \
  --start_date 2018-09-16 \
  --end_date   2019-12-07 \
  --patch_size 128 \
  --step_days 30 \
  --max_timesteps 64 \
  --output_dir D:/ProgramFiles/AEF/alphaearth-foundations/data/gee_multi \
  --sample_count -1 \
  --ee_project aef-489113
```

python -m data.download_gee_l8_s1_s2 `
--aoi_shapefile "D:\ProgramFiles\AEF\alphaearth-foundations\data\roi\metadata.shp" `
--start_date 2018-09-16 `
--end_date 2019-12-07 `
--patch_size 128 `
--step_days 5 `
--max_timesteps 64 `
--output_dir "D:\ProgramFiles\AEF\alphaearth-foundations\data\gee_multi1" `
--sample_count -1 `
--ee_project aef-489113


- `--sample_count -1`：使用 shapefile 中所有 patch；
- 每个 patch 生成一个 `sample_XXXXX.npz`；
- 后续多源训练和 embedding 会直接读取这些文件。

## 3. 多源 AEF 训练与重建

### 3.1 多源数据集与 DataLoader

代码位置：

- `src/alphaearth/data_gee_multisource.py`

主要功能：
 
  - `landsat` / `sentinel1` / `sentinel2`：形状 (T, H, W, C)
  - `timestamps`：形状 (T,)
- 对时间维做 padding / 对齐，归一化到 [0,1] 或标准范围；
- `__getitem__` 返回：
  - `source_data`：字典，三源张量
  - `timestamps`：时间戳（对齐后）
  - `valid_periods`：每个样本的起止时间 (t_start, t_end)

### 3.2 训练脚本与模型

训练入口：

- `src/alphaearth/run_train_gee_multisource.py`

关键参数：

- `--data_dir`：指向 `data/gee_multi`；
- `--model_size`：`tiny | small | base`，CPU 建议 `tiny`；
- `--batch_size`、`--max_steps`、`--warmup_steps` 等常规训练参数；
- `--reconstruction_weight` / `--uniformity_weight` / `--consistency_weight`：
  - 可根据重建 vs 表征的需求进行调节。

模型定义：

- `src/alphaearth/architecture/aef_module.py` 中的 `AlphaEarthFoundations`：
  - `input_sources = {"landsat": C_L8, "sentinel1": C_S1, "sentinel2": C_S2}`
  - `decode_sources` 同上；
  - `model_size` 控制 STP encoder 的维度与层数：
    - `tiny`：更小的 d_p/d_t/d_s 和 block 数，适合 CPU；
    - `small`：默认大小（与原实现一致）；
    - `base`：略大版本。

训练命令示例（CPU, tiny, 重建优先）：

```bash
python -m alphaearth.run_train_gee_multisource \
  --data_dir   D:/ProgramFiles/AEF/alphaearth-foundations/data/gee_multi \
  --output_dir D:/ProgramFiles/AEF/alphaearth-foundations/outputs_gee_multisource_tiny \
  --batch_size 1 \
  --num_workers 0 \
  --patch_size 128 \
  --model_size tiny \
  --max_steps 1000 \
  --warmup_steps 0 \
  --log_every 50 \
  --device cpu \
  --reconstruction_weight 1.0 \
  --uniformity_weight 0.05 \
  --consistency_weight 0.02
```
python -m alphaearth.run_train_gee_multisource `
--data_dir "D:/ProgramFiles/AEF/alphaearth-foundations/data/gee_multi1" `
--output_dir "D:/ProgramFiles/AEF/alphaearth-foundations/outputs_gee_multisource_tiny" `
--batch_size 1 `
--num_workers 0 `
--patch_size 128 `
--model_size tiny `
--max_steps 3000 `
--warmup_steps 0 `
--log_every 50 `
--device cpu `
--reconstruction_weight 1.0 `
--uniformity_weight 0.05 `
--consistency_weight 0.02



输出：

- `outputs_gee_multisource_tiny/checkpoint_latest.pt`：AEF 模型 checkpoint；
- `outputs_gee_multisource_tiny/reconstructions/*.png`：多源重建可视化；
- `outputs_gee_multisource_tiny/plots/`：loss 曲线。

## 4. 64 维 embedding 生成

推理脚本：`src/alphaearth/run_infer_gee_multisource.py`

### 4.1 整段时间 embedding（每 patch 一张）

- 使用 `summary_strategy=full_period`（默认）：

```bash
python -m alphaearth.run_infer_gee_multisource `
  --data_dir   D:/ProgramFiles/AEF/alphaearth-foundations/data/gee_multi `
  --checkpoint D:/ProgramFiles/AEF/alphaearth-foundations/outputs_gee_multisource_tiny/checkpoint_latest.pt `
  --output_dir D:/ProgramFiles/AEF/alphaearth-foundations/outputs_gee_multisource_tiny/embeddings `
  --batch_size 1 `
  --num_workers 0 `
  --patch_size 128 `
  --device cpu `
  --model_size tiny
```

- 对每个 `sample_XXXXX.npz` 生成 `embedding_YYYY.npz`，包含：
  - `embeddings`：(H', W', 64) 像素级 embedding；
  - `image_embedding`：(64,) 整个 patch 的 embedding；
  - `timestamps`：时间网格；
  - `tile_file`：对应原始 sample 文件路径。

### 4.2 多时相 embedding 时间序列

- 使用 `summary_strategy=per_timestamp`：

```bash
python -m alphaearth.run_infer_gee_multisource \
  --data_dir   D:/ProgramFiles/AEF/alphaearth-foundations/data/gee_multi \
  --checkpoint D:/ProgramFiles/AEF/alphaearth-foundations/outputs_gee_multisource_tiny/checkpoint_latest.pt \
  --output_dir D:/ProgramFiles/AEF/alphaearth-foundations/outputs_gee_multisource_tiny/embeddings_timeseries \
  --batch_size 1 \
  --num_workers 0 \
  --patch_size 128 \
  --device cpu \
  --model_size tiny \
  --summary_strategy per_timestamp \
  --max_time_steps 64
```

- 对每个 patch 生成 `embedding_timeseries_YYYY.npz`，包含：
  - `embeddings_per_time`：(T, H', W', 64)
  - `image_embeddings_per_time`：(T, 64)
  - `timestamps`：(T,)
  - `tile_file`：对应的 sample npz 路径。

## 5. annotation 处理与可视化

### 5.1 从 ParcelIDs_10000.npy 到像素级标签

脚本：`src/alphaearth/convert_label_to_indices.py`

行为：

- 支持三种情况：
  1. (H,W) 且取值已在 [0,19]：直接当作标签使用；
  2. (H,W) 且取值为大整数：
     - 视为 24 位打包 RGB（R<<16 | G<<8 | B），先解包为 (H,W,3) 颜色，再用 colormap 映射到 0–19；
  3. (H,W,3)：显式 RGB 颜色图，直接最近邻匹配 colormap 得到 0–19。

- 输出 `.npz`：
  - `labels`：(H,W) int64，取值 0–19；
  - 可选 `label_names`：从 label_names.json 读取。

#### 单文件示例（原始 10000 patch 编码）

```bash
python -m alphaearth.convert_label_to_indices `
  --label_npy D:/ProgramFiles/AEF/alphaearth-foundations/data/label/ParcelIDs_10000.npy `
  --colormap_txt D:/ProgramFiles/AEF/alphaearth-foundations/data/label/colormap.txt `
  --label_names_json D:/ProgramFiles/AEF/alphaearth-foundations/data/label/label_names.json `
  --output D:/ProgramFiles/AEF/alphaearth-foundations/data/label/ParcelIDs_10000_labels.npz
```

#### 批量示例（label 文件夹下的一系列 ParcelIDs_XXXXX.npy）

若 annotation 已经按 patch 拆分成多份 `ParcelIDs_XXXXX.npy`（例如 `ParcelIDs_00001.npy`、`ParcelIDs_00002.npy` 等），并且其中的 `XXXXX` 与 shapefile 中的 `ID_patch` 一一对应，可以使用目录批量转换：

```bash
python -m alphaearth.convert_label_to_indices `
  --label_dir D:/ProgramFiles/AEF/alphaearth-foundations/data/label `
  --colormap_txt D:/ProgramFiles/AEF/alphaearth-foundations/data/label/colormap.txt `
  --label_names_json D:/ProgramFiles/AEF/alphaearth-foundations/data/label/label_names.json `
  --output_dir D:/ProgramFiles/AEF/alphaearth-foundations/data/label/labels_npz
```

- 脚本会在 `--label_dir` 下查找所有 `ParcelIDs_*.npy` 文件；
- 对每个文件 `ParcelIDs_XXXXX.npy` 生成对应的 `ParcelIDs_XXXXX_labels.npz`（保存在 `--output_dir`，若未指定则默认写回 `--label_dir`）。

### 5.2 标签可视化

脚本：`src/alphaearth/visualize_labels.py`

- 将 `labels` + `colormap.txt` 渲染成彩色图（jpg/png），用于人工检查。

示例：

```bash
python -m alphaearth.visualize_labels \
  --labels_file D:/ProgramFiles/AEF/alphaearth-foundations/data/label/ParcelIDs_10000_labels.npz \
  --colormap_txt D:/ProgramFiles/AEF/alphaearth-foundations/data/label/colormap.txt \
  --output D:/ProgramFiles/AEF/alphaearth-foundations/data/label/ParcelIDs_10000_labels.jpg
```

## 6. 使用 64 维 embedding 进行 U-Net 分割训练

脚本：`src/alphaearth/train_unet_from_embeddings.py`

### 6.1 数据加载与对齐

- 读取 embedding：
  - 若文件含 `embeddings_per_time`：(T,H,W,64) → (T*64,H,W)，把时间维摊平成额外通道；
  - 若含 `embeddings`：(H,W,64) → (64,H,W)。
- 读取 label：
  - `.npz` 中的 `labels` 或 `.npy` 2D 整数 mask；
  - 若与 embedding 空间分辨率不同，使用最近邻插值到一致大小。
- 最终 Dataset 返回：
  - `features`：(C,H,W) float32（C = 64 或 T*64）；
  - `labels`：(H,W) int64，0–19，其中 19（Void）在 loss 中作为 `ignore_index`。

### 6.2 U-Net 结构

- 简化版 U-Net：
  - 编码器：3 层 (C→32→64→128)；
  - bottleneck：256 通道；
  - 解码器：3 层反卷积 + skip connection；
  - 输出层：`Conv2d(base_ch, num_classes)`，`num_classes=20`。

### 6.3 训练命令

```bash
python -m alphaearth.train_unet_from_embeddings `
  --embeddings_npz D:\ProgramFiles\AEF\alphaearth-foundations\outputs_gee_multisource_tiny\embeddings `
  --labels_file   D:\ProgramFiles\AEF\alphaearth-foundations\data\label\labels_npz `
  --output_dir    D:/ProgramFiles/AEF/alphaearth-foundations/outputs_seg_unet `
  --epochs 10 `
  --lr 1e-3 `
  --num_classes 20 `
  --ignore_index 19 `
  --base_channels 32 `
  --device cpu
```

- 控制台会打印：
  - `features` 与 `labels` 的 shape，对齐情况；
  - 每个 epoch 的 loss；
- 训练结束后：
  - `outputs_seg_unet/unet_from_embeddings_latest.pt`：U-Net checkpoint（包含 `model_state_dict` 与 `in_channels`）。

## 7. 将 pipeline 适配你的数据

1. **准备 shapefile**：
   - 确保每个 feature 是 128×128 patch polygon；
   - 添加 `ID_patch` 字段（int），与外部的 patch/parcel 编号一致；

2. **准备 annotation**：
   - 若已有 `ParcelIDs_10000.npy`（或类似）和 colormap + label_names：
     - 用 convert_label_to_indices 生成 `*_labels.npz`；
     - 用 visualize_labels 检查标签是否合理。

3. **下载多源时序影像**：
   - 用 download_gee_l8_s1_s2 处理整个 shapefile，输出 per-patch 的 sample_XXXXX.npz；

4. **训练 AEF 多源模型**：
   - 用 run_train_gee_multisource 在 gee_multi 数据上训练，获得 checkpoint；

5. **生成 64 维 embedding（含时间维）**：
   - 用 run_infer_gee_multisource 的 per_timestamp 模式生成 embedding_timeseries_XXXX.npz；

6. **构建分割训练数据 & 训练 U-Net**：
   - 选取若干 patch 的 embedding_timeseries_XXXX.npz + 对应标签（来自 *_labels.npz）；
   - 用 train_unet_from_embeddings 训练 U-Net，实现基于 AEF embedding 的分割。

