import argparse
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

try:
    from seg.train_unet_from_embeddings import (
        AugmentedTrainDataset,
        DeepLabLite,
        FeatureNormalizeDataset,
        FocalLossWithSampling,
        SoftDiceLoss,
        UNet,
        UNetDeep,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_segmentation_metrics,
        estimate_feature_channel_stats,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        resize_features_to,
        resize_labels_to,
        save_val_visualizations,
    )
except ImportError:
    from train_unet_from_embeddings import (
        AugmentedTrainDataset,
        DeepLabLite,
        FeatureNormalizeDataset,
        FocalLossWithSampling,
        SoftDiceLoss,
        UNet,
        UNetDeep,
        UNetResSE,
        align_logits_and_target_spatial,
        compute_segmentation_metrics,
        estimate_feature_channel_stats,
        estimate_sample_foreground_ratios,
        evaluate_per_class_iou,
        resize_features_to,
        resize_labels_to,
        save_val_visualizations,
    )

def _load_mat_key(path: Path, key: str) -> np.ndarray:
    data = scio.loadmat(path)
    if key in data:
        arr = data[key]
    else:
        candidates = [v for k, v in data.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        if len(candidates) != 1:
            raise KeyError(
                f"Key '{key}' not found in {path}, and cannot infer unique fallback key. "
                f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}"
            )
        arr = candidates[0]
    return np.asarray(arr)
