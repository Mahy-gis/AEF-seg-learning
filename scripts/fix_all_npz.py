import os
import numpy as np
from datetime import datetime

# 目标数据目录
DATA_DIR = "/mnt/data/RSFM/AEF-seg-learning/data/nonagri/gee_multi/image"
BACKUP_DIR = os.path.join(DATA_DIR, "backup_all_fixes")
os.makedirs(BACKUP_DIR, exist_ok=True)

SOURCES = ["sentinel1", "sentinel2", "landsat"]

def fix_file(fpath):
    changed = False
    data = np.load(fpath)
    arr = dict(data.items())
    # 1. 修复timestamps
    if "timestamps" not in arr and "date" in arr:
        date_val = arr["date"]
        if isinstance(date_val, np.ndarray):
            date_val = date_val.item()
        ts = None
        if isinstance(date_val, (int, np.integer)):
            sval = str(date_val)
            if len(sval) == 8 and sval.isdigit():
                try:
                    dt = datetime.strptime(sval, "%Y%m%d")
                    ts = int(dt.timestamp() * 1000)
                except Exception:
                    pass
            elif date_val > 1e12:
                ts = int(date_val)
        elif isinstance(date_val, (str, bytes)):
            s = date_val.decode() if isinstance(date_val, bytes) else date_val
            if len(s) == 8 and s.isdigit():
                try:
                    dt = datetime.strptime(s, "%Y%m%d")
                    ts = int(dt.timestamp() * 1000)
                except Exception:
                    pass
            else:
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d")
                    ts = int(dt.timestamp() * 1000)
                except Exception:
                    try:
                        dt = datetime.strptime(s, "%Y/%m/%d")
                        ts = int(dt.timestamp() * 1000)
                    except Exception:
                        pass
        if ts is not None:
            arr["timestamps"] = np.array([ts], dtype=np.int64)
            changed = True
    # 2. 修复shape
    for src in SOURCES:
        if src in arr:
            val = arr[src]
            if val.ndim == 3:
                arr[src] = val[None, ...]
                changed = True
    # 3. 修正timestamps shape
    if "timestamps" in arr:
        ts = arr["timestamps"]
        if ts.shape == () or ts.shape == (1,):
            arr["timestamps"] = np.array([ts.item()], dtype=np.int64)
            changed = True
    return arr, changed

fixed_count = 0
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".npz"):
        continue
    fpath = os.path.join(DATA_DIR, fname)
    try:
        arr, changed = fix_file(fpath)
        if changed:
            os.rename(fpath, os.path.join(BACKUP_DIR, fname))
            np.savez(fpath, **arr)
            print(f"修复: {fname}")
            fixed_count += 1
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"共修复 {fixed_count} 个文件 (timestamps/shape)")
