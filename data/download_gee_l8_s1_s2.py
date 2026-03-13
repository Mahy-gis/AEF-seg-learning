import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import ee
import geopandas as gpd
import numpy as np
from scipy.ndimage import zoom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download L8/S1/S2 patches from GEE into .npz files")
    parser.add_argument(
        "--aoi_shapefile",
        type=str,
        required=True,
        help=(
            "Path to a shapefile whose features are 128x128 (or similar) "
            "patch polygons. Each feature is treated as one tile. If the "
            "attribute 'ID_patch' exists, it will be used to name outputs."
        ),
    )
    parser.add_argument("--start_date", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2020-01-01", help="End date (YYYY-MM-DD, exclusive)")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size (pixels)")
    parser.add_argument("--step_days", type=int, default=30, help="Temporal step in days between samples")
    parser.add_argument("--max_timesteps", type=int, default=12, help="Maximum number of time steps per sample")
    parser.add_argument("--output_dir", type=str, default="./data/gee_multisource", help="Output directory for .npz files")
    parser.add_argument(
        "--sample_count",
        type=int,
        default=-1,
        help=(
            "Maximum number of tiles to process. -1 means use all patches "
            "from the shapefile (default)."
        ),
    )
    parser.add_argument("--ee_project", type=str, default=None, help="Google Cloud project ID for Earth Engine")
    parser.add_argument("--export_all_per_source", action="store_true", help="Also export per-source full time series npz files")
    parser.add_argument(
        "--patch_id_offset",
        type=int,
        default=0,
        help=(
            "Integer offset added to patch IDs before naming sample_XXXXX.npz. "
            "For example, use 10008 to start IDs from 10008 when the shapefile "
            "itself is indexed from 0."
        ),
    )
    return parser.parse_args()


def init_ee(project: Optional[str] = None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def load_aoi_tiles(shapefile: Path) -> tuple[ee.Geometry, List[ee.Geometry], List[int]]:
    """Load AOI shapefile and return union geometry plus per-tile geometries.

    The shapefile is expected to contain many small tiles (e.g. 128x128
    patches). We construct a global union geometry for filtering collections,
    and also a list of per-feature geometries for per-tile sampling.

    If the shapefile contains an attribute column named 'ID_patch', its value
    is used as the patch ID for that tile. Otherwise, a sequential index is
    used (0, 1, 2, ...). These IDs are then used when naming the
    sample_XXXX.npz files so that they can be aligned with external
    annotation files (e.g., ParcelIDs_10000.npy).
    """
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise ValueError(f"No geometries found in {shapefile}")

    gdf = gdf.to_crs(epsg=4326)

    # Union of all geometries for collection filtering.
    union_geom = gdf.union_all()
    if union_geom.geom_type == "MultiPolygon":
        union_coords = [np.array(p.exterior.coords).tolist() for p in union_geom.geoms]
        aoi_union = ee.Geometry.MultiPolygon(union_coords)
    else:
        union_coords = np.array(union_geom.exterior.coords).tolist()
        aoi_union = ee.Geometry.Polygon(union_coords)

    # Individual tile geometries and their patch IDs.
    tile_geoms: List[ee.Geometry] = []
    patch_ids: List[int] = []

    has_id_patch = "id" in gdf.columns

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Determine patch ID for this tile.
        if has_id_patch:
            try:
                patch_id = int(row["id"])
            except Exception:
                patch_id = len(patch_ids)
        else:
            patch_id = len(patch_ids)

        if geom.geom_type == "MultiPolygon":
            coords = [np.array(p.exterior.coords).tolist() for p in geom.geoms]
            tile_geoms.append(ee.Geometry.MultiPolygon(coords))
        else:
            coords = np.array(geom.exterior.coords).tolist()
            tile_geoms.append(ee.Geometry.Polygon(coords))
        patch_ids.append(patch_id)

    if not tile_geoms:
        raise ValueError("No valid polygon geometries found in shapefile")

    return aoi_union, tile_geoms, patch_ids


def build_collections(aoi: ee.Geometry, start: str, end: str):
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", 5))
    )

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
    )

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    )

    return l8, s2, s1


def date_range(start: ee.Date, end: ee.Date, step_days: int) -> List[ee.Date]:
    dates: List[ee.Date] = []
    cur = start
    while cur.millis().lt(end.millis()).getInfo():  # type: ignore[arg-type]
        dates.append(cur)
        cur = cur.advance(step_days, "day")
    return dates


def get_nearest_image(col: ee.ImageCollection, target: ee.Date, window_days: int = 16) -> ee.Image | None:
    start = target.advance(-window_days, "day")
    end = target.advance(window_days, "day")
    subset = col.filterDate(start, end).sort("system:time_start")
    size = subset.size().getInfo()
    if size == 0:
        return None
    return ee.Image(subset.first())


def sample_patch(image: ee.Image, aoi: ee.Geometry, patch_size: int, bands: List[str]) -> np.ndarray | None:
    """Sample a rectangular patch and resample to (patch_size, patch_size).

    Uses Image.sampleRectangle with default parameters (no defaultProjection/dimensions,
    for compatibility with the installed ee API), then rescales on the client side.
    """
    image = image.select(bands)
    region = aoi
    try:
        data = image.sampleRectangle(region=region, defaultValue=0).getInfo()
    except Exception as e:
        print(f"sampleRectangle error for bands {bands}: {e}")
        return None

    if not isinstance(data, dict) or not data:
        return None

    # Cloud EE often returns a Feature-like dict with pixel values in "properties".
    props = data.get("properties", None)
    if props is None:
        props = data

    # Debug one example of the returned structure to ensure correct parsing.
    if not hasattr(sample_patch, "_debug_printed") or not getattr(sample_patch, "_debug_printed"):
        print("sampleRectangle keys:", list(data.keys()))
        print("properties keys:", list(props.keys()))
        sample_patch._debug_printed = True  # type: ignore[attr-defined]

    # For multi-band images, properties maps band names to nested lists (H, W).
    # Different bands can have different native resolutions (e.g. Sentinel-2),
    # so we crop all bands to the minimal common H,W before stacking.
    band_arrays: List[np.ndarray] = []
    for b in bands:
        if b not in props:
            return None
        arr = props[b]
        # Some APIs wrap arrays as {"array": [[...]]}
        if isinstance(arr, dict) and "array" in arr:
            arr = arr["array"]
        arr_b = np.array(arr)
        if arr_b.ndim != 2:
            return None
        band_arrays.append(arr_b)

    # Ensure all bands share the same spatial shape via cropping.
    hs = [a.shape[0] for a in band_arrays]
    ws = [a.shape[1] for a in band_arrays]
    min_h = min(hs)
    min_w = min(ws)
    band_arrays = [a[:min_h, :min_w] for a in band_arrays]

    array = np.stack(band_arrays, axis=-1)  # (H, W, C)
    h, w, c = array.shape
    if h != patch_size or w != patch_size:
        zoom_factors = (patch_size / h, patch_size / w, 1.0)
        array = zoom(array, zoom_factors, order=1)

    return array.astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_ee(args.ee_project)
    aoi_union, tile_geoms, patch_ids = load_aoi_tiles(Path(args.aoi_shapefile))

    # Optionally shift patch IDs by a user-specified offset, so that
    # generated sample_XXXXX.npz files can continue from a given ID
    # (e.g., start from 10008 instead of 0).
    if getattr(args, "patch_id_offset", 0) != 0:
        patch_ids = [int(pid) + int(args.patch_id_offset) for pid in patch_ids]

    l8_col, s2_col, s1_col = build_collections(aoi_union, args.start_date, args.end_date)

    # Simple diagnostics to help understand when no files are produced.
    try:
        print("L8 images in range:", l8_col.size().getInfo())
        print("S2 images in range:", s2_col.size().getInfo())
        print("S1 images in range:", s1_col.size().getInfo())
    except Exception:
        pass

    l8_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    s2_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    s1_bands = ["VV", "VH"]

    start_date = ee.Date(args.start_date)
    end_date = ee.Date(args.end_date)
    dates = date_range(start_date, end_date, args.step_days)

    # Limit number of tiles if sample_count is set (>0). -1 means use all.
    if args.sample_count is not None and args.sample_count > 0:
        max_tiles = min(args.sample_count, len(tile_geoms))
    else:
        max_tiles = len(tile_geoms)

    # Process tiles in the order they appear in the shapefile so that
    # patch_ids align with external annotation indices.
    for local_idx in range(max_tiles):
        tile_geom = tile_geoms[local_idx]
        patch_id = patch_ids[local_idx]
        landsat_list: List[np.ndarray] = []
        s2_list: List[np.ndarray] = []
        s1_list: List[np.ndarray] = []
        ts_list: List[float] = []

        for d in dates:
            l8_img = get_nearest_image(l8_col, d)
            s2_img = get_nearest_image(s2_col, d)
            s1_img = get_nearest_image(s1_col, d)

            if l8_img is not None:
                l8_patch = sample_patch(l8_img, tile_geom, args.patch_size, l8_bands)
            else:
                l8_patch = None
            if s2_img is not None:
                s2_patch = sample_patch(s2_img, tile_geom, args.patch_size, s2_bands)
            else:
                s2_patch = None
            if s1_img is not None:
                s1_patch = sample_patch(s1_img, tile_geom, args.patch_size, s1_bands)
            else:
                s1_patch = None

            # 只保留三个数据源在该时间点附近都能找到影像的时间步，
            # 避免用 0 填补缺失源，减少全 0 patch 数量。
            if l8_patch is None or s2_patch is None or s1_patch is None:
                continue

            # Use grid time d as the reference timestamp (ms), to ensure
            # strictly increasing time steps even if multiple grid dates
            # map to the same acquisition.
            ts_ms = d.millis().getInfo()

            landsat_list.append(l8_patch)
            s2_list.append(s2_patch)
            s1_list.append(s1_patch)
            ts_list.append(float(ts_ms))

            if len(landsat_list) >= args.max_timesteps:
                break

        if not landsat_list:
            continue

        landsat_arr = np.stack(landsat_list, axis=0)
        s2_arr = np.stack(s2_list, axis=0)
        s1_arr = np.stack(s1_list, axis=0)
        ts_arr = np.array(ts_list, dtype=np.float64)

        # Also save human-readable ISO timestamps for convenience.
        ts_iso = np.array(
            [datetime.utcfromtimestamp(t / 1000.0).strftime("%Y-%m-%dT%H:%M:%S") for t in ts_arr],
            dtype="U19",
        )

        # Use patch_id in the filename so it can be matched to annotations
        # (e.g., ID_patch or entries in ParcelIDs_10000.npy).
        out_path = output_dir / f"sample_{patch_id:05d}.npz"
        np.savez(
            out_path,
            landsat=landsat_arr,
            sentinel2=s2_arr,
            sentinel1=s1_arr,
            timestamps=ts_arr,
            timestamps_iso=ts_iso,
            patch_id=int(patch_id),
        )
        print(
            f"Saved {out_path} (patch_id={patch_id}) with shape "
            f"L8 {landsat_arr.shape}, S2 {s2_arr.shape}, S1 {s1_arr.shape}"
        )

    # Optionally, export full per-source time series (all time points,
    # not aligned across sources) into separate npz files.
    if args.export_all_per_source:
        def export_series(name: str, col: ee.ImageCollection, bands: List[str]) -> None:
            size = col.size().getInfo()
            if size == 0:
                return
            img_list = col.toList(size)
            series_patches: List[np.ndarray] = []
            series_ts: List[float] = []
            for i in range(size):
                img = ee.Image(img_list.get(i))
                # Use union AOI for full-series export.
                patch = sample_patch(img, aoi_union, args.patch_size, bands)
                if patch is None:
                    continue
                ts = img.date().millis().getInfo()
                series_patches.append(patch)
                series_ts.append(float(ts))
            if not series_patches:
                return
            arr = np.stack(series_patches, axis=0)  # (T, H, W, C)
            ts_arr = np.array(series_ts, dtype=np.float64)
            ts_iso = np.array(
                [datetime.utcfromtimestamp(t / 1000.0).strftime("%Y-%m-%dT%H:%M:%S") for t in ts_arr],
                dtype="U19",
            )
            out_path = output_dir / f"all_{name}.npz"
            np.savez(
                out_path,
                images=arr,
                timestamps=ts_arr,
                timestamps_iso=ts_iso,
            )
            print(f"Saved full series {out_path} with shape {arr.shape}")

        export_series("landsat", l8_col, l8_bands)
        export_series("sentinel2", s2_col, s2_bands)
        export_series("sentinel1", s1_col, s1_bands)


if __name__ == "__main__":
    main()
