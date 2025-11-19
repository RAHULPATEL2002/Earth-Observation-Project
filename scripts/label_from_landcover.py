#!/usr/bin/env python3
"""
Extract land cover labels from ESA WorldCover raster for each image location.

Reads metadata.csv, opens the WorldCover TIF, reprojects coordinates if needed,
extracts 128x128 pixel patches, computes modal ESA code and dominance ratio.
"""

import argparse
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_geom, calculate_default_transform
from rasterio.windows import Window
from rasterio.transform import from_bounds
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ESA WorldCover code to label mapping
ESA_CODE_TO_LABEL = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare/sparse',
    70: 'Snow/ice',
    80: 'Water',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss/lichen'
}


def get_patch_at_location(ds, lat, lon, patch_size=128):
    """
    Extract a patch_size x patch_size pixel patch centered at (lat, lon).
    
    Uses boundless=True and fill_value=nodata to handle edge cases.
    Returns (patch_array, nodata_value)
    """
    # Get CRS
    if ds.crs is None:
        raise ValueError("land_cover.tif missing CRS — add .prj or reproject the raster")
    
    # Reproject lat/lon to raster CRS
    from rasterio.crs import CRS
    src_crs = CRS.from_epsg(4326)  # WGS84
    dst_crs = ds.crs
    
    geom = {'type': 'Point', 'coordinates': [lon, lat]}
    transformed = transform_geom(src_crs, dst_crs, geom)
    x, y = transformed['coordinates']
    
    # Convert to pixel coordinates
    row, col = ds.index(x, y)
    
    # Extract patch (centered at row, col)
    half_size = patch_size // 2
    window = Window(
        col_off=col - half_size,
        row_off=row - half_size,
        width=patch_size,
        height=patch_size
    )
    
    # Read with boundless=True
    nodata = ds.nodata if ds.nodata is not None else 0
    patch = ds.read(1, window=window, boundless=True, fill_value=nodata)
    
    return patch, nodata


def compute_label_and_dominance(patch, nodata):
    """
    Compute modal ESA code and dominance ratio from patch.
    
    Returns (assigned_code, dominance, comment)
    """
    # Mask nodata values
    valid_mask = patch != nodata
    valid_pixels = patch[valid_mask]
    
    if len(valid_pixels) == 0:
        return None, 0.0, "all_nodata"
    
    # Compute mode
    mode_result = stats.mode(valid_pixels, keepdims=True)
    assigned_code = int(mode_result.mode[0])
    mode_count = int(mode_result.count[0])
    
    # Compute dominance
    valid_count = len(valid_pixels)
    dominance = mode_count / valid_count if valid_count > 0 else 0.0
    
    comment = f"mode_count={mode_count}, valid={valid_count}"
    
    return assigned_code, dominance, comment


def process_single_record(args_tuple):
    """Process a single metadata record. Returns dict with label info."""
    record, tif_path, patch_size = args_tuple
    
    try:
        with rasterio.open(tif_path) as ds:
            patch, nodata = get_patch_at_location(
                ds, record['lat'], record['lon'], patch_size
            )
            assigned_code, dominance, comment = compute_label_and_dominance(patch, nodata)
            
            if assigned_code is None:
                label = "nodata"
                keep = False
            else:
                label = ESA_CODE_TO_LABEL.get(assigned_code, f"Unknown_{assigned_code}")
                keep = True
            
            return {
                'filename': record['filename'],
                'lat': record['lat'],
                'lon': record['lon'],
                'assigned_code': assigned_code if assigned_code is not None else -1,
                'label': label,
                'dominance': dominance,
                'comment': comment,
                'keep': keep
            }
    except Exception as e:
        return {
            'filename': record['filename'],
            'lat': record['lat'],
            'lon': record['lon'],
            'assigned_code': -1,
            'label': 'error',
            'dominance': 0.0,
            'comment': f"error: {str(e)}",
            'keep': False
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract land cover labels from ESA WorldCover raster'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='Dataset/metadata.csv',
        help='Input metadata CSV (default: Dataset/metadata.csv)'
    )
    parser.add_argument(
        '--tif',
        type=str,
        default='Data/worldcover_bbox_delhi_ncr_2021.tif',
        help='WorldCover TIF path (default: Data/worldcover_bbox_delhi_ncr_2021.tif)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=128,
        help='Patch size in pixels (default: 128)'
    )
    parser.add_argument(
        '--dominance-threshold',
        type=float,
        default=0.0,
        help='Minimum dominance to keep (default: 0.0)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Dataset/labels.csv',
        help='Output labels CSV (default: Dataset/labels.csv)'
    )
    
    args = parser.parse_args()
    
    # Read metadata
    if not os.path.exists(args.metadata):
        print(f"[label_from_landcover][ERROR] Metadata file not found: {args.metadata}")
        exit(1)
    
    print(f"[label_from_landcover] Reading metadata from {args.metadata}...")
    metadata = pd.read_csv(args.metadata)
    print(f"[label_from_landcover] Loaded {len(metadata)} records")
    
    # Check TIF exists and has CRS
    if not os.path.exists(args.tif):
        print(f"[label_from_landcover][ERROR] TIF file not found: {args.tif}")
        exit(1)
    
    print(f"[label_from_landcover] Opening raster: {args.tif}")
    with rasterio.open(args.tif) as ds:
        if ds.crs is None:
            print(f"[label_from_landcover][ERROR] land_cover.tif missing CRS — add .prj or reproject the raster")
            exit(1)
        print(f"[label_from_landcover] Raster CRS: {ds.crs}, shape: {ds.shape}")
    
    # Process records
    print(f"[label_from_landcover] Processing {len(metadata)} records with {args.threads} threads...")
    
    records_list = metadata.to_dict('records')
    args_list = [(r, args.tif, args.patch_size) for r in records_list]
    
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_single_record, args_tuple): args_tuple 
                   for args_tuple in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Apply dominance threshold
    if args.dominance_threshold > 0.0:
        df.loc[df['dominance'] < args.dominance_threshold, 'keep'] = False
        print(f"[label_from_landcover] Applied dominance threshold {args.dominance_threshold}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    # Print summary
    total = len(df)
    dropped = len(df[~df['keep']])
    processed = total - dropped
    
    print(f"[label_from_landcover] Summary:")
    print(f"  Images processed: {total}")
    print(f"  Dropped (keep=False): {dropped}")
    print(f"  Kept: {processed}")
    
    print(f"\n[label_from_landcover] Dominance distribution:")
    print(df['dominance'].describe())
    
    print(f"\n[label_from_landcover] Label distribution:")
    print(df['label'].value_counts())
    
    print(f"\n[label_from_landcover] Sample labeled records:")
    print(df[df['keep']].head().to_string())
    
    print(f"\n[label_from_landcover] Saved labels to {args.output}")


if __name__ == '__main__':
    main()
