#!/usr/bin/env python3
"""
Extract lat/lon coordinates from RGB image filenames and create metadata.csv.

Scans Data/rgb/ for .png files and attempts to parse coordinates from filenames
using common patterns. If parsing fails, exits with error message.
"""

import argparse
import os
import re
from pathlib import Path
import pandas as pd


def parse_coordinates_from_filename(filename):
    """
    Try to extract lat/lon from filename using multiple patterns.
    
    Patterns supported:
    - lat_lon.png (e.g., 28.7041_77.1025.png)
    - img_lat_lon.png (e.g., img_28.7041_77.1025.png)
    - lat_lon_img.png (e.g., 28.7041_77.1025_img.png)
    - tile_latN_lonE.png (e.g., tile_28.7041N_77.1025E.png)
    
    Returns (lat, lon) tuple or (None, None) if parsing fails.
    """
    # Remove extension
    base = Path(filename).stem
    
    # Pattern 1: lat_lon (e.g., 28.7041_77.1025)
    pattern1 = r'^([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)$'
    match = re.match(pattern1, base)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    
    # Pattern 2: prefix_lat_lon (e.g., img_28.7041_77.1025)
    pattern2 = r'^[^_]+_([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)$'
    match = re.match(pattern2, base)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    
    # Pattern 3: lat_lon_suffix (e.g., 28.7041_77.1025_img)
    pattern3 = r'^([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)_[^_]+$'
    match = re.match(pattern3, base)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    
    # Pattern 4: tile_latN_lonE (e.g., tile_28.7041N_77.1025E)
    pattern4 = r'^[^_]+_([+-]?\d+\.?\d*)[NS]_([+-]?\d+\.?\d*)[EW]$'
    match = re.match(pattern4, base, re.IGNORECASE)
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            # Check for S/W to negate
            if 'S' in base.upper():
                lat = -lat
            if 'W' in base.upper():
                lon = -lon
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Extract lat/lon from RGB image filenames and create metadata.csv'
    )
    parser.add_argument(
        '--rgb-dir',
        type=str,
        default='Data/rgb',
        help='Directory containing RGB PNG images (default: Data/rgb)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Dataset/metadata.csv',
        help='Output metadata CSV path (default: Dataset/metadata.csv)'
    )
    parser.add_argument(
        '--metadata-csv',
        type=str,
        default=None,
        help='Optional: use existing metadata CSV instead of parsing filenames'
    )
    
    args = parser.parse_args()
    
    # If metadata CSV provided, just copy it
    if args.metadata_csv and os.path.exists(args.metadata_csv):
        print(f"[create_metadata] Using provided metadata CSV: {args.metadata_csv}")
        df = pd.read_csv(args.metadata_csv)
        required_cols = ['filename', 'lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            print(f"[create_metadata][ERROR] CSV must contain columns: {required_cols}")
            exit(1)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df[required_cols].to_csv(args.output, index=False)
        print(f"[create_metadata] Saved {len(df)} records to {args.output}")
        print(f"[create_metadata] Sample rows:\n{df[required_cols].head()}")
        return
    
    # Scan RGB directory
    rgb_dir = Path(args.rgb_dir)
    if not rgb_dir.exists():
        print(f"[create_metadata][ERROR] RGB directory not found: {rgb_dir}")
        exit(1)
    
    print(f"[create_metadata] Scanning {rgb_dir} for PNG files...")
    png_files = list(rgb_dir.glob('*.png'))
    print(f"[create_metadata] Found {len(png_files)} PNG files")
    
    if len(png_files) == 0:
        print(f"[create_metadata][ERROR] No PNG files found in {rgb_dir}")
        exit(1)
    
    # Parse coordinates from filenames
    records = []
    parsed_count = 0
    
    for png_file in png_files:
        filename = png_file.name
        lat, lon = parse_coordinates_from_filename(filename)
        
        if lat is not None and lon is not None:
            records.append({
                'filename': filename,
                'lat': lat,
                'lon': lon
            })
            parsed_count += 1
    
    if parsed_count == 0:
        print(f"[create_metadata][ERROR] No coordinates detected in filenames — provide Dataset/metadata.csv with columns filename,lat,lon")
        exit(1)
    
    print(f"[create_metadata] Successfully parsed coordinates from {parsed_count}/{len(png_files)} files")
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"[create_metadata] Saved {len(df)} records to {args.output}")
    print(f"[create_metadata] Sample rows:")
    print(df.head().to_string())
    print(f"[create_metadata] Total count: {len(df)}")


if __name__ == '__main__':
    main()
