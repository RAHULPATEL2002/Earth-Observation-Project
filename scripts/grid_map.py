#!/usr/bin/env python3
"""
Create a grid map visualization of the dataset.

Reads metadata.csv, reprojects to EPSG:32644, creates a 60x60 km grid,
and saves HTML and PNG visualizations.
"""

import argparse
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import folium
from folium import plugins
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Create grid map visualization')
    parser.add_argument('--metadata', type=str, default='Dataset/metadata.csv',
                       help='Metadata CSV (default: Dataset/metadata.csv)')
    parser.add_argument('--html-out', type=str, default='Visualizations/grid_map.html',
                       help='Output HTML path (default: Visualizations/grid_map.html)')
    parser.add_argument('--png-out', type=str, default='Visualizations/grid_map.png',
                       help='Output PNG path (default: Visualizations/grid_map.png)')
    parser.add_argument('--grid-size', type=float, default=60.0,
                       help='Grid cell size in km (default: 60.0)')
    
    args = parser.parse_args()
    
    try:
        # Read metadata
        print(f"[grid_map] Reading metadata from {args.metadata}...")
        if not os.path.exists(args.metadata):
            print(f"[grid_map][WARNING] Metadata file not found: {args.metadata}")
            return
        
        df = pd.read_csv(args.metadata)
        print(f"[grid_map] Loaded {len(df)} records")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])],
            crs='EPSG:4326'
        )
        
        # Reproject to EPSG:32644 (UTM Zone 44N for Delhi region)
        print(f"[grid_map] Reprojecting to EPSG:32644...")
        gdf_utm = gdf.to_crs('EPSG:32644')
        
        # Get bounds
        bounds = gdf_utm.total_bounds
        minx, miny, maxx, maxy = bounds
        print(f"[grid_map] Bounds (UTM): minx={minx:.0f}, miny={miny:.0f}, maxx={maxx:.0f}, maxy={maxy:.0f}")
        
        # Create grid (60x60 km cells)
        grid_size_m = args.grid_size * 1000  # Convert km to meters
        
        x_min = np.floor(minx / grid_size_m) * grid_size_m
        y_min = np.floor(miny / grid_size_m) * grid_size_m
        x_max = np.ceil(maxx / grid_size_m) * grid_size_m
        y_max = np.ceil(maxy / grid_size_m) * grid_size_m
        
        x_cells = int((x_max - x_min) / grid_size_m)
        y_cells = int((y_max - y_min) / grid_size_m)
        
        print(f"[grid_map] Creating {x_cells}x{y_cells} grid ({args.grid_size} km cells)...")
        
        grid_cells = []
        for i in range(x_cells):
            for j in range(y_cells):
                x1 = x_min + i * grid_size_m
                y1 = y_min + j * grid_size_m
                x2 = x1 + grid_size_m
                y2 = y1 + grid_size_m
                grid_cells.append(box(x1, y1, x2, y2))
        
        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs='EPSG:32644')
        
        # Count points per grid cell
        grid_gdf['count'] = grid_gdf.geometry.apply(
            lambda cell: len(gdf_utm[gdf_utm.geometry.within(cell)])
        )
        
        # Reproject back to WGS84 for visualization
        grid_gdf_wgs84 = grid_gdf.to_crs('EPSG:4326')
        gdf_wgs84 = gdf_utm.to_crs('EPSG:4326')
        
        # Create HTML map with folium
        print(f"[grid_map] Creating HTML map...")
        center_lat = gdf_wgs84.geometry.y.mean()
        center_lon = gdf_wgs84.geometry.x.mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add grid cells
        for idx, row in grid_gdf_wgs84.iterrows():
            if row['count'] > 0:
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda feature, count=row['count']: {
                        'fillColor': 'blue' if count > 0 else 'white',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': min(0.5, count / 10.0)
                    },
                    tooltip=f"Count: {row['count']}"
                ).add_to(m)
        
        # Add points
        for idx, row in gdf_wgs84.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                popup=f"Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}",
                color='red',
                fill=True
            ).add_to(m)
        
        os.makedirs(os.path.dirname(args.html_out), exist_ok=True)
        m.save(args.html_out)
        print(f"[grid_map] Saved HTML map to {args.html_out}")
        
        # Create PNG visualization
        print(f"[grid_map] Creating PNG visualization...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot grid
        grid_gdf_wgs84.plot(ax=ax, column='count', cmap='Blues', edgecolor='black', 
                           linewidth=0.5, legend=True, alpha=0.6)
        
        # Plot points
        gdf_wgs84.plot(ax=ax, color='red', markersize=5, alpha=0.7, label='Image locations')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Grid Map ({args.grid_size} km cells) - Delhi NCR Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.png_out), exist_ok=True)
        plt.savefig(args.png_out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[grid_map] Saved PNG map to {args.png_out}")
        
        print(f"[grid_map] Grid map creation complete!")
        
    except Exception as e:
        print(f"[grid_map][ERROR] Error creating grid map: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
