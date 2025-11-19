#!/usr/bin/env python3
"""
Split dataset into train/test sets with optional stratification by land cover class.

Merges metadata and labels, filters by keep==True, and performs stratified split.
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/test sets'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='Dataset/metadata.csv',
        help='Metadata CSV (default: Dataset/metadata.csv)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='Dataset/labels.csv',
        help='Labels CSV (default: Dataset/labels.csv)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.4,
        help='Test set fraction (default: 0.4)'
    )
    parser.add_argument(
        '--dominance-threshold',
        type=float,
        default=0.0,
        help='Minimum dominance to keep (default: 0.0, uses keep column)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable stratification (use random split)'
    )
    
    args = parser.parse_args()
    
    # Read files
    print(f"[split_dataset] Reading metadata from {args.metadata}...")
    metadata = pd.read_csv(args.metadata)
    print(f"[split_dataset] Loaded {len(metadata)} metadata records")
    
    print(f"[split_dataset] Reading labels from {args.labels}...")
    labels = pd.read_csv(args.labels)
    print(f"[split_dataset] Loaded {len(labels)} label records")
    
    # Merge
    print(f"[split_dataset] Merging metadata and labels...")
    df = metadata.merge(labels, on=['filename', 'lat', 'lon'], how='inner')
    print(f"[split_dataset] Merged dataset: {len(df)} records")
    
    # Filter by keep==True
    if 'keep' in df.columns:
        initial_count = len(df)
        df = df[df['keep'] == True].copy()
        print(f"[split_dataset] Filtered to keep==True: {len(df)} records (dropped {initial_count - len(df)})")
    
    # Apply dominance threshold if specified
    if args.dominance_threshold > 0.0 and 'dominance' in df.columns:
        initial_count = len(df)
        df = df[df['dominance'] >= args.dominance_threshold].copy()
        print(f"[split_dataset] Applied dominance threshold {args.dominance_threshold}: {len(df)} records (dropped {initial_count - len(df)})")
    
    if len(df) == 0:
        print(f"[split_dataset][ERROR] No records remaining after filtering")
        exit(1)
    
    # Check if we can stratify
    stratify_col = None
    if not args.no_stratify and 'assigned_code' in df.columns:
        class_counts = df['assigned_code'].value_counts()
        min_class_count = class_counts.min()
        
        # Need at least 2 samples per class for stratification
        if min_class_count >= 2:
            stratify_col = df['assigned_code']
            print(f"[split_dataset] Stratifying by assigned_code (min class count: {min_class_count})")
        else:
            print(f"[split_dataset][WARNING] Cannot stratify: min class count is {min_class_count} (< 2). Using random split.")
            stratify_col = None
    else:
        print(f"[split_dataset] Using random split (--no-stratify or no assigned_code column)")
    
    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_col
    )
    
    print(f"[split_dataset] Split complete:")
    print(f"  Train: {len(train_df)} records ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Test: {len(test_df)} records ({100*len(test_df)/len(df):.1f}%)")
    
    # Print class distribution
    if 'assigned_code' in df.columns:
        print(f"\n[split_dataset] Train class distribution:")
        print(train_df['assigned_code'].value_counts().sort_index())
        print(f"\n[split_dataset] Test class distribution:")
        print(test_df['assigned_code'].value_counts().sort_index())
    
    # Save
    os.makedirs('Dataset', exist_ok=True)
    train_df.to_csv('Dataset/train_metadata.csv', index=False)
    test_df.to_csv('Dataset/test_metadata.csv', index=False)
    
    print(f"\n[split_dataset] Saved:")
    print(f"  Dataset/train_metadata.csv ({len(train_df)} records)")
    print(f"  Dataset/test_metadata.csv ({len(test_df)} records)")


if __name__ == '__main__':
    main()
