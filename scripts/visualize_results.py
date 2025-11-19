#!/usr/bin/env python3
"""
Visualize correct and incorrect predictions.

Reads predictions.csv and creates multi-panel visualizations showing
sample correct and incorrect predictions with their labels and probabilities.
"""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Visualize prediction results')
    parser.add_argument('--predictions', type=str, default='runs/predictions.csv',
                       help='Predictions CSV (default: runs/predictions.csv)')
    parser.add_argument('--img-dir', type=str, default='Data/rgb',
                       help='Image directory (default: Data/rgb)')
    parser.add_argument('--n-correct', type=int, default=5,
                       help='Number of correct predictions to show (default: 5)')
    parser.add_argument('--n-incorrect', type=int, default=5,
                       help='Number of incorrect predictions to show (default: 5)')
    parser.add_argument('--out-dir', type=str, default='Visualizations',
                       help='Output directory (default: Visualizations)')
    
    args = parser.parse_args()
    
    # Read predictions
    print(f"[visualize_results] Reading predictions from {args.predictions}...")
    if not os.path.exists(args.predictions):
        print(f"[visualize_results][ERROR] Predictions file not found: {args.predictions}")
        exit(1)
    
    df = pd.read_csv(args.predictions)
    print(f"[visualize_results] Loaded {len(df)} predictions")
    
    # Add correct/incorrect flag
    df['correct'] = df['true_label'] == df['pred_label']
    
    # Get probabilities
    def get_max_prob(prob_str):
        try:
            probs = json.loads(prob_str)
            return max(probs) if probs else 0.0
        except:
            return 0.0
    
    df['max_prob'] = df['probabilities'].apply(get_max_prob)
    
    # Sample correct and incorrect
    correct_df = df[df['correct']].copy()
    incorrect_df = df[~df['correct']].copy()
    
    n_correct = min(args.n_correct, len(correct_df))
    n_incorrect = min(args.n_incorrect, len(incorrect_df))
    
    if n_correct > 0:
        correct_sample = correct_df.sample(n=n_correct, random_state=42) if len(correct_df) > n_correct else correct_df
    else:
        correct_sample = pd.DataFrame()
    
    if n_incorrect > 0:
        incorrect_sample = incorrect_df.sample(n=n_incorrect, random_state=42) if len(incorrect_df) > n_incorrect else incorrect_df
    else:
        incorrect_sample = pd.DataFrame()
    
    print(f"[visualize_results] Sampling {n_correct} correct and {n_incorrect} incorrect predictions")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Visualize correct predictions
    if len(correct_sample) > 0:
        n_cols = 3
        n_rows = (len(correct_sample) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, row in enumerate(correct_sample.itertuples()):
            img_path = os.path.join(args.img_dir, row.filename)
            try:
                img = Image.open(img_path).convert('RGB')
                axes[idx].imshow(img)
                axes[idx].set_title(f"True: {row.true_label}\nPred: {row.pred_label}\nProb: {row.max_prob:.3f}",
                                   fontsize=10)
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error loading\n{row.filename}", 
                              ha='center', va='center')
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(correct_sample), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Correct Predictions', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'correct_predictions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[visualize_results] Saved correct_predictions.png")
    
    # Visualize incorrect predictions
    if len(incorrect_sample) > 0:
        n_cols = 3
        n_rows = (len(incorrect_sample) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, row in enumerate(incorrect_sample.itertuples()):
            img_path = os.path.join(args.img_dir, row.filename)
            try:
                img = Image.open(img_path).convert('RGB')
                axes[idx].imshow(img)
                axes[idx].set_title(f"True: {row.true_label}\nPred: {row.pred_label}\nProb: {row.max_prob:.3f}",
                                   fontsize=10, color='red')
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error loading\n{row.filename}", 
                              ha='center', va='center')
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(incorrect_sample), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Incorrect Predictions', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'incorrect_predictions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[visualize_results] Saved incorrect_predictions.png")
    
    print(f"[visualize_results] Visualization complete!")


if __name__ == '__main__':
    main()
