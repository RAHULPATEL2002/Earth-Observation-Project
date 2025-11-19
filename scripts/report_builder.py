#!/usr/bin/env python3
"""
Generate PDF report with methodology, results, and visualizations.

Creates a comprehensive PDF report including title page, abstract, methods,
datasets, results with visualizations, and metrics tables.
"""

import argparse
import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Generate PDF report')
    parser.add_argument('--mode', type=str, default='final', choices=['placeholder', 'final'],
                       help='Report mode (default: final)')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Metrics JSON file (default: auto-detect from runs/)')
    parser.add_argument('--predictions', type=str, default='runs/predictions.csv',
                       help='Predictions CSV (default: runs/predictions.csv)')
    parser.add_argument('--class-dist', type=str, default='Visualizations/class_distribution.png',
                       help='Class distribution image (default: Visualizations/class_distribution.png)')
    parser.add_argument('--confusion', type=str, default='Visualizations/confusion_matrix.png',
                       help='Confusion matrix image (default: Visualizations/confusion_matrix.png)')
    parser.add_argument('--loss-acc', type=str, default='Visualizations/loss_accuracy.png',
                       help='Loss/accuracy plot (default: Visualizations/loss_accuracy.png)')
    parser.add_argument('--pr-boxplot', type=str, default='Visualizations/precision_recall_boxplot.png',
                       help='Precision/recall boxplot (default: Visualizations/precision_recall_boxplot.png)')
    parser.add_argument('--correct-preds', type=str, default='Visualizations/correct_predictions.png',
                       help='Correct predictions image (default: Visualizations/correct_predictions.png)')
    parser.add_argument('--incorrect-preds', type=str, default='Visualizations/incorrect_predictions.png',
                       help='Incorrect predictions image (default: Visualizations/incorrect_predictions.png)')
    parser.add_argument('--report', type=str, default='report.pdf',
                       help='Output PDF path (default: report.pdf)')
    
    args = parser.parse_args()
    
    print(f"[report_builder] Generating PDF report: {args.report}")
    
    # Auto-detect metrics file if not provided
    if args.metrics is None:
        # Find latest metrics.json in runs/
        runs_dir = 'runs'
        if os.path.exists(runs_dir):
            subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if subdirs:
                latest = sorted(subdirs)[-1]
                metrics_path = os.path.join(runs_dir, latest, 'metrics.json')
                if os.path.exists(metrics_path):
                    args.metrics = metrics_path
                    print(f"[report_builder] Auto-detected metrics: {args.metrics}")
    
    # Load metrics if available
    metrics = None
    if args.metrics and os.path.exists(args.metrics):
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
        print(f"[report_builder] Loaded metrics from {args.metrics}")
    
    # Create PDF
    doc = SimpleDocTemplate(args.report, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Section style
    section_style = ParagraphStyle(
        'CustomSection',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    # Body style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Earth Observation Land Cover Classification", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Using Sentinel-2 RGB Imagery and ESA WorldCover Labels", 
                          ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=16, 
                                        alignment=TA_CENTER, textColor=colors.HexColor('#7f8c8d'))))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                          ParagraphStyle('Date', parent=styles['Normal'], fontSize=10,
                                        alignment=TA_CENTER, textColor=colors.HexColor('#95a5a6'))))
    story.append(PageBreak())
    
    # Abstract
    story.append(Paragraph("Abstract", section_style))
    abstract_text = """
    This report presents a comprehensive pipeline for land cover classification using Sentinel-2 RGB imagery 
    and ESA WorldCover 2021 labels. The methodology involves extracting spatial metadata from image filenames, 
    labeling images using raster-based land cover data, and training a ResNet18 convolutional neural network 
    for multi-class classification. The pipeline processes 128×128 pixel image patches over the Delhi NCR region, 
    extracts dominant land cover classes from WorldCover data, and trains a deep learning model with data 
    augmentation and class weighting. Results are evaluated using macro F1 score, per-class precision and recall, 
    and confusion matrix analysis.
    """
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 0.3*inch))

        # Methods
    story.append(Paragraph("Methods", section_style))
    methods_text = """
    <b>1. Data Preparation:</b> RGB image patches (128×128 pixels) are scanned from the Data/rgb/ directory. 
    Latitude and longitude coordinates are extracted from filenames using pattern matching. If coordinate 
    extraction fails, the pipeline requires a manually provided metadata CSV file.
    <br/><br/>
    <b>2. Label Extraction:</b> For each image location, a 128×128 pixel patch is extracted from the ESA 
    WorldCover 2021 raster. Coordinates are reprojected to match the raster CRS when necessary. The modal 
    land cover code is computed from valid pixels (excluding nodata), and a dominance ratio is calculated 
    as the fraction of pixels matching the modal class. Images with dominance below a threshold can be 
    filtered out.
    <br/><br/>
    <b>3. Dataset Splitting:</b> The dataset is split into 60% training and 40% test sets with stratified 
    sampling by land cover class. If stratification is not possible due to insufficient samples per class, 
    a random split is performed with a warning.
    <br/><br/>
    <b>4. Model Training:</b> A ResNet18 model with pretrained ImageNet weights is fine-tuned for land 
    cover classification. The final fully connected layer is replaced with a custom head (512 → ReLU → 
    Dropout → N classes). Training uses data augmentation (random flips, rotations, color jitter) and 
    ImageNet normalization. Class weights can be applied to handle class imbalance. The model is trained 
    with Adam optimizer and evaluated using macro F1 score (both custom implementation and torchmetrics).
    """
    story.append(Paragraph(methods_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Datasets & Preprocessing
    story.append(Paragraph("Datasets & Preprocessing", section_style))
    datasets_text = """
    <b>Input Data:</b>
    <br/>• Sentinel-2 RGB image patches: 128×128 pixel PNG files in Data/rgb/
    <br/>• ESA WorldCover 2021 raster: Data/worldcover_bbox_delhi_ncr_2021.tif
    <br/><br/>
    <b>Land Cover Classes:</b> The ESA WorldCover classification includes 11 classes: Tree cover (10), 
    Shrubland (20), Grassland (30), Cropland (40), Built-up (50), Bare/sparse vegetation (60), 
    Snow/ice (70), Water (80), Herbaceous wetland (90), Mangroves (95), and Moss/lichen (100).
    <br/><br/>
    <b>Preprocessing Steps:</b>
    <br/>• Coordinate extraction from filenames or manual metadata CSV
    <br/>• Spatial reprojection to match raster CRS
    <br/>• Patch extraction with boundless reading and nodata handling
    <br/>• Modal class computation and dominance filtering
    <br/>• Stratified train/test splitting
    """
    story.append(Paragraph(datasets_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Results
    story.append(Paragraph("Results", section_style))
    
    # Add metrics table if available
    if metrics:
        results_text = f"""
        <b>Training Configuration:</b>
        <br/>• Epochs: {metrics.get('epochs', 'N/A')}
        <br/>• Best epoch: {metrics.get('best_epoch', 'N/A')}
        <br/>• Number of classes: {metrics.get('num_classes', 'N/A')}
        <br/><br/>
        <b>Performance Metrics:</b>
        <br/>• Final Macro F1 (custom): {metrics.get('final_macro_f1', 0):.4f}
        <br/>• Final Macro F1 (torchmetrics): {metrics.get('final_torchmetrics_f1', 0):.4f}
        <br/>• Best Macro F1: {metrics.get('best_macro_f1', 0):.4f}
        """
        story.append(Paragraph(results_text, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Per-class metrics table
        if 'per_class_f1' in metrics and 'class_names' in metrics:
            class_data = []
            class_data.append(['Class', 'Precision', 'Recall', 'F1 Score'])
            for i, (name, prec, rec, f1) in enumerate(zip(
                metrics['class_names'],
                metrics['per_class_precision'],
                metrics['per_class_recall'],
                metrics['per_class_f1']
            )):
                class_data.append([
                    name,
                    f"{prec:.3f}",
                    f"{rec:.3f}",
                    f"{f1:.3f}"
                ])
            
            class_table = Table(class_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(class_table)
            story.append(Spacer(1, 0.3*inch))
    else:
        story.append(Paragraph("Metrics not available. Run training first.", body_style))
        story.append(Spacer(1, 0.3*inch))
    
    # Add visualizations
    viz_files = [
        ('Class Distribution', args.class_dist),
        ('Confusion Matrix', args.confusion),
        ('Loss and F1 Curves', args.loss_acc),
        ('Precision/Recall Boxplot', args.pr_boxplot),
    ]
    
    for viz_title, viz_path in viz_files:
        if os.path.exists(viz_path):
            story.append(Paragraph(f"<b>{viz_title}</b>", body_style))
            story.append(Spacer(1, 0.1*inch))
            img = Image(viz_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    
    # Sample predictions
    if os.path.exists(args.correct_preds):
        story.append(Paragraph("<b>Sample Correct Predictions</b>", body_style))
        story.append(Spacer(1, 0.1*inch))
        img = Image(args.correct_preds, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    if os.path.exists(args.incorrect_preds):
        story.append(Paragraph("<b>Sample Incorrect Predictions</b>", body_style))
        story.append(Spacer(1, 0.1*inch))
        img = Image(args.incorrect_preds, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    print(f"[report_builder] PDF report saved to {args.report}")


if __name__ == '__main__':
    main()
