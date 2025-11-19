# Earth Observation Classification Project

Comprehensive, end-to-end workflow for classifying Sentinel‑2 RGB tiles over
Delhi NCR using ESA WorldCover 2021 labels. The repository ships fully working
Python scripts, documentation, visualization utilities, and an automated
reporting pipeline so the assignment can be reproduced with the provided data.

---

## Repository Layout

```
Data/
├── worldcover_bbox_delhi_ncr_2021.tif
└── rgb/                           # 128×128 Sentinel-2 PNG tiles
Dataset/
├── metadata.csv                   # filename, lat, lon
├── labels.csv                     # assigned_code, label, dominance, keep
├── train_metadata.csv
└── test_metadata.csv
models/
└── resnet_model.py                # ResNet18 wrapper + save/load helpers
scripts/
├── create_metadata.py
├── label_from_landcover.py
├── split_dataset.py
├── train_model.py
├── visualize_results.py
├── grid_map.py
└── report_builder.py
Visualizations/
└── *.png / *.html                 # class distribution, confusion matrix, grids, map
run_all.sh / run_all.bat           # sequential pipeline runner
report.pdf                         # generated PDF report
requirements.txt                   # exact dependency list
```

---

## Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> Dependencies: numpy, pandas, geopandas, rasterio, shapely, matplotlib,
> seaborn, scikit-learn, torch, torchvision, torchmetrics, tqdm, pillow,
> folium, geemap.

---

## End-to-End Pipeline

You can execute each script manually (recommended for experimentation) or run
the entire flow via `run_all.sh`/`run_all.bat`. Every script emits progress
logs, sample rows, and helpful warnings.

1. **Create metadata**
   ```powershell
   python scripts/create_metadata.py ^
       --rgb-dir Data/rgb ^
       --output Dataset/metadata.csv ^
       [--metadata-csv path\to\custom_metadata.csv]
   ```
   Parses filenames (e.g., `img_28.7041_77.1025.png`) to recover lat/lon. If
   filenames lack coordinates, provide your own CSV with `filename,lat,lon`.

2. **Label from ESA WorldCover**
   ```powershell
    python scripts/label_from_landcover.py ^
        --metadata Dataset/metadata.csv ^
        --tif Data/worldcover_bbox_delhi_ncr_2021.tif ^
        --patch-size 128 ^
        --dominance-threshold 0.0 ^
        --threads 4 ^
        --output Dataset/labels.csv
   ```
   Reprojects coordinates, samples 128×128 patches (boundless, nodata-safe),
   computes modal ESA code, dominance ratio, keep flag, and drops nodata tiles.
   Warns when many patches fall below dominance 0.4.

3. **Split dataset**
   ```powershell
   python scripts/split_dataset.py ^
       --metadata Dataset/metadata.csv ^
       --labels Dataset/labels.csv ^
       --test-size 0.4 ^
       --dominance-threshold 0.0 ^
       --random-state 42 ^
       [--no-stratify]
   ```
   Merges metadata + labels, filters by dominance if requested, and performs a
   stratified 60/40 split (fallback to random split when class counts are low).

4. **Train ResNet18**
   ```powershell
   python scripts/train_model.py ^
       --model cnn ^
       --epochs 10 ^
       --batch-size 32 ^
       --img-dir Data/rgb ^
       --lr 1e-3 ^
       --class-weight ^
       --use-cuda
   ```
   - Uses `models/resnet_model.ResNetLandCover` (ResNet18 pretrained).
   - Applies heavy data augmentation for training and standard transforms for
     eval.
   - Supports class weighting via `--class-weight`.
   - Logs epoch loss + macro F1 (custom) vs torchmetrics F1Score.
   - Saves best checkpoint to `models/resnet18_epochXX_f1YYY.pth`.
   - Writes `runs/<timestamp>/metrics.json`, `runs/<timestamp>/predictions.csv`,
     and also copies `runs/predictions.csv` for downstream scripts.
   - Generates `Visualizations/class_distribution.png`,
     `Visualizations/confusion_matrix.png`, and `Visualizations/loss_curve.png`.
   - Updates `report.pdf` with the latest plots, metrics, and sample
     predictions.

5. **Visualize qualitative results**
   ```powershell
   python scripts/visualize_results.py ^
       --predictions runs/predictions.csv ^
       --img-dir Data/rgb ^
       --n-correct 5 ^
       --n-incorrect 5 ^
       --out-dir Visualizations
   ```
   Creates `correct_predictions.png` and `incorrect_predictions.png`.

6. **Grid QA map**
   ```powershell
   python scripts/grid_map.py ^
       --metadata Dataset/metadata.csv ^
       --html-out Visualizations/grid_map.html ^
       --png-out Visualizations/grid_map.png
   ```
   Reprojects metadata to EPSG:32644, builds a 60×60 km grid, and exports both a
   folium HTML map and static PNG.

7. **Report regeneration**
   - Placeholder: `python scripts/report_builder.py --mode placeholder`
   - Final update happens automatically after training, but you can rerun:
     ```powershell
     python scripts/report_builder.py --mode final ^
         --metrics runs\<timestamp>\metrics.json ^
         --predictions runs\<timestamp>\predictions.csv ^
         --class-dist Visualizations/class_distribution.png ^
         --confusion Visualizations/confusion_matrix.png ^
         --report report.pdf
     ```

### One-click pipeline

```
./run_all.sh        # macOS/Linux
run_all.bat         # Windows
```

Each step exits immediately on failure and prints `[run_all][ERROR] ...`.

---

## Outputs & Artifacts

- `Dataset/*.csv` – metadata, labels (with dominance & keep), train/test splits.
- `models/resnet18_epochXX_f1YYY.pth` – best checkpoint (with metadata).
- `runs/<timestamp>/metrics.json` – training curves + best metrics.
- `runs/<timestamp>/predictions.csv` + `runs/predictions.csv` – per-image
  outcomes with probabilities.
- `Visualizations/*.png/.html` – plots, qualitative grids, grid map.
- `report.pdf` – automatically regenerated PDF summarizing methodology,
  datasets, results, discussion, and reproduction appendix.

---

## Troubleshooting

- **No coordinates detected**: ensure filenames embed lat/lon or provide a CSV
  via `--metadata-csv`.
- **Label dominance too low**: review `Dataset/labels.csv` (columns `dominance`,
  `keep`). Increase `--dominance-threshold` or adjust bounding boxes.
- **Stratification failure**: rerun `split_dataset.py --no-stratify` or collect
  more samples per class.
- **CUDA unavailable**: omit `--use-cuda` or install GPU-enabled PyTorch.
- **Report not updating**: run `python scripts/report_builder.py --mode final`
  manually after confirming visuals/metrics exist.

---

## Regenerating Everything

1. Clean `Dataset/`, `models/`, `runs/`, `Visualizations/` if needed.
2. Run `run_all.bat` (Windows) or `run_all.sh` (macOS/Linux).
3. Inspect `pipeline_run.log` (created when you redirect output) for summary
   counts, final macro F1, checkpoint location, and report path.

Happy mapping!


