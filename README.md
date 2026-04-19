# NeuroMetrics: 90-Class Animal Classification & Benchmarking 🐾

This repository houses a comprehensive, end-to-end Computer Vision pipeline designed natively to classify over 5,400 raw images spanning 90 distinct animal categories. Beyond typical notebook foundations, this project natively evaluates structural Transfer Learning backbones against each other and routes test outputs directly into a custom-built Web Dashboard.

## 🚀 Key Features

*   **Deep Transfer Learning**: Utilizes pre-trained `EfficientNetB3` architectures paired with custom sequential classification tops (`BatchNormalization` + `Dropout`), mapping aggressive Data Augmentation filters (`RandomRotation`, `RandomFlip`, `RandomZoom`).
*   **Fully Automated Benchmarking**: Natively evaluates competitive network baseline structures (`MobileNetV2` and `ResNet50`) iteratively scoring metric gaps on uniform constraints iteratively!
*   **Dynamic Visual UI Toolkit**: Features a standalone custom Single Page Application UI ("Dashboard") utilizing vanilla Glassmorphism CSS and `Chart.js` tracking automated epoch evaluations cleanly across classes!
*   **Intelligent Evaluation Pipelines**: Auto-generates high-level statistical evaluation grids mapping raw image inferences, exact confidence scores, and overall Matplotlib `Seaborn` Confusion Heatmaps seamlessly!

---

## 🛠️ Installation Requirements

Ensure your Python ecosystem is mapping the required computational packages structurally:

```bash
# Core Machine Learning & Visualization Libraries
pip install tensorflow pandas scikit-learn matplotlib 

# Notebooks & OS Ingestion
pip install kagglehub notebook jupyter
```

---

## 💻 How to Run (Step-by-Step)

Due to modern Keras framework API behavioral updates (v3.0+), some structural execution constraints heavily recommend setting **Legacy flags natively** in your environments when loading specific backward-compatible optimizer configurations.

### 1. View & Execute the Notebook
To explore data pipelines natively and visually observe the initial 15-Epoch EfficientNet configuration mapping:
```powershell
$env:TF_USE_LEGACY_KERAS="1"
jupyter notebook implementation.ipynb
```

### 2. Run the Model Benchmarking Script
To train the additional Neural Networks (`MobileNetV2` and `ResNet50`) sequentially over standard parameters and organically build out dataset comparison JSON mappings:
```powershell
$env:TF_USE_LEGACY_KERAS="1"
python train_comparison.py
```
*(This extracts evaluation histories & macros directly into `dashboard/metrics.json`)*

### 3. Launch the Graphical Dashboard
Modern browser securities strictly prevent Javascript files from rendering local `.json` documents offline natively (`CORS`). To vividly explore the metrics, boot an automatic Python server locally:
```powershell
python -m http.server 8000 --directory dashboard
```
👉 **Open your browser natively to:** [http://localhost:8000](http://localhost:8000)

### 4. Test Inferencing Visualizations
Need raw graphical proof of network functionality? Execute the standalone visualization pipeline mapping the raw tests sequentially against accurate bounding confidence outputs natively generated in `prediction_samples.png`:
```powershell
$env:TF_USE_LEGACY_KERAS="1"
python visualize_predictions.py
```

---

## 📂 Project Architecture

*   `implementation.ipynb`: The core analytical Jupyter layout executing extraction, augmentations dynamically, and heavy visual matrix mapping mathematically highlighting specific wild-life class cross-overs!
*   `train_comparison.py`: A native Python benchmarker dynamically exporting validation gaps between models.
*   `visualize_preprocessing.py`: Automatically produces grids proving explicit Augmentation variances internally.
*   `visualize_predictions.py`: Randomly allocates dynamic ground-truth inference examples structurally natively outputting colorful confident accuracy representations!
*   `dashboard/`: The root directory mapped storing our native glass-styled Single-Page UI and Chart outputs!
