# Automated Valuation Model (AVM) — End-to-End Execution Guide

This repository contains the complete end-to-end processing pipeline, model training, conformal prediction framework, and diagnostic evaluation scripts for the **Real Estate Automated Valuation Model (AVM)** focused on Zimbabwean suburban property markets.

---

## Technical Architecture & Pipeline Overview

```
                      END-TO-END PIPELINE ARCHITECTURE
  
  +------------------+     +-------------------+     +---------------------+
  | 1. Data Ingestion| --> | 2. Feature Eng.   | --> | 3. Model Training   |
  |  • Multi-modal   |     |  • Log Transforms |     |  • Log-OLS Baseline |
  |  • Text / Vision |     |  • TF-IDF & CNN   |     |  • RF / XGBoost     |
  +------------------+     +-------------------+     +---------------------+
                                                                |
                                                                v
  +------------------+     +-------------------+     +---------------------+
  | 6. Audit & Report| <-- | 5. Extrapolation  | <-- | 4. Inference &      |
  |  • Actuarial Memo|     |  • k-NN Distance  |     |    Conformal Bands  |
  |  • Summary Logs  |     |  • Shell Cost Adj |     |  • 90% Intervals    |
  +------------------+     +-------------------+     +---------------------+

```

---

## Prerequisites & Installation

Ensure you have Python 3.9 or higher installed.

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/MSACS512_ASSIGNMENT1.git
cd MSACS512_ASSIGNMENT1

```

### 2. Create and Activate a Virtual Environment

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

#### `requirements.txt` Dependencies

```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scipy>=1.7.0
torch>=1.10.0
torchvision>=0.11.0
pillow>=8.4.0
matplotlib>=3.4.0
seaborn>=0.11.0

```

---

## Repository Structure

```text
.
├── data/
│   ├── raw_listings.csv          # Multi-modal raw dataset (n=245)
│   └── images/                   # Property listing photo directory
├── src/
│   ├── data_preprocessing.py     # Clean, impute, and extract TF-IDF/CNN features
│   ├── train_models.py           # Fit Log-OLS, ElasticNet, RF, XGBoost models
│   ├── conformal_inference.py    # Compute split conformal interval bounds
│   ├── extrapolation_diag.py     # Calculate k-NN distance & cost-to-complete
│   └── evaluate_interpret.py     # Generate error metrics & permutation SHAP
├── run_pipeline.py               # Master orchestration script
├── outputs/                      # Generated evaluation metrics & log outputs
└── README.md                     # End-to-end execution guide

```

---

## Step-by-Step Pipeline Execution

You can execute the pipeline step-by-step or run the master script.

### Option A: Running Step-by-Step Modules

#### Step 1: Preprocess Data & Extract Multimodal Features

Cleans input listings, applies log-transformations to target prices and stand sizes, handles median suburb imputation, extracts text flags via TF-IDF, and extracts image embeddings using a pre-trained ResNet-50 model.

```bash
python src/data_preprocessing.py --input data/raw_listings.csv --output_dir data/processed/

```

#### Step 2: Model Training & Hyperparameter Tuning

Trains all target model architectures (Log-OLS, ElasticNet, Random Forest, XGBoost) using stratified splits on `strat_suburb` to prevent spatial leakage.

```bash
python src/train_models.py --data_dir data/processed/ --model_dir models/

```

#### Step 3: Conformal Prediction Calibration & Case Evaluation

Calibrates 90% Split Conformal Prediction intervals on validation bounds and evaluates the **Madokero vs. Mabvazuva** unfinished shell case study.

```bash
python src/conformal_inference.py --model_path models/random_forest.joblib --case_study

```

#### Step 4: Run Extrapolation Diagnostics & Actuarial Adjustments

Computes $k$-NN feature space Euclidean distance ($d_5$) for structural shell evaluation and applies the Hybrid Cost-to-Complete deduction pipeline.

```bash
python src/extrapolation_diag.py --k_neighbors 5 --threshold 2.50

```

---

### Option B: Execution via Master Orchestration Script

To run the entire pipeline end-to-end from raw data ingestion to report metric output:

```bash
python run_pipeline.py --raw_data data/raw_listings.csv --output_dir outputs/

```

---

## Output Verification & Expected Logs

Upon successful completion of `run_pipeline.py`, your console output should display key validation steps:

```text
========================================================================================
AUTOMATED VALUATION MODEL (AVM) PIPELINE EXECUTION
========================================================================================
[INFO] Loading raw data from 'data/raw_listings.csv'... (n = 245)
[INFO] Preprocessing completed. Stratified splits generated (Train: 171, Val: 37, Test: 37).
[INFO] Models successfully trained: Log-OLS, ElasticNet, Random Forest, XGBoost.

----------------------------------------------------------------------------------------
MODEL EVALUATION SUMMARY (HOLDOUT TEST SET, n=37)
----------------------------------------------------------------------------------------
Model                | RMSE (USD)   | MAE (USD)   | MAPE (%) | R² Score | Coverage (90%)
----------------------------------------------------------------------------------------
GLM Benchmark        | $142,500.00  | $88,200.00  | 24.5%    | 0.712    | 91.9%
ElasticNet           | $138,100.00  | $84,100.00  | 22.8%    | 0.730    | 89.2%
XGBoost Regressor    | $118,900.00  | $68,500.00  | 17.8%    | 0.801    | 89.2%
Random Forest        | $112,350.00  | $64,200.00  | 16.2%    | 0.821    | 89.2%
----------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------
CASE STUDY EXTENDED DIAGNOSTICS
----------------------------------------------------------------------------------------
House A (Madokero):  Point = $72,450.00 | d_5 Distance = 3.84 std | [HIGH EXTRAPOLATION RISK]
House B (Mabvazuva): Point = $58,200.00 | d_5 Distance = 1.92 std | [IN-DISTRIBUTION]

Applying Cost-To-Complete Hybrid Framework on House A...
  Base Valuation:                          $72,450.00
  Total Structural Deductions:             -$20,000.00
  Actuarial Adjusted Net Collateral Value: $52,450.00

[SUCCESS] All outputs, evaluation plots, and diagnostic logs saved to 'outputs/'.

```

---

## Troubleshooting & Common Configuration Fixes

* **PyTorch CPU vs. GPU Acceleration:** If CUDA is available, image processing via ResNet-50 will run on GPU automatically. For CPU-only environments, force CPU execution using `--device cpu`.
* **Missing Value Errors on Suburbs:** Ensure sparse suburb classes ($n < 5$) are properly mapped to `'Other'` during preprocessing to prevent runtime crashes during one-hot encoding.
