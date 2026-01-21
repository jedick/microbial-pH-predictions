# Microbial pH predictions

## Project Overview

This project aims to develop machine learning models for predicting pH from microbial abundances and 16S rRNA gene sequences. The approach incorporates both traditional machine learning methods (e.g., Random Forests and KNN regression) and deep learning techniques, with particular emphasis on language models for predictions from gene sequences.

## Methods

- **Traditional ML**: Random Forests, KNN regression, and other classical approaches
- **Deep Learning**: Language models and other neural network architectures, especially for gene sequence-based predictions

## Data Processing

The `process_rdp_data.py` script is used to aggregate taxonomic counts from RDP classifier output files. The script processes `.tab` files in the `data/RDP_classifier` directory and generates aggregated count tables saved as CSV files in the `data` directory. The output files contain taxonomic counts at various levels (phylum, class, order, family, genus) with samples as rows and taxonomic groups as columns.

**Note**: The source RDP classifier files are not stored in this repository but can be downloaded from the [JMDplots repository](https://github.com/jedick/JMDplots/tree/main/inst/extdata/orp16S/RDP-GTDB).

## pH Prediction Workflow

The `predict_ph.py` script implements a machine learning pipeline for predicting pH values from bacterial phylum abundances. The workflow includes:

1. **Data Loading and Preprocessing**:
   - Filters sample data to include only Bacteria lineage
   - Removes samples with missing pH values
   - Joins sample metadata with phylum count data
   - Performs L1 normalization to convert counts to relative abundances

2. **Train-Test Split**:
   - Creates an 80-20 stratified split based on environment type (`envirotype`)
   - Ensures balanced representation of environment types in both sets

3. **Model Training**:
   - Supports four regression models: Linear Regression, KNN, Random Forest, and HistGradientBoosting
   - Optional grid search with progress bar for hyperparameter tuning
   - Uses scikit-learn Pipeline for preprocessing and model fitting

4. **Output**:
   - Saves test set predictions to CSV files with study names and sample IDs
   - Generates summary files with model metrics (RMSE, R-squared, MAE) and grid search results

### Usage

<details>
<summary>Click to expand command-line usage examples</summary>

```bash
# Test grid search with KNN (default, 4 combinations)
python predict_ph.py --model knn --grid-search test

# Full grid search with Random Forest (comprehensive)
python predict_ph.py --model rf --grid-search full

# Skip grid search and use default parameters
python predict_ph.py --model hgb --grid-search none

# Custom CV folds and random seed
python predict_ph.py --model knn --grid-search test --cv-folds 10 --random-seed 123

# Specify custom data paths and output directory
python predict_ph.py --model rf \
    --sample-data data/sample_data.csv \
    --phylum-counts data/bacteria_phylum_counts.csv.xz \
    --output-dir results/rf_experiment
```

**Available Models:**
- `linear`: Linear Regression
- `knn`: K-Nearest Neighbors Regression
- `rf`: Random Forest Regression
- `hgb`: HistGradientBoosting Regression

**Grid Search Modes:**
- `test`: Small grids (2-4 combinations) for quick testing
- `full`: Comprehensive grids (hundreds of combinations) for thorough tuning
- `none`: Skip grid search and use default parameters

**Output Files:**
- `{model}_test_predictions.csv`: Test set predictions with study_name, sample_id, true_pH, predicted_pH, and residual
- `{model}_summary.txt`: Model summary with metrics and grid search results (if performed)

</details>

## Project History

_Notable changes to architecture, data, features, or results will be documented here._
