# Microbial pH predictions

## Project overview

This project aims to develop machine learning models for predicting pH from microbial abundances and 16S rRNA gene sequences. The approach incorporates both traditional machine learning methods (e.g., Random Forests and KNN regression) and deep learning techniques, with particular emphasis on language models for predictions from gene sequences.

## Methods

- **Traditional ML**: Random Forests, KNN regression, and other classical approaches
- **Deep Learning**: Language models and other neural network architectures, especially for gene sequence-based predictions

## Data processing

The `process_rdp_data.py` script is used to aggregate taxonomic counts from RDP classifier output files. The script processes `.tab` files in the `data/RDP_classifier` directory and generates aggregated count tables saved as CSV files in the `data` directory. The output files contain taxonomic counts at various levels (phylum, class, order, family, genus) with samples as rows and taxonomic groups as columns.

**Note**: The source RDP classifier files are not stored in this repository but can be downloaded from the [JMDplots repository](https://github.com/jedick/JMDplots/tree/main/inst/extdata/orp16S/RDP-GTDB).

## SRA data download

The `download_sra_data.py` script downloads DNA sequence data from the NCBI Sequence Read Archive (SRA) for samples listed in `sample_data.csv`. The script downloads only the first 0.25MB of each sample to manage storage requirements while preserving sequence data for analysis.

```bash
python download_sra_data.py
```

<details>
<summary>Details</summary>

The script processes samples from `sample_data.csv` that have sample_ids matching the `SRR*` or `ERR*` pattern (NCBI SRA identifiers). For each sample, it downloads the first 0.25MB from the NCBI SRA fasta endpoint, processes the incomplete gzip file, and saves the result to `data/fasta/{study_name}/{sample_id}.fasta.gz`.

- **Partial downloads**: Uses HTTP Range headers to download only the first 0.25MB, reducing storage requirements
- **Incomplete gzip handling**: Extracts truncated gzip files using `zcat`, which can handle incomplete archives
- **Sequence truncation protection**: Removes the last sequence from each file since it may be truncated due to the partial download
- **Duplicate handling**: Automatically skips samples that have already been downloaded or appear multiple times in the CSV
- **Pattern filtering**: Only processes sample_ids matching `SRR*` or `ERR*` patterns; other identifiers are skipped
- **Progress tracking**: Shows current study_name and sample_id during processing with a summary at completion
- **Cleanup on interruption**: Removes temporary files if the script is interrupted (Ctrl+C), preventing leftover files

**Output Structure:**
- Files are saved to `data/fasta/{study_name}/{sample_id}.fasta.gz`
- Each file contains complete DNA sequences (except the last one, which is removed to avoid truncation artifacts)

**Requirements:**
- `requests` library for HTTP downloads
- `zcat` command (part of gzip utilities) for extracting incomplete gzip files

</details>

## Hugging Face dataset creation

The `create_hf_dataset.py` script creates a Hugging Face dataset from DNA sequences stored in gzipped FASTA files and sample metadata. This dataset is designed for training language models on 16S rRNA gene sequences to predict pH values.

```bash
# Requires HF_TOKEN in .env file
python create_hf_dataset.py
```

<details>
<summary>Details</summary>

The script processes DNA sequences from FASTA files organized by study (`data/fasta/{study_name}/{sample_id}.fasta.gz`) and combines them with metadata from `sample_data.csv` to create a structured dataset suitable for machine learning workflows.

- **Automatic validation**: Ensures all FASTA files have corresponding entries in `sample_data.csv`
- **Incremental updates**: Skips samples that already exist in the dataset, allowing for incremental additions
- **Size management**: Limits total DNA length per sample to 1 MB to manage dataset size
- **Metadata integration**: Includes sample metadata (pH, environment, domain) alongside sequences

</details>

## pH prediction workflow

The `predict_ph.py` script implements a machine learning pipeline for predicting pH values from bacterial phylum abundances. The workflow includes data loading and preprocessing, train-test split, model training, and output.

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

<details>
<summary>Details</summary>

1. **Data Loading and Preprocessing**:
   - Filters sample data to include only the domain Bacteria
   - Removes samples with missing pH values
   - Joins sample metadata with phylum count data
   - Performs L1 normalization to convert counts to relative abundances

2. **Train-Test Split**:
   - Creates an 80-20 stratified split based on environment
   - Ensures balanced representation of environment types in both sets

3. **Model Training**:
   - Optional grid search with progress bar for hyperparameter tuning
   - Uses scikit-learn Pipeline for preprocessing and model fitting

4. **Output**:
   - Saves test set predictions to CSV files with study names and sample IDs
   - Generates summary files with model metrics (RMSE, R-squared, MAE) and grid search results

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

## Project history

_Notable changes to architecture, data, features, or results will be documented here._
