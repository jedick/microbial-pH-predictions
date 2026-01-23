# Microbial pH predictions

## Project overview

This project aims to develop machine learning models for predicting pH from microbial abundances and 16S rRNA gene sequences. The approach incorporates both traditional machine learning methods (e.g., Random Forests and KNN regression) and deep learning techniques, with particular emphasis on language models for predictions from gene sequences.

## Methods

- **Traditional ML**: Random Forests, KNN regression, and other classical approaches
- **Deep Learning**: Language models and other neural network architectures, especially for gene sequence-based predictions

## SRA data download

The `download_sra_data.py` script downloads DNA sequence data from the NCBI Sequence Read Archive (SRA) for samples listed in `sample_data.csv`. The script downloads only the first 0.25MB of each sample to manage storage requirements while preserving sequence data for analysis.

```bash
python download_sra_data.py
```

<details>
<summary>Details</summary>

The script processes samples from `sample_data.csv` that have sample_ids matching the `SRR*`, `ERR*`, or `DRR*` pattern (NCBI SRA identifiers). For each sample, it downloads the first 0.25MB from the NCBI SRA fasta endpoint, processes the incomplete gzip file, and saves the result to `data/fasta/{study_name}/{sample_id}.fasta.gz`.

- **Partial downloads**: Uses HTTP Range headers to download only the first 0.25MB, reducing storage requirements
- **Incomplete gzip handling**: Extracts truncated gzip files using `zcat`, which can handle incomplete archives
- **Sequence truncation protection**: Removes the last sequence from each file since it may be truncated due to the partial download
- **Duplicate handling**: Automatically skips samples that have already been downloaded or appear multiple times in the CSV
- **Pattern filtering**: Only processes sample_ids matching `SRR*`, `ERR*`, or `DRR*` patterns; other identifiers are skipped
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

The `create_hf_dataset.py` script creates a Hugging Face dataset from DNA sequences stored in gzipped FASTA files and sample metadata. This dataset serves as the single source of truth for sample data across all downstream tasks, ensuring consistent train-test splits between traditional ML and deep learning workflows.

**Important**: After this step, `sample_data.csv` is no longer required for downstream tasks. The HuggingFace dataset contains all necessary sample metadata (pH, environment, domain, study_name, sample_id) and is used by both the traditional ML workflow (`predict_ph.py`) and the deep learning workflow (`train_hyenadna_ph.py`) to maintain consistent data splits.

```bash
# Requires HF_TOKEN in .env file
python create_hf_dataset.py
```

<details>
<summary>Details</summary>

The script processes DNA sequences from FASTA files organized by study (`data/fasta/{study_name}/{sample_id}.fasta.gz`) and combines them with metadata from `sample_data.csv` to create a structured dataset suitable for machine learning workflows.

- **Automatic validation**: Ensures all FASTA files have corresponding entries in `sample_data.csv`
- **Incremental updates**: Skips samples that already exist in the dataset, allowing for incremental additions
- **Size management**: Limits total DNA length per sample to 0.25 MB to manage dataset size
- **Metadata integration**: Includes sample metadata (pH, environment, domain) alongside sequences
- **Consistent splits**: Used by both workflows to ensure identical train-test splits (same random seed, same test set)

</details>

## RDP phylum counts (features)

The `process_rdp_data.py` script is used to aggregate taxonomic counts from RDP classifier output files. The script processes `.tab` files in the `data/RDP_classifier` directory and generates aggregated count tables saved as CSV files in the `data` directory. The output files contain taxonomic counts at various levels (phylum, class, order, family, genus) with samples as rows and taxonomic groups as columns.

**Note**: The source RDP classifier files are not stored in this repository but can be downloaded from the [JMDplots repository](https://github.com/jedick/JMDplots/tree/main/inst/extdata/orp16S/RDP-GTDB).

## pH prediction from phylum counts

The `predict_ph.py` script implements a machine learning pipeline for predicting pH values from bacterial phylum abundances. The workflow loads sample metadata from the HuggingFace dataset, merges it with phylum count features, and performs an 80:20 train-test split (same test set as the deep learning workflow).

```bash
# Test grid search with KNN (default, 4 combinations)
python predict_ph.py --model knn --grid-search test

# Full grid search with Random Forest (comprehensive)
python predict_ph.py --model rf --grid-search full

# Skip grid search and use default parameters
python predict_ph.py --model hgb --grid-search none

# Custom CV folds and random seed
python predict_ph.py --model knn --grid-search test --cv-folds 10 --random-seed 123

# Specify custom dataset and output directory
python predict_ph.py --model rf \
    --dataset-repo jedick/microbial-DNA-pH \
    --phylum-counts data/bacteria_phylum_counts.csv.xz \
    --output-dir results/rf_experiment
```

<details>
<summary>Details</summary>

1. **Data Loading and Preprocessing**:
   - Loads sample metadata from HuggingFace dataset (filters null pH values)
   - Merges with phylum count data from CSV file
   - Performs L1 normalization to convert counts to relative abundances

2. **Train-Test Split**:
   - Creates an 80:20 stratified split based on environment
   - Uses the same random seed (42) and test set as the deep learning workflow
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

## pH prediction from HyenaDNA sequence model

The `train_hyenadna_ph.py` script trains a HyenaDNA model with a regression head to predict continuous pH values from DNA sequences. The script loads data from the HuggingFace dataset, uses pretrained HyenaDNA weights, concatenates multiple sequences per sample with [SEP] tokens, and performs a 75:5:20 train-val-test split (same test set as the traditional ML workflow).

```bash
# Basic training with defaults (tiny model, downloads from HuggingFace)
python train_hyenadna_ph.py --download-model

# Custom configuration with larger model
python train_hyenadna_ph.py \
    --model-name hyenadna-small-32k-seqlen \
    --head-architecture mlp3 \
    --pooling-mode last \
    --batch-size 2 \
    --learning-rate 5e-5 \
    --num-epochs 20 \
    --freeze-backbone \
    --download-model

# Train with frozen backbone (only regression head)
python train_hyenadna_ph.py --freeze-backbone --download-model
```

<details>
<summary>Details</summary>

The script implements a separate `RegressionHead` class that maintains separation from the HyenaDNA codebase. It loads DNA sequences from the HuggingFace dataset (`jedick/microbial-DNA-pH`), concatenates multiple sequences per sample with [SEP] tokens up to the model's maximum length, and fine-tunes pretrained HyenaDNA weights for pH prediction.

- **Model options**: Supports all HyenaDNA model sizes (tiny-1k to large-1m-seqlen), default is tiny-1k-seqlen for testing
- **Regression head architectures**: Linear, 2-layer MLP (default), 3-layer MLP, or 2-layer MLP with LayerNorm
- **Pooling modes**: Mean pooling (default), last token, first token, or sum pooling
- **Sequence handling**: Concatenates multiple DNA sequences per sample with [SEP] tokens, truncates to max_length if needed
- **Training options**: Can freeze backbone (only train head) or fine-tune entire model, configurable loss functions (MSE, MAE, SmoothL1, Huber)
- **Evaluation**: Computes MSE, RMSE, MAE, and R² metrics on train/val/test splits
- **Data split**: Creates a 75:5:20 train-val-test split using the same random seed (42) and test set as the traditional ML workflow

**Output Files:**
- `best_model.pt`: Best model checkpoint based on validation loss
- `training_log.json`: Training history with metrics per epoch
- `test_predictions.csv`: Test set predictions with true and predicted pH values
- `config.json`: Training configuration used

**Key Hyperparameters:**
- Default batch size: 4 (adjust based on GPU memory and sequence length)
- Default learning rate: 1e-4 (typical for fine-tuning)
- Default loss: SmoothL1Loss (good balance between MSE and MAE)
- Train/val/test split: 75:5:20 (same test set as traditional ML workflow)

</details>
