"""
Predict pH from bacterial abundances using various regression models.

This script loads and preprocesses bacterial abundance data, performs
train-test splitting, and fits regression models to predict pH values.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split
from datasets import load_dataset

from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
import warnings

# Try to import tqdm for progress bar, fallback to None if not available
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print(
        "Note: tqdm not available. Install with 'pip install tqdm' for progress bars."
    )

warnings.filterwarnings("ignore")


def load_and_preprocess_data(dataset_repo, phylum_counts_path, random_seed=42):
    """
    Load and preprocess data for pH prediction.

    Parameters
    ----------
    dataset_repo : str
        HuggingFace dataset repository (e.g., "jedick/microbial-DNA-pH")
    phylum_counts_path : str
        Path to bacteria_phylum_counts.csv.xz (compressed)
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, test_metadata)
        where test_metadata is a DataFrame with study_name and sample_id for test set
    """
    # Load sample data from HF dataset
    print(f"Loading dataset from {dataset_repo}...")
    dataset = load_dataset(dataset_repo, split="train")

    # Convert to DataFrame and filter out null pH values
    print("Filtering data...")
    sample_data_list = []
    for item in dataset:
        if item["pH"] is not None:  # Filter out null pH values
            sample_data_list.append(
                {
                    "study_name": item["study_name"],
                    "sample_id": item["sample_id"],
                    "pH": item["pH"],
                    "environment": item["environment"],
                }
            )

    sample_data = pd.DataFrame(sample_data_list)
    print(f"Sample data shape after filtering (null pH removed): {sample_data.shape}")

    # Create stratified 80:20 train-test split on HF dataset (before merging with phylum counts)
    # This ensures the same test split as train_hyenadna_ph.py
    print("Creating stratified train-test split (80:20)...")
    sample_ids = sample_data["sample_id"].values
    environment = sample_data["environment"].values

    ids_train, ids_test = train_test_split(
        sample_ids,
        test_size=0.2,
        random_state=random_seed,
        stratify=environment,
    )

    # Convert to sets for easier filtering
    ids_train_set = set(ids_train)
    ids_test_set = set(ids_test)

    # Load phylum counts (pandas can read .xz compressed files directly)
    print("Loading phylum counts from compressed file...")
    phylum_counts = pd.read_csv(phylum_counts_path, compression="xz")
    print(f"Phylum counts shape: {phylum_counts.shape}")

    # Join on sample_id only (preserves all samples after filtering)
    print("Joining datasets...")
    merged_data = sample_data.merge(
        phylum_counts,
        left_on=["sample_id"],
        right_on=["sample_id"],
        how="inner",
        suffixes=("", "_y"),
    )

    # Keep study_name from sample_data (left side of merge)
    if "study_name_y" in merged_data.columns:
        merged_data = merged_data.drop(columns=["study_name_y"])
    if (
        "study_name" not in merged_data.columns
        and "study_name_x" in merged_data.columns
    ):
        merged_data = merged_data.rename(columns={"study_name_x": "study_name"})

    print(f"Merged data shape: {merged_data.shape}")

    # Identify phylum count columns (all columns starting with 'phylum__')
    phylum_cols = [col for col in merged_data.columns if col.startswith("phylum__")]
    print(f"Number of phylum features: {len(phylum_cols)}")

    # Filter to train and test sets (only samples with phylum counts)
    train_mask = merged_data["sample_id"].isin(ids_train_set).values
    test_mask = merged_data["sample_id"].isin(ids_test_set).values

    X_train = merged_data.loc[train_mask, phylum_cols].values
    X_test = merged_data.loc[test_mask, phylum_cols].values
    y_train = merged_data.loc[train_mask, "pH"].values
    y_test = merged_data.loc[test_mask, "pH"].values

    metadata_test = (
        merged_data.loc[test_mask, ["study_name", "sample_id"]]
        .copy()
        .reset_index(drop=True)
    )

    env_train = merged_data.loc[train_mask, "environment"].values
    env_test = merged_data.loc[test_mask, "environment"].values

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Train environment distribution:\n{pd.Series(env_train).value_counts()}")
    print(f"Test environment distribution:\n{pd.Series(env_test).value_counts()}")

    return X_train, X_test, y_train, y_test, phylum_cols, metadata_test


def create_pipeline(model_type):
    """
    Create a scikit-learn pipeline with L1 normalization and regression model.

    Parameters
    ----------
    model_type : str
        Type of model: 'linear', 'knn', 'rf', or 'hgb'

    Returns
    -------
    Pipeline
        scikit-learn Pipeline object ready for fitting and GridSearchCV
    """
    # L1 normalization (normalize to sum to 1 for each sample)
    normalizer = Normalizer(norm="l1")

    # Select model based on model_type
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "knn":
        # Default parameters, ready for GridSearchCV tuning
        model = KNeighborsRegressor()
    elif model_type == "rf":
        # Default parameters, ready for GridSearchCV tuning
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif model_type == "hgb":
        # Default parameters, ready for GridSearchCV tuning
        model = HistGradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from: linear, knn, rf, hgb"
        )

    # Create pipeline: normalization -> model
    pipeline = Pipeline([("normalizer", normalizer), ("regressor", model)])

    return pipeline


def get_param_grid(model_type, grid_type="test"):
    """
    Get parameter grid for GridSearchCV.

    Parameters
    ----------
    model_type : str
        Type of model: 'linear', 'knn', 'rf', or 'hgb'
    grid_type : str
        'test' for small grids (2-4 combinations) or 'full' for comprehensive grids

    Returns
    -------
    dict or None
        Parameter grid dictionary, or None if model doesn't need tuning
    """
    if model_type == "linear":
        # Linear regression doesn't need hyperparameter tuning
        return None

    elif model_type == "knn":
        if grid_type == "test":
            # Small test grid: 2x2 = 4 combinations
            return {
                "regressor__n_neighbors": [5, 10],
                "regressor__weights": ["uniform", "distance"],
            }
        else:  # full
            return {
                "regressor__n_neighbors": [3, 5, 7, 10, 15, 20],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2],  # Manhattan and Euclidean distances
            }

    elif model_type == "rf":
        if grid_type == "test":
            # Small test grid: 2x2 = 4 combinations
            return {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [10, 20],
            }
        else:  # full
            return {
                "regressor__n_estimators": [100, 200, 300, 500],
                "regressor__max_depth": [10, 20, 30, None],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
                "regressor__max_features": ["sqrt", "log2", None],
            }

    elif model_type == "hgb":
        if grid_type == "test":
            # Small test grid: 2x2 = 4 combinations
            return {
                "regressor__max_iter": [100, 200],
                "regressor__learning_rate": [0.1, 0.2],
            }
        else:  # full
            return {
                "regressor__max_iter": [100, 200, 300, 500],
                "regressor__max_depth": [10, 20, 30, None],
                "regressor__learning_rate": [0.01, 0.1, 0.2],
                "regressor__min_samples_leaf": [1, 5, 10, 20],
                "regressor__l2_regularization": [0.0, 0.1, 0.5],
            }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_predictions(y_true, y_pred, metadata, output_path):
    """
    Save predictions to CSV file with study_name and sample_id.

    Parameters
    ----------
    y_true : array-like
        True pH values
    y_pred : array-like
        Predicted pH values
    metadata : pd.DataFrame
        DataFrame with study_name and sample_id columns
    output_path : str
        Path to output CSV file
    """
    results_df = pd.DataFrame(
        {
            "study_name": metadata["study_name"].values,
            "sample_id": metadata["sample_id"].values,
            "true_pH": y_true,
            "predicted_pH": np.round(y_pred, 3),
            "residual": np.round(y_true - y_pred, 3),
        }
    )

    results_df.to_csv(output_path, index=False)
    print(f"Saved test predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict pH from bacterial abundances using regression models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["linear", "knn", "rf", "hgb"],
        help="Model type: linear (Linear Regression), knn (KNN Regression), "
        "rf (Random Forest), or hgb (HistGradientBoosting)",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="jedick/microbial-DNA-pH",
        help="HuggingFace dataset repository (default: jedick/microbial-DNA-pH)",
    )
    parser.add_argument(
        "--phylum-counts",
        type=str,
        default="data/bacteria_phylum_counts.csv.xz",
        help="Path to bacteria_phylum_counts.csv.xz (default: data/bacteria_phylum_counts.csv.xz)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--grid-search",
        type=str,
        choices=["test", "full", "none"],
        default="test",
        help='Grid search mode: "test" for small grids (2-4 combinations), '
        '"full" for comprehensive grids, or "none" to skip (default: test)',
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for grid search (default: 5)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, test_metadata = (
        load_and_preprocess_data(
            args.dataset_repo, args.phylum_counts, args.random_seed
        )
    )

    # Create pipeline
    print(f"\nCreating {args.model} pipeline...")
    pipeline = create_pipeline(args.model)

    # Perform grid search if requested
    grid_search_time = None
    best_params = None
    best_cv_score = None
    n_combinations = None

    if args.grid_search != "none" and args.model != "linear":
        param_grid = get_param_grid(args.model, args.grid_search)
        if param_grid:
            # Create ParameterGrid and calculate number of combinations
            param_grid_list = list(ParameterGrid(param_grid))
            n_combinations = len(param_grid_list)

            print(
                f"\nPerforming grid search ({args.grid_search} grid, {n_combinations} combinations)..."
            )
            print(f"Parameter grid: {param_grid}")
            print(f"CV folds: {args.cv_folds}\n")

            # Prepare progress bar iterator
            if HAS_TQDM:
                param_iter = tqdm(
                    param_grid_list, desc="Grid Search Progress", unit="combination"
                )
            else:
                param_iter = param_grid_list
                print(f"Evaluating {n_combinations} parameter combinations...")

            results = []
            start_time = time.time()

            # Manual grid search with progress bar
            for i, params in enumerate(param_iter):
                # Set parameters on the pipeline
                pipeline.set_params(**params)

                # Perform cross-validation
                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=args.cv_folds,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )

                # Store mean score (negative MSE, so higher is better)
                mean_score = scores.mean()
                results.append((params, mean_score))

                # Print progress if not using tqdm
                if not HAS_TQDM and (i + 1) % max(1, n_combinations // 10) == 0:
                    print(f"  Completed {i + 1}/{n_combinations} combinations...")

            grid_search_time = time.time() - start_time

            # Find best parameters (highest score = lowest MSE)
            best_params, best_cv_score = max(results, key=lambda x: x[1])
            best_cv_score = -best_cv_score  # Convert back to positive MSE

            # Set pipeline to best parameters and fit on full training set
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train)

            print(f"\nGrid search completed in {grid_search_time:.2f} seconds")
            print(f"Best parameters: {best_params}")
            print(f"Best CV score (RMSE): {np.sqrt(best_cv_score):.4f}")
        else:
            print("No parameter grid available for this model.")
    else:
        if args.model == "linear":
            print("Linear regression doesn't require hyperparameter tuning.")
        print(f"Fitting {args.model} model with default parameters...")
        pipeline.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Train R-squared:   {train_r2:.4f}")
    print(f"Test R-squared:    {test_r2:.4f}")
    print(f"Train MAE:  {train_mae:.4f}")
    print(f"Test MAE:   {test_mae:.4f}")
    print(f"{'='*60}\n")

    # Save test set predictions only
    test_output_path = output_dir / f"{args.model}_test_predictions.csv"
    save_predictions(y_test, y_test_pred, test_metadata, test_output_path)

    # Save model summary
    summary_path = output_dir / f"{args.model}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Random Seed: {args.random_seed}\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write(f"Train set size: {X_train.shape[0]}\n")
        f.write(f"Test set size: {X_test.shape[0]}\n")
        # Grid search information
        if (
            args.grid_search != "none"
            and args.model != "linear"
            and best_params is not None
        ):
            f.write(f"\nGrid Search:\n")
            f.write(f"  Mode: {args.grid_search}\n")
            f.write(f"  CV folds: {args.cv_folds}\n")
            f.write(f"  Number of combinations: {n_combinations}\n")
            f.write(f"  Time elapsed: {grid_search_time:.2f} seconds\n")
            f.write(f"  Best parameters: {best_params}\n")
            f.write(f"  Best CV RMSE: {np.sqrt(best_cv_score):.4f}\n")
        elif args.model == "linear":
            f.write(f"\nGrid Search: Not applicable (linear regression)\n")
        else:
            f.write(f"\nGrid Search: Not performed\n")

        f.write(f"\nFinal Model Metrics:\n")
        f.write(f"Train RMSE: {train_rmse:.4f}\n")
        f.write(f"Test RMSE:  {test_rmse:.4f}\n")
        f.write(f"Train R-squared:   {train_r2:.4f}\n")
        f.write(f"Test R-squared:    {test_r2:.4f}\n")
        f.write(f"Train MAE:  {train_mae:.4f}\n")
        f.write(f"Test MAE:   {test_mae:.4f}\n")

    print(f"Saved model summary to {summary_path}")

    # Print full parameter grids for reference
    if args.model != "linear":
        import itertools

        print("\nFull parameter grids available for command-line execution:")
        full_grid = get_param_grid(args.model, "full")
        if full_grid:
            n_full_combinations = len(list(itertools.product(*full_grid.values())))
            print(f"  Full grid ({n_full_combinations} combinations):")
            for key, value in full_grid.items():
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
