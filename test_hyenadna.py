#!/usr/bin/env python3
"""
Run test predictions for trained HyenaDNA pH prediction model.

This script loads a trained checkpoint and generates predictions on the test set,
saving results to a CSV file.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add hyena-dna to path
sys.path.insert(0, str(Path(__file__).parent / "hyena-dna"))

from huggingface_wrapper import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer

# Import from train script
from train_hyenadna import (
    MODEL_CONFIGS,
    HyenaDNAForRegression,
    RegressionHead,
    collate_fn,
    load_dataset_from_hf,
    pHDataset,
    setup_model,
)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    head_architecture: str,
    pooling_mode: str,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    download: bool = False,
) -> Tuple[nn.Module, CharacterTokenizer, int]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Model name from MODEL_CONFIGS
        head_architecture: Regression head architecture
        pooling_mode: Pooling mode for regression head
        checkpoint_dir: Directory for model checkpoints
        device: Device to use
        download: Whether to download model if not present

    Returns:
        (model, tokenizer, max_length)
    """
    # Setup model architecture
    model, tokenizer, max_length = setup_model(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        head_architecture=head_architecture,
        pooling_mode=pooling_mode,
        freeze_backbone=False,  # Not relevant for inference
        device=device,
        download=download,
    )

    # Wrap model
    wrapped_model = HyenaDNAForRegression(model, model.regression_head)
    wrapped_model = wrapped_model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    wrapped_model.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully")

    return wrapped_model, tokenizer, max_length


def run_test_predictions(
    checkpoint_path: str,
    output_file: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Run test predictions and save to CSV.

    All hyperparameters are loaded from config.json in the checkpoint directory.
    This ensures consistency with the training configuration.

    Args:
        checkpoint_path: Path to checkpoint file (e.g., best_model.pt)
        output_file: Path to output CSV file (e.g., test_predictions.csv)
        device: Device to use (runtime setting, not in config.json)

    Returns:
        Dictionary with test metrics
    """
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config.json from the same directory as checkpoint (required)
    config_path = checkpoint_path_obj.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in {checkpoint_path_obj.parent}. "
            "config.json is required to load model hyperparameters."
        )

    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract all hyperparameters from config.json
    model_name = config.get("model_name")
    head_architecture = config.get("head_architecture")
    pooling_mode = config.get("pooling_mode")
    dataset_repo = config.get("dataset_repo", "jedick/microbial-DNA-pH")
    random_seed = config.get("random_seed", 42)
    batch_size = config.get("batch_size", 4)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    download = config.get("download_model", False)

    # Validate required config values
    if model_name is None or head_architecture is None or pooling_mode is None:
        raise ValueError(
            f"Missing required configuration in {config_path}: "
            "model_name, head_architecture, and pooling_mode are required."
        )

    print(
        f"Loaded config: model_name={model_name}, head_architecture={head_architecture}, "
        f"pooling_mode={pooling_mode}, batch_size={batch_size}, random_seed={random_seed}"
    )

    # Load model
    model, tokenizer, max_length = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        head_architecture=head_architecture,
        pooling_mode=pooling_mode,
        checkpoint_dir=checkpoint_dir,
        device=device,
        download=download,
    )

    # Load dataset
    print(f"Loading dataset from {dataset_repo}...")
    data = load_dataset_from_hf(dataset_repo, filter_missing_ph=True)

    # Create same train/val/test split as training
    print("Creating stratified train-val-test split (70:10:20)...")
    sample_ids = [d["sample_id"] for d in data]
    ph_values = [d["pH"] for d in data]
    environments = [d["environment"] for d in data]

    # First split: 80% train+val, 20% test
    ids_trainval, ids_test, data_trainval, data_test, env_trainval, env_test = (
        train_test_split(
            sample_ids,
            data,
            environments,
            test_size=0.2,
            random_state=random_seed,
            stratify=environments,
        )
    )

    # Second split: split train+val into 70% train and 10% val
    n_total = len(data)
    n_val = int(round(n_total * 0.10))
    val_size = n_val / len(data_trainval)

    ids_train, ids_val, data_train, data_val = train_test_split(
        ids_trainval,
        data_trainval,
        test_size=val_size,
        random_state=random_seed,
        stratify=env_trainval,
    )

    test_data = data_test
    print(f"Test set size: {len(test_data)}")

    # Create test dataset and dataloader
    test_dataset = pHDataset(test_data, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Run inference once to collect predictions, true values, and metadata
    print("\nRunning inference on test set...")
    model.eval()
    test_preds = []
    test_true = []
    test_study_names = []
    test_sample_ids = []
    test_set_indices = []
    criterion = nn.SmoothL1Loss()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            ph_true = batch["ph"].to(device)
            ph_pred = model(input_ids)

            # Compute loss for metrics
            loss = criterion(ph_pred, ph_true)
            total_loss += loss.item()
            num_batches += 1

            # Store predictions, true values, and metadata
            test_preds.extend(ph_pred.cpu().numpy())
            test_true.extend(ph_true.cpu().numpy())
            test_study_names.extend(batch["study_name"])
            test_sample_ids.extend(batch["sample_id"])
            test_set_indices.extend(batch["set_idx"])

    # Compute metrics from collected predictions
    test_preds_array = np.array(test_preds)
    test_true_array = np.array(test_true)

    mse = mean_squared_error(test_true_array, test_preds_array)
    mae = mean_absolute_error(test_true_array, test_preds_array)
    r2 = r2_score(test_true_array, test_preds_array)

    test_metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mae,
        "r2": r2,
    }

    print(f"\nTest metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Aggregate predictions by sample_id
    # Each sample_id should have up to 5 predictions (one per set)
    sample_data = defaultdict(
        lambda: {
            "study_name": None,
            "true_ph": None,
            "predictions": {},  # set_idx -> predicted_ph
        }
    )

    for i, sample_id in enumerate(test_sample_ids):
        set_idx = test_set_indices[i]
        sample_data[sample_id]["study_name"] = test_study_names[i]
        sample_data[sample_id]["true_ph"] = test_true[i]
        sample_data[sample_id]["predictions"][set_idx] = test_preds[i]

    # Create DataFrame with one row per sample_id
    rows = []
    for sample_id, data in sorted(sample_data.items()):
        predictions = data["predictions"]
        # Get predictions for sets 1-5, fill with NaN if missing
        pred_1 = predictions.get(1, np.nan)
        pred_2 = predictions.get(2, np.nan)
        pred_3 = predictions.get(3, np.nan)
        pred_4 = predictions.get(4, np.nan)
        pred_5 = predictions.get(5, np.nan)

        # Calculate mean of available predictions
        available_preds = [
            p for p in [pred_1, pred_2, pred_3, pred_4, pred_5] if not np.isnan(p)
        ]
        mean_pred = np.mean(available_preds) if available_preds else np.nan

        rows.append(
            {
                "sample_id": sample_id,
                "study_name": data["study_name"],
                "true_ph": data["true_ph"],
                "predicted_ph_set1": (
                    np.round(pred_1, 3) if not np.isnan(pred_1) else np.nan
                ),
                "predicted_ph_set2": (
                    np.round(pred_2, 3) if not np.isnan(pred_2) else np.nan
                ),
                "predicted_ph_set3": (
                    np.round(pred_3, 3) if not np.isnan(pred_3) else np.nan
                ),
                "predicted_ph_set4": (
                    np.round(pred_4, 3) if not np.isnan(pred_4) else np.nan
                ),
                "predicted_ph_set5": (
                    np.round(pred_5, 3) if not np.isnan(pred_5) else np.nan
                ),
                "predicted_ph_mean": (
                    np.round(mean_pred, 3) if not np.isnan(mean_pred) else np.nan
                ),
                "residual_mean": (
                    np.round(data["true_ph"] - mean_pred, 3)
                    if not np.isnan(mean_pred)
                    else np.nan
                ),
            }
        )

    predictions_df = pd.DataFrame(rows)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"\nSaved test predictions to {output_path}")
    print(
        f"  Aggregated {len(test_sample_ids)} predictions into {len(predictions_df)} unique samples"
    )

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run test predictions for HyenaDNA pH prediction model. "
        "All hyperparameters are loaded from config.json in the checkpoint directory."
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., best_model.pt). "
        "config.json must exist in the same directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file (e.g., test_predictions.csv)",
    )

    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run test predictions (all hyperparameters loaded from config.json)
    test_metrics = run_test_predictions(
        checkpoint_path=args.checkpoint,
        output_file=args.output,
        device=device,
    )

    print("\nTest evaluation complete!")
    print(f"Final metrics: {test_metrics}")


if __name__ == "__main__":
    main()
