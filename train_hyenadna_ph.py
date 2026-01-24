#!/usr/bin/env python3
"""
Train HyenaDNA model for pH prediction from DNA sequences.

This script implements a regression head for HyenaDNA to predict continuous pH values.
It loads data from HuggingFace, concatenates multiple sequences with [SEP] tokens,
and trains using pretrained HyenaDNA weights.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add hyena-dna to path
sys.path.insert(0, str(Path(__file__).parent / "hyena-dna"))

from huggingface_wrapper import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer, HyenaDNAModel

# Model size configurations
MODEL_CONFIGS = {
    "hyenadna-tiny-1k-seqlen": {"max_length": 1024},
    "hyenadna-small-32k-seqlen": {"max_length": 32768},
    "hyenadna-medium-160k-seqlen": {"max_length": 160000},
    "hyenadna-medium-450k-seqlen": {"max_length": 450000},
    "hyenadna-large-1m-seqlen": {"max_length": 1_000_000},
}

# Special token IDs
SEP_TOKEN_ID = 1
PAD_TOKEN_ID = 4
CLS_TOKEN_ID = 0


class RegressionHead(nn.Module):
    """Regression head for pH prediction from HyenaDNA embeddings."""

    def __init__(
        self,
        d_model: int,
        architecture: str = "mlp2",
        pooling_mode: str = "pool",
        dropout: float = 0.1,
    ):
        """
        Initialize regression head.

        Args:
            d_model: Hidden dimension from backbone
            architecture: One of 'linear', 'mlp2', 'mlp3', 'mlp2_ln'
            pooling_mode: One of 'pool', 'last', 'first', 'sum'
            dropout: Dropout rate for MLP layers
        """
        super().__init__()
        self.d_model = d_model
        self.pooling_mode = pooling_mode
        self.architecture = architecture

        # Build output transform based on architecture
        if architecture == "linear":
            self.output_transform = nn.Linear(d_model, 1)
        elif architecture == "mlp2":
            self.output_transform = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        elif architecture == "mlp3":
            self.output_transform = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        elif architecture == "mlp2_ln":
            self.output_transform = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                "Choose from: linear, mlp2, mlp3, mlp2_ln"
            )

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence representations.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, d_model)
        """
        if self.pooling_mode == "pool":
            # Mean pooling
            return x.mean(dim=1)
        elif self.pooling_mode == "last":
            # Last token
            return x[:, -1, :]
        elif self.pooling_mode == "first":
            # First token
            return x[:, 0, :]
        elif self.pooling_mode == "sum":
            # Sum pooling
            return x.sum(dim=1)
        else:
            raise ValueError(
                f"Unknown pooling mode: {self.pooling_mode}. "
                "Choose from: pool, last, first, sum"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (batch, seq_len, d_model) from backbone

        Returns:
            (batch,) pH predictions
        """
        # Pool to sequence-level representation
        pooled = self.pool(hidden_states)  # (batch, d_model)

        # Transform to pH value
        ph_pred = self.output_transform(pooled)  # (batch, 1)

        # Squeeze to (batch,)
        return ph_pred.squeeze(-1)


class pHDataset(Dataset):
    """Dataset for pH prediction from DNA sequences."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: CharacterTokenizer,
        max_length: int,
        sep_token_id: int = SEP_TOKEN_ID,
        num_sets: int = 5,
    ):
        """
        Initialize dataset.

        Args:
            data: List of dicts with 'sequences' (list of str) and 'pH' (float)
            tokenizer: CharacterTokenizer instance
            max_length: Maximum sequence length
            sep_token_id: [SEP] token ID for concatenation
            num_sets: Number of non-overlapping sequence sets per sample (default: 5)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token_id = sep_token_id
        self.num_sets = num_sets

        # Expand data: each original sample becomes num_sets samples
        # Pre-compute sequence sets for each sample
        self.expanded_data = []
        for item in data:
            sequences = item["sequences"]
            # Split sequences into num_sets non-overlapping sets
            sequence_sets = split_sequences_into_sets(sequences, max_length, num_sets)

            # Create one entry per set
            for set_idx, seq_set in enumerate(sequence_sets):
                if seq_set:  # Only add non-empty sets
                    self.expanded_data.append(
                        {
                            "sequences": seq_set,
                            "pH": item["pH"],
                            "study_name": item.get("study_name", ""),
                            "sample_id": item.get("sample_id", ""),
                            "set_idx": set_idx + 1,  # 1-indexed for output
                        }
                    )

    def __len__(self):
        return len(self.expanded_data)

    def __getitem__(self, idx):
        item = self.expanded_data[idx]
        sequences = item["sequences"]
        ph = item["pH"]

        # Concatenate sequences with [SEP] tokens
        token_ids = concatenate_sequences(
            sequences, self.tokenizer, self.max_length, self.sep_token_id
        )

        return {
            "input_ids": torch.LongTensor(token_ids),
            "ph": torch.tensor(ph, dtype=torch.float32),
            "study_name": item.get("study_name", ""),
            "sample_id": item.get("sample_id", ""),
            "set_idx": item.get("set_idx", 1),
        }


def split_sequences_into_sets(
    sequences: List[str],
    max_length: int,
    num_sets: int = 5,
) -> List[List[str]]:
    """
    Split sequences into non-overlapping sets, each fitting within max_length.

    Args:
        sequences: List of DNA sequence strings
        max_length: Maximum total length for each set (in characters, before tokenization)
        num_sets: Number of non-overlapping sets to create

    Returns:
        List of num_sets lists of sequences, each fitting within max_length
    """
    if not sequences:
        # Return num_sets empty sets
        return [[] for _ in range(num_sets)]

    # Filter out empty sequences
    valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]

    if not valid_sequences:
        return [[] for _ in range(num_sets)]

    # Create num_sets non-overlapping sets
    sets = [[] for _ in range(num_sets)]
    current_set_idx = 0
    sequence_idx = 0

    while sequence_idx < len(valid_sequences) and current_set_idx < num_sets:
        current_set = sets[current_set_idx]
        cumulative_length = sum(len(seq) for seq in current_set)

        # Try to add sequences to current set
        while sequence_idx < len(valid_sequences):
            seq = valid_sequences[sequence_idx]
            seq_len = len(seq)
            # Estimate tokens needed: seq_len characters + 1 SEP token (if not first sequence)
            tokens_needed = seq_len + (1 if current_set else 0)

            if cumulative_length + tokens_needed <= max_length:
                current_set.append(seq)
                cumulative_length += tokens_needed
                sequence_idx += 1
            else:
                # This sequence would exceed max_length for current set
                # Move to next set
                break

        # Move to next set (either we filled this one or can't add more)
        current_set_idx += 1

        # If all sequences processed, we're done
        if sequence_idx >= len(valid_sequences):
            break

    return sets


def concatenate_sequences(
    sequences: List[str],
    tokenizer: CharacterTokenizer,
    max_length: int,
    sep_token_id: int = SEP_TOKEN_ID,
) -> List[int]:
    """
    Concatenate multiple DNA sequences with [SEP] tokens.

    Args:
        sequences: List of DNA sequence strings
        tokenizer: CharacterTokenizer instance
        max_length: Maximum total length (including special tokens)
        sep_token_id: [SEP] token ID

    Returns:
        List of token IDs of length <= max_length
    """
    if not sequences:
        # Return single [SEP] token if no sequences
        return [sep_token_id]

    # Filter out empty sequences first
    valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]

    if not valid_sequences:
        # No valid sequences, return single [SEP]
        return [sep_token_id]

    # Precompute how many sequences we need based on character lengths
    # For character tokenizer, each DNA character maps to one token (after filtering special tokens)
    # We need to account for: sequence lengths + (n-1) SEP tokens for n sequences
    cumulative_length = 0
    sequences_to_use = []

    for seq in valid_sequences:
        seq_len = len(seq)
        # Estimate tokens needed: seq_len characters + 1 SEP token (if not first sequence)
        tokens_needed = seq_len + (1 if sequences_to_use else 0)

        if cumulative_length + tokens_needed <= max_length:
            sequences_to_use.append(seq)
            cumulative_length += tokens_needed
        else:
            # This sequence would exceed max_length, stop here
            break

    if not sequences_to_use:
        # Even the first sequence is too long, we'll truncate it
        sequences_to_use = [valid_sequences[0]]

    # Now tokenize only the sequences we'll use
    tokenized_seqs = []
    for seq in sequences_to_use:
        # Tokenize and extract input_ids
        tokens = tokenizer(seq)["input_ids"]
        # Remove special tokens (keep only DNA character tokens, which have IDs >= 7)
        dna_tokens = [t for t in tokens if t >= 7]
        if dna_tokens:
            tokenized_seqs.append(dna_tokens)

    if not tokenized_seqs:
        # No valid tokens after filtering, return single [SEP]
        return [sep_token_id]

    # Concatenate with [SEP] tokens between sequences
    result = []
    for i, seq_tokens in enumerate(tokenized_seqs):
        if i > 0:
            result.append(sep_token_id)
        result.extend(seq_tokens)

    # Truncate to max_length if needed (keep last part for causal model)
    if len(result) > max_length:
        result = result[-max_length:]

    return result


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Args:
        batch: List of dicts with 'input_ids', 'ph', 'study_name', 'sample_id', 'set_idx'

    Returns:
        Dictionary with batched tensors (input_ids, attention_mask, ph) and
        lists (study_name, sample_id, set_idx)
    """
    input_ids = [item["input_ids"] for item in batch]
    ph_values = torch.stack([item["ph"] for item in batch])
    study_names = [item.get("study_name", "") for item in batch]
    sample_ids = [item.get("sample_id", "") for item in batch]
    set_indices = [item.get("set_idx", 1) for item in batch]

    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad sequences to max length (left padding for causal model)
    padded_ids = []
    attention_masks = []
    for ids in input_ids:
        pad_length = max_len - len(ids)
        # Left pad with PAD_TOKEN_ID
        padded = [PAD_TOKEN_ID] * pad_length + ids.tolist()
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = [0] * pad_length + [1] * len(ids)
        padded_ids.append(padded)
        attention_masks.append(mask)

    return {
        "input_ids": torch.LongTensor(padded_ids),
        "attention_mask": torch.LongTensor(attention_masks),
        "ph": ph_values,
        "study_name": study_names,
        "sample_id": sample_ids,
        "set_idx": set_indices,
    }


def load_dataset_from_hf(
    dataset_repo: str = "jedick/microbial-DNA-pH",
    filter_missing_ph: bool = True,
) -> List[Dict]:
    """
    Load dataset from HuggingFace.

    Includes study_name and sample_id for subsetting by shared train/val/test splits.
    HF dataset contains SRA samples only (SRR*, ERR*, DRR*).

    Args:
        dataset_repo: HuggingFace dataset repository
        filter_missing_ph: Whether to filter samples with missing pH

    Returns:
        List of dicts with 'sequences', 'pH', 'study_name', 'sample_id', 'environment'
        Sorted by sample_id for consistent ordering across workflows.
    """
    print(f"Loading dataset from {dataset_repo}...")
    dataset = load_dataset(dataset_repo, split="train")

    data = []
    for item in dataset:
        # Filter missing pH if requested
        if filter_missing_ph and item["pH"] is None:
            continue

        data.append(
            {
                "sequences": item["sequences"],
                "pH": item["pH"] if item["pH"] is not None else 0.0,
                "study_name": item["study_name"],
                "sample_id": item["sample_id"],
                "environment": item.get("environment", "unknown"),
            }
        )

    # Sort by sample_id to ensure consistent ordering across workflows
    data.sort(key=lambda x: x["sample_id"])

    print(f"Loaded {len(data)} samples")
    return data


def setup_model(
    model_name: str = "hyenadna-tiny-1k-seqlen",
    checkpoint_dir: str = "./checkpoints",
    head_architecture: str = "mlp2",
    pooling_mode: str = "pool",
    freeze_backbone: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    download: bool = False,
) -> Tuple[nn.Module, CharacterTokenizer, int]:
    """
    Load pretrained HyenaDNA model and attach regression head.

    Args:
        model_name: Model name from MODEL_CONFIGS
        checkpoint_dir: Directory for model checkpoints
        head_architecture: Regression head architecture
        pooling_mode: Pooling mode for regression head
        freeze_backbone: Whether to freeze backbone weights
        device: Device to use
        download: Whether to download model if not present

    Returns:
        (model, tokenizer, max_length)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    max_length = MODEL_CONFIGS[model_name]["max_length"]
    print(f"Loading model: {model_name} (max_length={max_length})...")

    # Load pretrained model (embeddings only, no head)
    model = HyenaDNAPreTrainedModel.from_pretrained(
        checkpoint_dir,
        model_name,
        download=download,
        config=None,
        device=device,
        use_head=False,
        n_classes=2,  # Not used when use_head=False
    )

    # Get d_model from model config
    # The model is a HyenaDNAModel, access d_model from backbone
    d_model = model.backbone.embeddings.word_embeddings.embedding_dim

    # Create and attach regression head
    regression_head = RegressionHead(
        d_model=d_model,
        architecture=head_architecture,
        pooling_mode=pooling_mode,
    )
    model.regression_head = regression_head

    # Freeze backbone if requested
    if freeze_backbone:
        print("Freezing backbone weights...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Only regression head will be trained
        for param in model.regression_head.parameters():
            param.requires_grad = True

    # Move to device
    model = model.to(device)

    # Create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "C", "G", "T", "N"],
        model_max_length=max_length + 2,  # Account for special tokens
        padding_side="left",  # Causal model uses left padding
    )

    print(f"Model setup complete. d_model={d_model}, device={device}")
    return model, tokenizer, max_length


class HyenaDNAForRegression(nn.Module):
    """Wrapper model combining HyenaDNA backbone and regression head."""

    def __init__(self, hyenadna_model: nn.Module, regression_head: nn.Module):
        super().__init__()
        self.hyenadna_model = hyenadna_model
        self.regression_head = regression_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            (batch,) pH predictions
        """
        # Get embeddings from HyenaDNA model (when use_head=False, returns hidden_states)
        hidden_states = self.hyenadna_model(input_ids)  # (batch, seq_len, d_model)

        # Apply regression head
        ph_pred = self.regression_head(hidden_states)  # (batch,)

        return ph_pred


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ph_true = batch["ph"].to(device)

        # Forward pass
        ph_pred = model(input_ids)

        # Compute loss (only on non-padding tokens if using attention mask)
        loss = criterion(ph_pred, ph_true)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_true = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            ph_true = batch["ph"].to(device)

            # Forward pass
            ph_pred = model(input_ids)

            # Compute loss
            loss = criterion(ph_pred, ph_true)

            # Store predictions
            all_preds.extend(ph_pred.cpu().numpy())
            all_true.extend(ph_true.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    # Compute metrics
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    mse = mean_squared_error(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mae,
        "r2": r2,
    }


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }

    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train HyenaDNA for pH prediction")

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="hyenadna-tiny-1k-seqlen",
        choices=list(MODEL_CONFIGS.keys()),
        help="HyenaDNA model name",
    )
    parser.add_argument(
        "--head-architecture",
        type=str,
        default="mlp2",
        choices=["linear", "mlp2", "mlp3", "mlp2_ln"],
        help="Regression head architecture",
    )
    parser.add_argument(
        "--pooling-mode",
        type=str,
        default="pool",
        choices=["pool", "last", "first", "sum"],
        help="Pooling mode for regression head",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone weights (only train head)",
    )

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--loss-function",
        type=str,
        default="smooth_l1",
        choices=["mse", "mae", "smooth_l1", "huber"],
        help="Loss function",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for AdamW"
    )

    # Data arguments
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="jedick/microbial-DNA-pH",
        help="HuggingFace dataset repository",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyenadna_ph",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model from HuggingFace if not present",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved configuration to {config_path}")

    # Load dataset (HF has SRA samples only; includes study_name, sample_id)
    data = load_dataset_from_hf(args.dataset_repo, filter_missing_ph=True)

    # Create stratified 70:10:20 split (same test set as traditional ML)
    # First split 80:20 (train+val : test) with same random seed
    # Then split 80% into 70% train and 10% val
    print("Creating stratified train-val-test split (70:10:20)...")

    # Extract sample_ids, pH, and environment for stratification
    # Data is already sorted by sample_id from load_dataset_from_hf
    sample_ids = [d["sample_id"] for d in data]
    ph_values = [d["pH"] for d in data]
    environments = [d["environment"] for d in data]

    # First split: 80% train+val, 20% test (same as traditional ML)
    ids_trainval, ids_test, data_trainval, data_test, env_trainval, env_test = (
        train_test_split(
            sample_ids,
            data,
            environments,
            test_size=0.2,
            random_state=args.random_seed,
            stratify=environments,
        )
    )

    # Second split: split train+val into 70% train and 10% val
    # val_size = 10% of total = 10/80 = 0.125 of trainval
    n_total = len(data)
    n_val = int(round(n_total * 0.10))
    val_size = n_val / len(data_trainval)

    ids_train, ids_val, data_train, data_val = train_test_split(
        ids_trainval,
        data_trainval,
        test_size=val_size,
        random_state=args.random_seed,
        stratify=env_trainval,
    )

    train_data = data_train
    val_data = data_val
    test_data = data_test

    print(
        f"Data split (70:10:20): train={len(train_data)}, val={len(val_data)}, "
        f"test={len(test_data)}"
    )

    # Setup model
    model, tokenizer, max_length = setup_model(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        head_architecture=args.head_architecture,
        pooling_mode=args.pooling_mode,
        freeze_backbone=args.freeze_backbone,
        device=device,
        download=args.download_model,
    )

    # The model already has regression_head attached
    # We can use it directly, but wrap for cleaner interface
    wrapped_model = HyenaDNAForRegression(model, model.regression_head)
    wrapped_model = wrapped_model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(
        p.numel() for p in wrapped_model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    train_dataset = pHDataset(train_data, tokenizer, max_length)
    val_dataset = pHDataset(val_data, tokenizer, max_length)
    test_dataset = pHDataset(test_data, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Setup loss function
    if args.loss_function == "mse":
        criterion = nn.MSELoss()
    elif args.loss_function == "mae":
        criterion = nn.L1Loss()
    elif args.loss_function == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    elif args.loss_function == "huber":
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        wrapped_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_val_loss = float("inf")
    training_log = []

    print("\nStarting training...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        # Train
        train_loss = train_epoch(
            wrapped_model, train_loader, optimizer, criterion, device
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_metrics = evaluate(wrapped_model, val_loader, criterion, device)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val MSE: {val_metrics['mse']:.4f}")
        print(f"Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f}")
        print(f"Val R²: {val_metrics['r2']:.4f}")

        # Save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        save_checkpoint(wrapped_model, epoch, val_metrics, output_dir, is_best=is_best)

        # Log metrics
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        training_log.append(log_entry)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(wrapped_model, test_loader, criterion, device)
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\nSaved training log to {log_path}")

    # Save test predictions
    wrapped_model.eval()
    test_preds = []
    test_true = []
    test_study_names = []
    test_sample_ids = []
    test_set_indices = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            ph_true = batch["ph"].to(device)
            ph_pred = wrapped_model(input_ids)
            test_preds.extend(ph_pred.cpu().numpy())
            test_true.extend(ph_true.cpu().numpy())
            test_study_names.extend(batch["study_name"])
            test_sample_ids.extend(batch["sample_id"])
            test_set_indices.extend(batch["set_idx"])

    # Aggregate predictions by sample_id
    # Each sample_id should have up to 5 predictions (one per set)
    import pandas as pd
    from collections import defaultdict

    # Group predictions by sample_id
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
                "study_name": data["study_name"],
                "sample_id": sample_id,
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
    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved test predictions to {predictions_path}")
    print(
        f"  Aggregated {len(test_sample_ids)} predictions into {len(predictions_df)} unique samples"
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
