#!/usr/bin/env python3
"""
Create a Hugging Face dataset from DNA sequences in fasta files and sample data.
"""

import os
import gzip
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, login

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "jedick"
HF_DATASET_NAME = "microbial-DNA-pH"
HF_DATASET_REPO = f"{HF_USERNAME}/{HF_DATASET_NAME}"

# Constants
MAX_DNA_LENGTH = 1 * 1024 * 1024  # 1 MB in bytes (1,048,576 bytes)


def parse_fasta_header(header: str) -> Optional[str]:
    """Extract sequence identifier from FASTA header."""
    # Headers look like: >SRR7775171.1 1 length=251
    match = re.match(r">(\S+)", header)
    if match:
        return match.group(1)
    return None


def read_sequences_from_fasta(
    fasta_path: Path, max_length: int = MAX_DNA_LENGTH
) -> List[str]:
    """
    Read sequences from a gzipped FASTA file until reaching max_length total DNA.
    Returns a list of sequence strings.
    """
    sequences = []
    current_sequence = []
    current_header = None
    total_length = 0

    with gzip.open(fasta_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if exists
                if current_sequence:
                    seq = "".join(current_sequence)
                    seq_len = len(seq)
                    if total_length + seq_len <= max_length:
                        sequences.append(seq)
                        total_length += seq_len
                    else:
                        # Take partial sequence if we haven't reached max yet
                        remaining = max_length - total_length
                        if remaining > 0:
                            sequences.append(seq[:remaining])
                        break

                # Start new sequence
                current_header = line
                current_sequence = []
            else:
                # Add to current sequence
                current_sequence.append(line)

        # Handle last sequence
        if current_sequence and total_length < max_length:
            seq = "".join(current_sequence)
            seq_len = len(seq)
            if total_length + seq_len <= max_length:
                sequences.append(seq)
                total_length += seq_len
            else:
                remaining = max_length - total_length
                if remaining > 0:
                    sequences.append(seq[:remaining])

    return sequences


def load_sample_data(csv_path: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load sample data from CSV and return a dictionary mapping (study, Run) -> data.
    """
    sample_data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study = row["study"]
            run_id = row["Run"]
            key = (study, run_id)
            sample_data[key] = {
                "study_name": study,
                "sample_id": run_id,
                "pH": float(row["pH"]) if row["pH"] and row["pH"] != "NA" else None,
                "sample": row["sample"],
                "envirotype": row["envirotype"],
                "lineage": row["lineage"],
            }
    return sample_data


def validate_fasta_files(
    fasta_dir: Path, sample_data: Dict[Tuple[str, str], Dict]
) -> List[Tuple[Path, str, str]]:
    """
    Validate that all fasta files have matching entries in sample_data.csv.
    Returns list of (fasta_path, study_name, sample_id) tuples.
    """
    fasta_files = []
    missing_entries = []

    # Walk through fasta directory structure: data/fasta/{study_name}/{sample_id}.fasta.gz
    for study_dir in fasta_dir.iterdir():
        if not study_dir.is_dir():
            continue

        study_name = study_dir.name
        for fasta_file in study_dir.glob("*.fasta.gz"):
            sample_id = fasta_file.stem.replace(
                ".fasta", ""
            )  # Remove .fasta.gz -> get SRR7775171

            key = (study_name, sample_id)
            if key not in sample_data:
                missing_entries.append((fasta_file, study_name, sample_id))
            else:
                fasta_files.append((fasta_file, study_name, sample_id))

    if missing_entries:
        print(
            "ERROR: The following fasta files do not have matching entries in sample_data.csv:"
        )
        for fasta_file, study_name, sample_id in missing_entries:
            print(f"  - {fasta_file}: study_name={study_name}, sample_id={sample_id}")
        raise ValueError("Some fasta files are missing from sample_data.csv")

    print(f"✓ Validated {len(fasta_files)} fasta files against sample_data.csv")
    return fasta_files


def get_existing_records(dataset_repo: str) -> set:
    """
    Get set of (study_name, sample_id) tuples that already exist in the dataset.
    """
    try:
        dataset = load_dataset(dataset_repo, split="train", token=HF_TOKEN)
        existing = set()
        for row in dataset:
            existing.add((row["study_name"], row["sample_id"]))
        print(f"✓ Found {len(existing)} existing records in dataset")
        return existing
    except Exception as e:
        # Dataset doesn't exist yet or is empty
        print(f"  Dataset doesn't exist yet or is empty: {e}")
        return set()


def create_dataset_records(
    fasta_files: List[Tuple[Path, str, str]],
    sample_data: Dict[Tuple[str, str], Dict],
    existing_records: set,
) -> List[Dict]:
    """
    Create dataset records from fasta files.
    Returns a list of dictionaries with study_name, sample_id, pH, and sequences.
    """
    records = []

    for fasta_path, study_name, sample_id in fasta_files:
        key = (study_name, sample_id)

        # Skip if already exists
        if key in existing_records:
            print(f"  Skipping {study_name}/{sample_id} (already exists)")
            continue

        print(f"  Processing {study_name}/{sample_id}...")

        # Read sequences
        sequences = read_sequences_from_fasta(fasta_path)
        total_length = sum(len(seq) for seq in sequences)

        # Get sample data
        data = sample_data[key]

        # Create record
        record = {
            "study_name": study_name,
            "sample_id": sample_id,
            "pH": data["pH"],
            "sequences": sequences,  # List of strings for easy slicing
            "num_sequences": len(sequences),
            "total_dna_length": total_length,
            "sample": data["sample"],
            "envirotype": data["envirotype"],
            "lineage": data["lineage"],
        }

        records.append(record)
        print(f"    Read {len(sequences)} sequences ({total_length:,} nucleotides)")

    return records


def upload_dataset(records: List[Dict], dataset_repo: str):
    """
    Create or update the Hugging Face dataset with new records.
    """
    if not records:
        print("No new records to upload")
        return

    print(f"\nCreating dataset with {len(records)} records...")

    # Check if dataset already exists and needs merging
    try:
        existing_dataset = load_dataset(dataset_repo, split="train", token=HF_TOKEN)
        print(f"  Merging with {len(existing_dataset)} existing records...")
        # Convert existing dataset to list and append new records
        all_records = list(existing_dataset) + records
        dataset = Dataset.from_list(all_records)
    except Exception:
        # Dataset doesn't exist, create new one
        print("  Creating new dataset...")
        dataset = Dataset.from_list(records)

    # Push to Hugging Face
    print(f"Uploading to {dataset_repo}...")
    dataset.push_to_hub(dataset_repo, token=HF_TOKEN, private=False)
    print(f"✓ Successfully uploaded {len(records)} new records to {dataset_repo}")


def main():
    """Main function to orchestrate the dataset creation."""
    # Paths
    project_root = Path(__file__).parent
    fasta_dir = project_root / "data" / "fasta"
    sample_data_path = project_root / "data" / "sample_data.csv"

    # Check HF_TOKEN
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in .env file")

    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)

    # Load sample data
    print("\nLoading sample data...")
    sample_data = load_sample_data(sample_data_path)
    print(f"✓ Loaded {len(sample_data)} sample records")

    # Validate fasta files
    print("\nValidating fasta files...")
    fasta_files = validate_fasta_files(fasta_dir, sample_data)

    # Check existing records
    print("\nChecking existing dataset records...")
    existing_records = get_existing_records(HF_DATASET_REPO)

    # Create dataset records
    print("\nProcessing fasta files...")
    records = create_dataset_records(fasta_files, sample_data, existing_records)

    # Upload dataset
    if records:
        print("\nUploading dataset...")
        upload_dataset(records, HF_DATASET_REPO)
    else:
        print("\nAll records already exist in the dataset. Nothing to upload.")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
