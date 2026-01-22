#!/usr/bin/env python3
"""
Download SRA data from NCBI Sequence Read Archive.

This script:
- Reads sample_data.csv to find sample_ids matching SRR* or ERR* pattern
- Downloads only the first 0.25MB from NCBI SRA URLs
- Processes incomplete gzip files (extract, remove last sequence, re-gzip)
- Saves to data/fasta/<study_name>/<sample_id>.fasta.gz
- Skips existing files and handles duplicates
- Cleans up temporary files on interruption
"""

import csv
import gzip
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Set, Tuple

import requests


# Global variables for cleanup
temp_files: Set[Path] = set()
interrupted = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by cleaning up temp files."""
    global interrupted
    interrupted = True
    print("\n\nInterrupted! Cleaning up temporary files...")
    cleanup_temp_files()
    sys.exit(1)


def cleanup_temp_files():
    """Remove all temporary files."""
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove {temp_file}: {e}")
    temp_files.clear()


def download_first_mb(url: str, output_path: Path, max_bytes: int = 256 * 1024) -> bool:
    """
    Download only the first max_bytes (default 0.25MB) from a URL.

    Uses HTTP Range header to limit download size.
    """
    try:
        headers = {"Range": f"bytes=0-{max_bytes - 1}"}
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    if output_path.stat().st_size >= max_bytes:
                        break

        return True
    except Exception as e:
        print(f"    Error downloading: {e}")
        return False


def extract_incomplete_gzip(gzip_path: Path, output_path: Path) -> bool:
    """
    Extract an incomplete gzip file using zcat.

    This works even if the gzip file is truncated.
    """
    try:
        with open(output_path, "w") as outfile:
            result = subprocess.run(
                ["zcat", str(gzip_path)],
                stdout=outfile,
                stderr=subprocess.PIPE,
                text=True,
            )
            # zcat may exit with non-zero if file is incomplete, but still produces output
            # Check if we got any output
            if output_path.stat().st_size == 0 and result.returncode != 0:
                print(
                    f"    Warning: zcat may have failed (return code {result.returncode})"
                )
                if result.stderr:
                    print(f"    zcat stderr: {result.stderr[:200]}")
                return False
        return True
    except FileNotFoundError:
        print("    Error: zcat command not found. Please install gzip utilities.")
        return False
    except Exception as e:
        print(f"    Error extracting gzip: {e}")
        return False


def remove_last_sequence(fasta_path: Path, output_path: Path) -> bool:
    """
    Remove the last sequence from a FASTA file.

    The last sequence may be truncated due to incomplete download.
    """
    try:
        sequences = []
        current_header = None
        current_sequence = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_header is not None and current_sequence:
                        sequences.append((current_header, "".join(current_sequence)))
                    # Start new sequence
                    current_header = line
                    current_sequence = []
                else:
                    if line:  # Skip empty lines
                        current_sequence.append(line)

        # Don't include the last sequence (may be truncated)
        # The last sequence is the one we're currently building (current_header, current_sequence)
        # which we intentionally don't add to sequences

        if len(sequences) == 0:
            print(
                "    Warning: No complete sequences found in file (only one sequence which may be truncated)"
            )
            return False

        # Write all sequences (the last one is already excluded)
        with open(output_path, "w") as f:
            for header, sequence in sequences:
                f.write(f"{header}\n")
                # Write sequence in chunks of 80 characters (FASTA format)
                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i+80]}\n")

        return True
    except Exception as e:
        print(f"    Error processing FASTA: {e}")
        return False


def compress_fasta(fasta_path: Path, output_path: Path) -> bool:
    """Compress a FASTA file using gzip."""
    try:
        with open(fasta_path, "rb") as f_in:
            with gzip.open(output_path, "wb") as f_out:
                f_out.writelines(f_in)
        return True
    except Exception as e:
        print(f"    Error compressing FASTA: {e}")
        return False


def process_sample(study_name: str, sample_id: str, output_dir: Path) -> bool:
    """
    Download and process a single SRA sample.

    Returns True if successful, False otherwise.
    """
    # Check if output file already exists
    output_file = output_dir / f"{sample_id}.fasta.gz"
    if output_file.exists():
        return True  # Already downloaded

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct download URL
    url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra-reads-be/fasta?acc={sample_id}"

    # Create temporary files
    temp_dir = Path(tempfile.gettempdir())
    temp_gzip = temp_dir / f"sra_download_{sample_id}_{os.getpid()}.gz"
    temp_fasta = temp_dir / f"sra_extracted_{sample_id}_{os.getpid()}.fasta"
    temp_fasta_processed = temp_dir / f"sra_processed_{sample_id}_{os.getpid()}.fasta"

    temp_files.add(temp_gzip)
    temp_files.add(temp_fasta)
    temp_files.add(temp_fasta_processed)

    try:
        # Step 1: Download first 0.25MB
        if not download_first_mb(url, temp_gzip):
            return False

        # Step 2: Extract incomplete gzip
        if not extract_incomplete_gzip(temp_gzip, temp_fasta):
            return False

        # Step 3: Remove last sequence
        if not remove_last_sequence(temp_fasta, temp_fasta_processed):
            return False

        # Step 4: Compress and save
        if not compress_fasta(temp_fasta_processed, output_file):
            return False

        return True

    except Exception as e:
        print(f"    Error processing {sample_id}: {e}")
        return False
    finally:
        # Clean up temporary files
        for temp_file in [temp_gzip, temp_fasta, temp_fasta_processed]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    temp_files.discard(temp_file)
            except Exception:
                pass


def main():
    """Main function to process all samples from sample_data.csv."""
    global interrupted

    # Set up signal handler for cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "data" / "sample_data.csv"
    fasta_base_dir = script_dir / "data" / "fasta"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    # Pattern to match SRR* or ERR* sample_ids
    pattern = re.compile(r"^(SRR|ERR)\d+$")

    # Track processed sample_ids to skip duplicates
    processed_samples: Set[Tuple[str, str]] = set()

    # Statistics
    total_samples = 0
    skipped_pattern = 0
    skipped_existing = 0
    skipped_duplicate = 0
    downloaded = 0
    failed = 0

    print("Reading sample_data.csv...")
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"Found {len(rows)} rows in CSV\n")

        for row in rows:
            if interrupted:
                break

            study_name = row["study_name"]
            sample_id = row["sample_id"]

            # Skip if doesn't match pattern
            if not pattern.match(sample_id):
                skipped_pattern += 1
                continue

            total_samples += 1

            # Check for duplicates
            key = (study_name, sample_id)
            if key in processed_samples:
                skipped_duplicate += 1
                continue

            processed_samples.add(key)

            # Check if already downloaded
            output_dir = fasta_base_dir / study_name
            output_file = output_dir / f"{sample_id}.fasta.gz"
            if output_file.exists():
                skipped_existing += 1
                continue

            # Process sample
            print(f"[{total_samples}] Processing {study_name}/{sample_id}...")
            if process_sample(study_name, sample_id, output_dir):
                downloaded += 1
                print(f"    ✓ Downloaded and processed")
            else:
                failed += 1
                print(f"    ✗ Failed")

        # Final cleanup
        cleanup_temp_files()

        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total samples matching pattern: {total_samples}")
        print(f"  Downloaded: {downloaded}")
        print(f"  Skipped (already exists): {skipped_existing}")
        print(f"  Skipped (duplicate in CSV): {skipped_duplicate}")
        print(f"  Failed: {failed}")
        print(f"  Skipped (wrong pattern): {skipped_pattern}")
        print("=" * 60)

    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"\nError: {e}")
        cleanup_temp_files()
        sys.exit(1)


if __name__ == "__main__":
    main()
