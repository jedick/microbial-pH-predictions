"""
Process RDP classifier tab files to extract and aggregate taxonomic counts.

This module provides functionality to parse RDP classifier output files,
extract taxonomic information from lineage strings, and aggregate counts
by taxonomic level (phylum, class, order, family, genus).
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional


def process_rdp_data(filename: str, domain: str = "Bacteria") -> dict:
    """
    Process RDP classifier tab file and return aggregated taxonomic counts.

    Parameters
    ----------
    filename : str
        Path to the RDP classifier tab file (e.g., "APV+20.tab")
    domain : str, optional
        Domain to filter by: "Bacteria" or "Archaea" (default: "Bacteria")

    Returns
    -------
    dict
        Dictionary with structure {sample_id: {taxon_name: count}}.
        Taxonomic names use double underscores (e.g., "phylum__Omnitrophota",
        "class__Koll11").

    Examples
    --------
    >>> result = process_rdp_data("data/RDP_classifier/APV+20.tab", domain="Bacteria")
    >>> print(result['SRR7775171']['phylum__Omnitrophota'])
    """
    # Validate domain
    if domain not in ["Bacteria", "Archaea"]:
        raise ValueError(f"domain must be 'Bacteria' or 'Archaea', got '{domain}'")

    # Read the tab file
    df = pd.read_csv(filename, sep="\t")

    # Filter by domain
    domain_pattern = f"{domain};domain"
    df = df[df["lineage"].str.contains(domain_pattern, na=False, regex=False)]

    # Filter lineages: must end with "genus;" or have "unclassified_" in last position
    def is_valid_lineage(lineage: str) -> bool:
        if pd.isna(lineage):
            return False
        # Check if ends with "genus;"
        if lineage.endswith("genus;"):
            return True
        # Check if has "unclassified_" in the last position before final semicolons
        # Pattern: ...;unclassified_XXX;;
        if re.search(r";unclassified_[^;]+;;$", lineage):
            return True
        return False

    df = df[df["lineage"].apply(is_valid_lineage)]

    # Get sample columns (all columns except first 4: taxid, lineage, name, rank)
    sample_cols = df.columns[4:].tolist()

    # Dictionary to accumulate counts: {sample_id: {taxon_name: count}}
    # This structure is more memory-efficient than creating intermediate lists
    sample_counts = {}

    # Initialize all sample dictionaries
    for sample_col in sample_cols:
        sample_counts[sample_col] = {}

    # Process each row
    for idx, row in df.iterrows():
        lineage = row["lineage"]

        # Parse taxonomic names from lineage
        taxa = parse_lineage(lineage)

        # Get counts for this row across all samples
        for sample_col in sample_cols:
            count = row[sample_col]
            if pd.isna(count) or count == 0:
                continue

            # Convert to int early (source data has integer counts)
            count = int(count)

            # Add counts for each taxonomic level present
            for taxon_name in taxa:
                if taxon_name not in sample_counts[sample_col]:
                    sample_counts[sample_col][taxon_name] = 0
                sample_counts[sample_col][taxon_name] += count

    return sample_counts


def parse_lineage(lineage: str) -> list:
    """
    Parse lineage string to extract taxonomic names at each level.

    Parameters
    ----------
    lineage : str
        Lineage string in format:
        "Root;rootrank;{domain};domain;{phylum};phylum;{class};class;..."

    Returns
    -------
    list
        List of taxonomic names with prefixes (e.g.,
        ["phylum__Omnitrophota", "class__Koll11", ...])
    """
    taxa = []

    # Pattern to match taxonomic levels: {name};{level};
    # We want to extract pairs where level is phylum, class, order, family, or genus
    pattern = r"([^;]+);(phylum|class|order|family|genus);"
    matches = re.findall(pattern, lineage)

    for name, level in matches:
        # Skip empty names
        if not name or name.strip() == "":
            continue
        taxon_name = f"{level}__{name}"
        taxa.append(taxon_name)

    # Handle unclassified case: ...;unclassified_XXX;;
    # For unclassified lineages, we only include the known taxonomic levels
    # (those that appear before the unclassified marker), not the unclassified itself
    # The unclassified marker indicates unknown classification at that level,
    # so we don't create a separate taxon entry for it
    unclassified_match = re.search(r";unclassified_([^;]+);;$", lineage)
    if unclassified_match:
        # For unclassified lineages, we've already extracted all the known levels
        # above. The unclassified itself should NOT be added as a separate taxon.
        # The counts will contribute to the known levels only.
        pass

    return taxa


def process_rdp_directory(
    directory: str,
    domain: str = "Bacteria",
    output_file: Optional[str] = None,
    taxonomic_level: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process all RDP classifier tab files in a directory and combine results.

    Parameters
    ----------
    directory : str
        Path to directory containing RDP classifier tab files (e.g., "data/RDP_classifier")
    domain : str, optional
        Domain to filter by: "Bacteria" or "Archaea" (default: "Bacteria")
    output_file : str, optional
        Path to save the combined results as CSV (e.g., "data/bacteria_counts.csv").
        If None, defaults to "data/{domain.lower()}_counts.csv" or
        "data/{domain.lower()}_{level}_counts.csv" if taxonomic_level is specified
    taxonomic_level : str, optional
        Taxonomic level to filter results: "phylum", "class", "order", "family", or "genus".
        If None, includes all levels (default: None)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with index sample_id and taxonomic names as columns.
        All samples from all files are included. If taxonomic_level is specified,
        only columns for that level are included.

    Examples
    --------
    >>> df = process_rdp_directory("data/RDP_classifier", domain="Bacteria")
    >>> df_phylum = process_rdp_directory("data/RDP_classifier", domain="Bacteria", taxonomic_level="phylum")
    >>> print(df.head())
    """
    # Validate domain
    if domain not in ["Bacteria", "Archaea"]:
        raise ValueError(f"domain must be 'Bacteria' or 'Archaea', got '{domain}'")

    # Validate taxonomic_level if provided
    valid_levels = ["phylum", "class", "order", "family", "genus"]
    if taxonomic_level is not None:
        if taxonomic_level not in valid_levels:
            raise ValueError(
                f"taxonomic_level must be one of {valid_levels}, got '{taxonomic_level}'"
            )

    # Get directory path
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all .tab files
    tab_files = sorted(dir_path.glob("*.tab"))

    if not tab_files:
        print(f"No .tab files found in: {directory}")
        return pd.DataFrame()

    print(f"Found {len(tab_files)} .tab files to process")
    print(f"Domain: {domain}")
    if taxonomic_level:
        print(f"Taxonomic level: {taxonomic_level}")
    else:
        print(f"Taxonomic level: all")
    print("-" * 80)

    # List to store dictionaries from each file (will convert to DataFrame later)
    all_results = []

    # Process each file
    for i, tab_file in enumerate(tab_files, 1):
        print(f"\n[{i}/{len(tab_files)}] Processing: {tab_file.name}")

        try:
            # Process the file (returns dict: {sample_id: {taxon_name: count}})
            sample_counts = process_rdp_data(str(tab_file), domain=domain)

            if not sample_counts:
                print(f"  ⚠️  No data found for {domain} domain")
                continue

            # Convert dictionary to list of records for DataFrame creation
            # This is more memory-efficient than creating intermediate DataFrames
            records = []
            all_taxa = set()

            for sample_id, taxa_dict in sample_counts.items():
                if not taxa_dict:  # Skip empty samples
                    continue
                record = {"sample_id": sample_id}
                record.update(taxa_dict)
                records.append(record)
                all_taxa.update(taxa_dict.keys())

            if not records:
                print(f"  ⚠️  No data found for {domain} domain")
                continue

            # Create DataFrame from records
            df = pd.DataFrame(records)

            # Set index
            df = df.set_index("sample_id")

            # Convert all count columns to integers (they're already ints in the dict, but
            # pandas might infer float dtype, so we explicitly convert)
            # Fill any NaN values with 0 (can occur when combining DataFrames)
            for col in df.columns:
                df[col] = df[col].fillna(0).astype("int64")

            # Print summary statistics
            num_samples = len(df)
            num_taxa = len(df.columns)
            total_reads = df.sum().sum()

            print(f"  ✓ Samples: {num_samples}")
            print(f"  ✓ Taxonomic groups: {num_taxa}")
            print(f"  ✓ Total reads: {total_reads:,.0f}")

            all_results.append(df)

        except Exception as e:
            print(f"  ✗ Error processing {tab_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_results:
        print("\n⚠️  No data was successfully processed.")
        return pd.DataFrame()

    # Combine all DataFrames
    print("\n" + "=" * 80)
    print("Combining results from all files...")
    combined_df = pd.concat(all_results, axis=0)

    # Fill NaN values with 0 (can occur when different files have different taxa)
    # and ensure all count columns are integers
    for col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna(0).astype("int64")

    # Filter by taxonomic level if specified
    if taxonomic_level:
        level_prefix = f"{taxonomic_level}__"
        matching_cols = [
            col for col in combined_df.columns if col.startswith(level_prefix)
        ]
        if not matching_cols:
            print(
                f"\n⚠️  Warning: No columns found for taxonomic level '{taxonomic_level}'"
            )
            return pd.DataFrame()
        combined_df = combined_df[matching_cols]
        print(
            f"Filtered to {taxonomic_level} level: {len(matching_cols)} taxonomic groups"
        )

    # Reorder columns alphabetically
    combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)

    # Print combined summary
    print(f"\nCombined results:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Total taxonomic groups: {len(combined_df.columns)}")
    print(f"  Total reads: {combined_df.sum().sum():,.0f}")
    print(f"  Studies: {len(all_results)}")

    # Save to CSV
    if output_file is None:
        if taxonomic_level:
            output_file = f"data/{domain.lower()}_{taxonomic_level}_counts.csv"
        else:
            output_file = f"data/{domain.lower()}_counts.csv"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_file}")
    combined_df.to_csv(output_file)
    print(f"✓ Saved successfully!")

    return combined_df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python process_rdp_data.py <filename> [domain]")
        print(
            "  Directory:   python process_rdp_data.py <directory> [domain] [taxonomic_level] [output_file]"
        )
        print("\nExamples:")
        print("  python process_rdp_data.py data/RDP_classifier/APV+20.tab Bacteria")
        print("  python process_rdp_data.py data/RDP_classifier Bacteria")
        print("  python process_rdp_data.py data/RDP_classifier Bacteria phylum")
        print(
            "  python process_rdp_data.py data/RDP_classifier Bacteria phylum data/bacteria_phylum_counts.csv"
        )
        print("\nValid taxonomic levels: phylum, class, order, family, genus")
        sys.exit(1)

    path = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else "Bacteria"
    taxonomic_level = None
    output_file = None

    # Parse remaining arguments
    # Check if 3rd arg is a taxonomic level or output file
    if len(sys.argv) > 3:
        arg3 = sys.argv[3]
        valid_levels = ["phylum", "class", "order", "family", "genus"]
        if arg3 in valid_levels:
            taxonomic_level = arg3
            output_file = sys.argv[4] if len(sys.argv) > 4 else None
        else:
            # Assume it's an output file
            output_file = arg3

    path_obj = Path(path)

    if path_obj.is_file():
        # Process single file
        study_name = path_obj.stem
        sample_counts = process_rdp_data(path, domain)

        if not sample_counts:
            print("No data found")
            sys.exit(0)

        # Convert to DataFrame for display
        records = []
        for sample_id, taxa_dict in sample_counts.items():
            if not taxa_dict:
                continue
            record = {"sample_id": sample_id}
            record.update(taxa_dict)
            records.append(record)

        if not records:
            print("No data found")
            sys.exit(0)

        df = pd.DataFrame(records).set_index("sample_id")
        df = df.reindex(sorted(df.columns), axis=1)

        # Convert all count columns to integers
        for col in df.columns:
            df[col] = df[col].fillna(0).astype("int64")

        print(f"\nProcessed {len(df)} samples")
        print(f"Found {len(df.columns)} taxonomic groups")
        print("\nFirst few rows:")
        print(df.head())
        print("\nDataFrame shape:", df.shape)
        print("\nColumn names (first 10):", list(df.columns[:10]))
    elif path_obj.is_dir():
        # Process directory
        df = process_rdp_directory(
            path,
            domain=domain,
            output_file=output_file,
            taxonomic_level=taxonomic_level,
        )
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)
