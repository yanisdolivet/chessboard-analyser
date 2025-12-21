#!/usr/bin/env python3
"""
Balance chess dataset by creating a new dataset with specified distribution.
Combines multiple source files and balances the game state distribution.
"""

import sys
import random
import argparse
from collections import defaultdict


def load_positions(file_paths):
    """Load and categorize positions from multiple files.

    Args:
        file_paths (list): List of file paths to load

    Returns:
        dict: Dictionary with keys 'Nothing', 'Check', 'Checkmate'
              containing lists of position strings
    """
    positions = {"Nothing": [], "Check": [], "Checkmate": []}

    total_lines = 0
    skipped_lines = 0

    for file_path in file_paths:
        print(f"Loading {file_path}...", end=" ")
        file_lines = 0

        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    total_lines += 1
                    file_lines += 1

                    # Extract game state (last word)
                    parts = line.split()
                    if len(parts) < 7:  # FEN has at least 6 parts + state
                        skipped_lines += 1
                        continue

                    game_state = parts[-1]

                    if game_state in positions:
                        positions[game_state].append(line)
                    else:
                        skipped_lines += 1

                print(f"✓ ({file_lines:,} lines)")

        except FileNotFoundError:
            print(f"✗ (not found)")
            continue
        except Exception as e:
            print(f"✗ (error: {e})")
            continue

    print(f"\n{'='*70}")
    print("Loading Summary:")
    print(f"{'='*70}")
    print(f"Total lines read:     {total_lines:,}")
    print(f"Skipped/invalid:      {skipped_lines:,}")
    print(f"\nPositions by category:")
    for state in ["Nothing", "Check", "Checkmate"]:
        count = len(positions[state])
        percentage = (count / total_lines * 100) if total_lines > 0 else 0
        print(f"  {state:<12} {count:,} ({percentage:.1f}%)")

    return positions


def balance_dataset(positions, target_distribution, max_size=None):
    """Balance dataset according to target distribution.

    Args:
        positions (dict): Dictionary of positions by game state
        target_distribution (dict): Target percentages for each state
        max_size (int, optional): Maximum total dataset size

    Returns:
        list: Balanced list of position strings
    """
    print(f"\n{'='*70}")
    print("Balancing Dataset:")
    print(f"{'='*70}")

    # Calculate available positions for each category
    available = {state: len(positions[state]) for state in positions}

    # If max_size not specified, calculate based on the limiting factor
    if max_size is None:
        # Find the limiting category (smallest ratio of available/target)
        ratios = {}
        for state, target_pct in target_distribution.items():
            if target_pct > 0 and available[state] > 0:
                ratios[state] = available[state] / (target_pct / 100.0)

        if not ratios:
            print("Error: No positions available for balancing!")
            return []

        # Use the smallest ratio to determine max size
        max_size = int(min(ratios.values()))
        limiting_state = min(ratios, key=ratios.get)
        print(f"Auto-calculated max size: {max_size:,} (limited by {limiting_state})")

    # Calculate target counts for each category
    target_counts = {}
    for state, target_pct in target_distribution.items():
        target_counts[state] = int(max_size * target_pct / 100.0)

    # Check if we have enough positions
    print(f"\nTarget distribution:")
    balanced_positions = []

    for state in ["Nothing", "Check", "Checkmate"]:
        target = target_counts[state]
        avail = available[state]
        target_pct = target_distribution[state]

        if target > avail:
            print(
                f"  {state:<12} {target:,} ({target_pct}%) - WARNING: Only {avail:,} available!"
            )
            actual_count = avail
        else:
            print(f"  {state:<12} {target:,} ({target_pct}%)")
            actual_count = target

        # Sample positions
        if actual_count > 0:
            sampled = random.sample(positions[state], actual_count)
            balanced_positions.extend(sampled)

    # Shuffle the final dataset
    random.shuffle(balanced_positions)

    print(f"\nFinal dataset size: {len(balanced_positions):,}")

    return balanced_positions


def save_dataset(positions, output_file):
    """Save balanced dataset to file.

    Args:
        positions (list): List of position strings
        output_file (str): Output file path
    """
    print(f"\n{'='*70}")
    print(f"Saving to {output_file}...")
    print(f"{'='*70}")

    with open(output_file, "w") as f:
        for position in positions:
            f.write(position + "\n")

    print(f"✓ Saved {len(positions):,} positions")

    # Verify the distribution
    state_counts = defaultdict(int)
    for position in positions:
        state = position.split()[-1]
        state_counts[state] += 1

    print(f"\nFinal distribution:")
    total = len(positions)
    for state in ["Nothing", "Check", "Checkmate"]:
        count = state_counts[state]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {state:<12} {count:,} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Balance chess dataset by game state distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Create balanced dataset with 40% Nothing, 40% Check, 20% Checkmate
  ./balance_dataset.py -o balanced.txt \\
      --nothing 40 --check 40 --checkmate 20 \\
      data/train_sample6.txt data/training_data.txt
  
  # Limit to 50000 positions
  ./balance_dataset.py -o balanced.txt --max-size 50000 \\
      --nothing 40 --check 40 --checkmate 20 \\
      data/train_sample6.txt data/training_data.txt
        """,
    )

    parser.add_argument("input_files", nargs="+", help="Input files to combine")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument(
        "--nothing",
        type=float,
        default=40.0,
        help="Percentage of Nothing positions (default: 40)",
    )
    parser.add_argument(
        "--check",
        type=float,
        default=40.0,
        help="Percentage of Check positions (default: 40)",
    )
    parser.add_argument(
        "--checkmate",
        type=float,
        default=20.0,
        help="Percentage of Checkmate positions (default: 20)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum dataset size (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate percentages sum to 100
    total_pct = args.nothing + args.check + args.checkmate
    if abs(total_pct - 100.0) > 0.01:
        print(f"Error: Percentages must sum to 100 (got {total_pct})", file=sys.stderr)
        sys.exit(1)

    # Set random seed
    random.seed(args.seed)

    print("=" * 70)
    print("CHESS DATASET BALANCER")
    print("=" * 70)
    print(f"Input files: {len(args.input_files)}")
    for f in args.input_files:
        print(f"  - {f}")
    print(f"\nTarget distribution:")
    print(f"  Nothing:   {args.nothing}%")
    print(f"  Check:     {args.check}%")
    print(f"  Checkmate: {args.checkmate}%")
    if args.max_size:
        print(f"\nMax size: {args.max_size:,} positions")
    print()

    # Load positions
    positions = load_positions(args.input_files)

    # Check if we have positions
    if not any(positions.values()):
        print("\nError: No valid positions found in input files!", file=sys.stderr)
        sys.exit(1)

    # Balance dataset
    target_distribution = {
        "Nothing": args.nothing,
        "Check": args.check,
        "Checkmate": args.checkmate,
    }

    balanced = balance_dataset(positions, target_distribution, args.max_size)

    if not balanced:
        print("\nError: Could not create balanced dataset!", file=sys.stderr)
        sys.exit(1)

    # Save dataset
    save_dataset(balanced, args.output)

    print(f"\n{'='*70}")
    print("✓ Dataset balancing complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
