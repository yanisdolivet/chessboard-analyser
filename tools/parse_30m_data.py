#!/usr/bin/env python3
"""
Parse 30M_data.csv and convert to standard format:
FEN Nothing/Check/Checkmate

Assumes the CSV format is: FEN,state_number
Where state_number is:
  0 = Nothing
  1 = Check
  2 = Checkmate
"""

import sys
import argparse
from collections import defaultdict

# State number to name mapping
STATE_MAPPING = {"0": "Nothing", "1": "Check", "2": "Checkmate"}


def parse_csv(input_file, output_file, sample_size=None):
    """Parse CSV file and convert to standard format.

    Args:
        input_file (str): Input CSV file path
        output_file (str): Output file path
        sample_size (int, optional): Only process first N lines
    """
    print("=" * 70)
    print("30M_DATA.CSV PARSER")
    print("=" * 70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    if sample_size:
        print(f"Sample: First {sample_size:,} lines only")
    print()

    state_counts = defaultdict(int)
    total_lines = 0
    skipped_lines = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            # Check sample size limit
            if sample_size and line_num > sample_size:
                break

            line = line.strip()
            if not line:
                continue

            # Parse CSV line
            parts = line.split(",")
            if len(parts) != 2:
                skipped_lines += 1
                if skipped_lines <= 10:
                    print(
                        f"Warning: Skipping malformed line {line_num}: {line}",
                        file=sys.stderr,
                    )
                continue

            fen, state_num = parts[0].strip(), parts[1].strip()

            # Map state number to name
            if state_num not in STATE_MAPPING:
                skipped_lines += 1
                if skipped_lines <= 10:
                    print(
                        f"Warning: Unknown state '{state_num}' at line {line_num}",
                        file=sys.stderr,
                    )
                continue

            state_name = STATE_MAPPING[state_num]
            state_counts[state_name] += 1
            total_lines += 1

            # Write transformed line
            outfile.write(f"{fen} {state_name}\n")

            # Progress indicator
            if line_num % 100000 == 0:
                print(
                    f"Processed {line_num:,} lines... ({total_lines:,} valid)", end="\r"
                )

    print(f"\nProcessed {line_num:,} lines... ({total_lines:,} valid)")

    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Total lines processed: {line_num:,}")
    print(f"Valid positions:       {total_lines:,}")
    print(f"Skipped/invalid:       {skipped_lines:,}")

    if total_lines > 0:
        print(f"\nGame state distribution:")
        for state in ["Nothing", "Check", "Checkmate"]:
            count = state_counts[state]
            percentage = (count / total_lines * 100) if total_lines > 0 else 0
            print(f"  {state:<12} {count:,} ({percentage:.1f}%)")

    print(f"\nOutput saved to: {output_file}")
    print(f"{'='*70}")


def auto_detect_mapping(input_file, sample_lines=1000):
    """Auto-detect the state number mapping by sampling the file.

    Args:
        input_file (str): Input file path
        sample_lines (int): Number of lines to sample
    """
    print(f"Auto-detecting state mapping from first {sample_lines} lines...")

    state_numbers = defaultdict(int)

    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break

            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) == 2:
                state_num = parts[1].strip()
                state_numbers[state_num] += 1

    print(f"\nFound state numbers:")
    for state_num, count in sorted(state_numbers.items()):
        print(f"  {state_num}: {count} occurrences")

    print(f"\nAssumed mapping:")
    for num, name in STATE_MAPPING.items():
        print(f"  {num} → {name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Parse 30M_data.csv and convert to standard format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse entire file
  ./parse_30m_data.py data/30M_data.csv data/30m_parsed.txt
  
  # Parse only first 100,000 lines (for testing)
  ./parse_30m_data.py data/30M_data.csv data/sample.txt --sample 100000
  
  # Auto-detect state mapping
  ./parse_30m_data.py data/30M_data.csv output.txt --detect

State Mapping:
  0 → Nothing
  1 → Check
  2 → Checkmate
        """,
    )

    parser.add_argument("input_file", help="Input CSV file (FEN,state_number)")
    parser.add_argument("output_file", help="Output file (FEN game_state)")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N lines (for testing)",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Auto-detect state number mapping before parsing",
    )

    args = parser.parse_args()

    try:
        # Auto-detect mapping if requested
        if args.detect:
            auto_detect_mapping(args.input_file)
            print("Continue with parsing? (y/n): ", end="")
            if input().strip().lower() != "y":
                print("Cancelled.")
                return

        # Parse the file
        parse_csv(args.input_file, args.output_file, args.sample)

        print("\n✓ Conversion complete!")

    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
