#!/usr/bin/env python-real

import argparse
from pathlib import Path

from __CrownSegmentation.dataset import create_dataset_csv

def main():
    parser = argparse.ArgumentParser(
        description='Create dataset CSV file for Crown Segmentation from dental data directory'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Root directory containing dental data (e.g., /path/to/teeth_data)'
    )
    parser.add_argument(
        'output_csv',
        type=str,
        help='Output path for the dataset CSV file'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if all files in the CSV exist before saving'
    )

    args = parser.parse_args()

    try:
        create_dataset_csv(
            root_dir=args.input_dir,
            output_csv=args.output_csv
        )

        if args.check:
            # CSVファイル内の全ファイルの存在確認
            import pandas as pd
            df = pd.read_csv(args.output_csv)
            missing_files = []
            for _, row in df.iterrows():
                if not Path(row['input_path']).exists():
                    missing_files.append(row['input_path'])
            
            if missing_files:
                print("\nWarning: Following files are missing:")
                for f in missing_files:
                    print(f"  - {f}")
            else:
                print("\nAll files in the CSV exist.")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main()) 