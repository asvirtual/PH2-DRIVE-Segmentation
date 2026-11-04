"""Run a controlled loss-function ablation for PH2 and/or DRIVE."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Select the number of epochs to train for. 
    # Default is 50.
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of training epochs",
    )
    # Data directory. 
    # Default is /dtu/datasets1/02516.
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/dtu/datasets1/02516"),
        help="folder containing PH2_Dataset_images/ and DRIVE/",
    )
    # Select the output directory for the results.
    # Default is outputs.
    parser.add_argument(
        "--out-dir", 
        type=Path, 
        default=Path("outputs"),
        help="Path to the output directory where results will be saved",
    )
    # Number of batches to train on per epoch.
    # Default is None, which means all batches will be used.
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="limit batches per epoch for a quick smoke test",
    )
    return parser.parse_args()


def run_drive_bce(args):
    """Run train.py with the shared DRIVE/BCE training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    # Build the command to run train.py with the specified arguments
    command = [
        sys.executable,
        str(train_script),
        "--dataset",        # use the DRIVE dataset
        "drive",
        "--loss",           # use the BCE loss function
        "bce",              
        "--epochs",         # number of epochs to train for
        str(args.epochs),
        "--data-root",      # path to the data root directory
        str(args.data_root),
        "--out-dir",        # path to the output directory
        str(args.out_dir),
    ]
    
    # If max_batches is specified, we add it to the command
    if args.max_batches is not None:
        command.extend(["--max-batches", str(args.max_batches)])

    print("Running DRIVE with BCE:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # DRIVE dataset with BCE loss
    run_drive_bce(args)


if __name__ == "__main__":
    main()