"""Run a controlled loss-function ablation for PH2 and/or DRIVE."""

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = "/zhome/b4/2/233360/project2/bin/python"


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



# -----------------------------------------------------------
# BCE LOSS FUNCTIONS
# -----------------------------------------------------------
def run_drive_bce(args):
    """Run train.py with the shared DRIVE/BCE training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    model = "unet"
    # Build the command to run train.py with the specified arguments
    command = [
        PYTHON,
        str(train_script),
        "--dataset",        # use the DRIVE dataset
        "drive",
        "--model",           # use the UNet model
        model,
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
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    print("\n\n")
    subprocess.run(command, check=True)          # With unet model

    model = "encdec"
    command[5] = model  
    print("\n\n")
    subprocess.run(command, check=True)          # With encdec model


def run_ph2_bce(args):
    """Run train.py with the shared PH2/BCE training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    model = "unet"
    # Build the command to run train.py with the specified arguments
    command = [
        PYTHON,
        str(train_script),
        "--dataset",        # use the PH2 dataset
        "ph2",
        "--model",           # use the UNet model
        model,
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

    print("Running PH2 with BCE:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    print("\n\n")
    subprocess.run(command, check=True)          # With unet model

    model = "encdec"
    command[5] = model  
    print("\n\n")
    subprocess.run(command, check=True)          # With encdec model




# -----------------------------------------------------------
# FOCAL LOSS FUNCTIONS
# -----------------------------------------------------------
def run_drive_focal_loss(args):
    """Run train.py with the shared DRIVE/Focal Loss training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    model = "unet"
    # Build the command to run train.py with the specified arguments
    command = [
        PYTHON,
        str(train_script),
        "--dataset",        # use the DRIVE dataset
        "drive",
        "--model",           # use the UNet model
        model,
        "--loss",           # use the Focal Loss function
        "focal",              
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

    print("Running DRIVE with Focal Loss:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    print("\n\n")
    subprocess.run(command, check=True)          # With unet model

    model = "encdec"
    command[5] = model  
    print("\n\n")
    subprocess.run(command, check=True)          # With encdec model


def run_ph2_focal_loss(args):
    """Run train.py with the shared PH2/Focal Loss training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    model = "unet"
    # Build the command to run train.py with the specified arguments
    command = [
        PYTHON,
        str(train_script),
        "--dataset",        # use the PH2 dataset
        "ph2",
        "--model",           # use the UNet model
        model,
        "--loss",           # use the Focal Loss function
        "focal",              
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

    print("Running PH2 with Focal Loss:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    print("\n\n")
    subprocess.run(command, check=True)          # With unet model

    model = "encdec"
    command[5] = model  
    print("\n\n")
    subprocess.run(command, check=True)          # With encdec model




# -----------------------------------------------------------
# WEIGHTED LOSS FUNCTIONS
# -----------------------------------------------------------

def run_drive_weighted_loss(args):
    """Run train.py with the shared DRIVE/Weighted Loss training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    model = "unet"
    # Build the command to run train.py with the specified arguments
    command = [
        PYTHON,
        str(train_script),
        "--dataset",        # use the DRIVE dataset
        "drive",
        "--model",           # use the UNet model
        model,
        "--loss",           # use the Weighted Loss function
        "weighted_bce",              
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

    print("Running DRIVE with Weighted Cross-Entropy Loss:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    print("\n\n")
    subprocess.run(command, check=True)          # With unet model

    model = "encdec"
    command[5] = model  
    print("\n\n")
    subprocess.run(command, check=True)          # With encdec model



def run_ph2_weighted_loss(args):
    """Run train.py with the shared PH2/Weighted Loss training implementation."""
    train_script = Path(__file__).with_name("train.py")
    
    model = "unet"
    # Build the command to run train.py with the specified arguments
    command = [
        PYTHON,
        str(train_script),
        "--dataset",        # use the PH2 dataset
        "ph2",
        "--model",           # use the UNet model
        model,
        "--loss",           # use the Weighted Loss function
        "weighted_bce",              
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

    print("Running PH2 with Weighted Cross-Entropy Loss:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
        
    print("\n\n")
    subprocess.run(command, check=True)          # With unet model

    model = "encdec"
    command[5] = model  
    print("\n\n")
    subprocess.run(command, check=True)          # With encdec model





def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Binary Cross-Entropy (BCE) Loss
    run_drive_bce(args)                      # DRIVE
    #run_ph2_bce(args)                         # PH2
        
    # DRIVE dataset with Focal Loss
    #run_drive_focal_loss(args)                # DRIVE
    #run_ph2_focal_loss(args)                  # PH2

    # DRIVE dataset with Weighted Cross-Entropy Loss
    #run_drive_weighted_loss(args)             # DRIVE
    #run_ph2_weighted_loss(args)               # PH2




if __name__ == "__main__":
    main()