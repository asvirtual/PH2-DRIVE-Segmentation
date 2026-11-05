#!/bin/bash
#BSUB -q c02516
#BSUB -J ablation
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -o ./output/ablation_drive_bce1.out
#BSUB -e ./output/ablation_drive_bce1.err

cd /zhome/b4/2/233360/DL/PH2-DRIVE-Segmentation || exit 1
which python
python --version
python -u abalation.py