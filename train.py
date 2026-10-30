from dataloaders import (
    create_ph2_loaders,
    create_drive_loaders,
    create_drive_official_test_loader
)

# PH2
ph2_loaders, ph2_datasets = create_ph2_loaders(
    "/dtu/datasets1/02516/PH2_Dataset_images"
)

ph2_train, ph2_val, ph2_test = ph2_loaders

# DRIVE
drive_loaders, drive_datasets = create_drive_loaders(
    "/dtu/datasets1/02516/DRIVE"
)

drive_train, drive_val, drive_test = drive_loaders