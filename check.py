from pathlib import Path

root = Path("/dtu/datasets1/02516/DRIVE")

folders = [
    "training/images",
    "training/1st_manual",
    "training/mask",
    "test/images",
    "test/mask"
]

for folder in folders:
    path = root / folder

    print(f"\n{'=' * 50}")
    print(folder)

    if not path.exists():
        print("DIRECTORY DOES NOT EXIST!")
        continue

    files = sorted([f for f in path.iterdir() if f.is_file()])

    print("Number of files:", len(files))
    print("First 5 files:")

    for file in files[:5]:
        print(" ", file.name)