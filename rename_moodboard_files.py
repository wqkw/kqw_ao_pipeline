import os
from pathlib import Path

def rename_files_to_numbers(directory: str):
    """Rename all files in directory to sequential numbers (1, 2, 3, etc.)"""
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"Directory {directory} does not exist")
        return

    # Get all files (not directories) and sort them
    files = sorted([f for f in dir_path.iterdir() if f.is_file()])

    if not files:
        print("No files found in directory")
        return

    # Rename files to sequential numbers
    for i, file_path in enumerate(files, start=1):
        extension = file_path.suffix
        new_name = f"{i}{extension}"
        new_path = dir_path / new_name

        print(f"Renaming {file_path.name} -> {new_name}")
        file_path.rename(new_path)

    print(f"\nRenamed {len(files)} files")

if __name__ == "__main__":
    rename_files_to_numbers("data/ref_moodboard4")
