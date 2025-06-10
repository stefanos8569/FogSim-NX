import os
from pathlib import Path

def clear_folder(folder_path: str) -> None:
    """
    Deletes all files in the specified folder.

    Args:
        folder_path: Path to the folder to clear.
    """
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()  # Delete the file
        print(f"Cleared all files from: {folder_path}")
    else:
        print(f"Folder does not exist or is not a directory: {folder_path}")

def check_topology_folder(folder_path: str) -> bool:
    """
    Checks if the folder contains any .graphml files.
    """
    folder = Path(folder_path)
    return any(folder.glob("*.graphml"))

if __name__ == "__main__":
    results_folder = "Results"
    topology_folder = "Topology"

    while True:
        confirmation = input("This procedure will delete all content from the 'Results' and 'Topology' folders. Are you sure? (y/n): ").strip().lower()
        if confirmation == 'y':
            if not check_topology_folder(topology_folder):
                print("Error: No .graphml files found in the 'Topology' folder. Please generate a topology first.")
                continue  # Loop back to the input prompt
            clear_folder(results_folder)
            clear_folder(topology_folder)
            break
        elif confirmation == 'n':
            print("Operation canceled.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
