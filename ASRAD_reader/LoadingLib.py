from pathlib import Path

def find_all_folder_path(folder_path: str | Path) -> list[Path]:
    """
    Find all folder paths in the given folder path (including subdirectories).

    Args:
        folder_path: The root folder path to search.
    Returns:
        A list of Path objects representing all folder paths found.
    Raises:
        NotADirectoryError: If the provided path is not a directory.
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"The provided path '{folder_path}' is not a directory.")
    # Use pathlib's rglob to find all subdirectories
    folders = [p for p in folder_path.rglob('*') if p.is_dir()]
    return folders