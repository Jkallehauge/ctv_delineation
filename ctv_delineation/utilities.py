import os

def get_files_with_extension(folder_name, ext):
    """Find all files in a folder with a given extension.

    Params:
        folder_name: Full path of folder to search.
        ext: Extension to search for.
    """
    return [os.path.join(folder_name, f) for f in os.listdir(folder_name) if
            os.path.isfile(os.path.join(folder_name, f)) and f.endswith(ext)]
