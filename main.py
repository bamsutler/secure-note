import sys
import os

# Add the project root to the Python path to allow importing from src
# This ensures that 'import src.module' works correctly
# when main.py is executed directly from the root directory.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.transcribe import cli_main

if __name__ == "__main__":
    # The src.transcribe module and its imported services handle their own
    # initialization (config loading, DB setup via StorageService, etc.).
    # The cli_main function within src.transcribe handles Whisper model loading.
    cli_main() 