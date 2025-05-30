import sys
import os

# Add the project root directory to the Python path
# This allows pytest to find modules in 'src' and the root 'transcribe.py'
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# You can add a log to verify this runs if needed, e.g.:
# print(f"conftest.py: Added {PROJECT_ROOT} to sys.path") 