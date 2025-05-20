import os
import subprocess
import sys
import shutil

def clean_build_dirs():
    """Clean build directories"""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    # Clean up any .spec files
    for file in os.listdir('.'):
        if file.endswith('.spec'):
            os.remove(file)

def build_executable(script_name, output_name, is_uninstaller=False):
    """Build a single executable using PyInstaller"""
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--name=' + output_name,
        '--onefile',  # Create a single executable
        '--clean',    # Clean PyInstaller cache
        '--strip',    # Strip symbols to reduce size
    ]
    
    if not is_uninstaller:
        # Add these options only for the main application
        cmd.extend([
            '--collect-all=whisper',  # Ensure all whisper models are included
            '--collect-all=torch',    # Ensure all torch dependencies are included
            '--hidden-import=uvicorn.logging',
            '--hidden-import=uvicorn.loops',
            '--hidden-import=uvicorn.loops.auto',
            '--hidden-import=uvicorn.protocols',
            '--hidden-import=uvicorn.protocols.http',
            '--hidden-import=uvicorn.protocols.http.auto',
            '--hidden-import=uvicorn.protocols.websockets',
            '--hidden-import=uvicorn.protocols.websockets.auto',
            '--hidden-import=uvicorn.lifespan',
            '--hidden-import=uvicorn.lifespan.on',
            '--hidden-import=python_multipart',  # Required for FastAPI form data
        ])
    else:
        # Add the main application path as a data file for the uninstaller
        main_app_path = os.path.join('dist', 'secure-note')
        if os.path.exists(main_app_path):
            cmd.extend([
                '--add-data', f'{main_app_path}:.'
            ])
    
    # Add the main script
    cmd.append(script_name)
    
    # Run PyInstaller
    subprocess.run(cmd, check=True)

def main():
    print("Cleaning previous build artifacts...")
    clean_build_dirs()
    
    print("Building main application...")
    build_executable('launcher.py', 'secure-note')
    
    print("Building uninstaller...")
    build_executable('uninstaller_template.py', 'uninstall-secure-note', is_uninstaller=True)
    
    print("\nBuild complete!")
    print("Two executables have been created in the 'dist' directory:")
    print("1. secure-note - The main application")
    print("2. uninstall-secure-note - The uninstaller")

if __name__ == '__main__':
    main() 