import os
import subprocess
import sys
import shutil
import zipfile

VERSION_FILE = "VERSION"

def get_version():
    """Reads version from VERSION_FILE."""
    with open(VERSION_FILE, "r") as f:
        return f.read().strip()

def increment_version(version_str, level="patch"):
    """Increments the specified part of a version string.
    level can be 'patch', 'minor', or 'major'.
    e.g., 0.1.0 -> 0.1.1 for patch
    e.g., 0.1.0 -> 0.2.0 for minor
    e.g., 0.1.0 -> 1.0.0 for major
    """
    major, minor, patch = map(int, version_str.split('.'))
    if level == "patch":
        patch += 1
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError("Invalid increment level. Choose from 'patch', 'minor', 'major'.")
    return f"{major}.{minor}.{patch}"

def set_version(version_str):
    """Writes version to VERSION_FILE."""
    with open(VERSION_FILE, "w") as f:
        f.write(version_str)

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

def build_executable(script_name, output_name, app_version, is_uninstaller=False):
    """Build a single executable using PyInstaller"""
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--name=' + output_name,
        '--onefile',  # Create a single executable
        '--clean',    # Clean PyInstaller cache
        '--strip',    # Strip symbols to reduce size
        # Potentially embed version, though app reads from VERSION file
        # '--distpath=dist', # Default, but explicit
        # '--workpath=build', # Default, but explicit
        f'--specpath=.', # Default, but explicit
        # Example of how to pass version to app if it didn't read VERSION file:
        # f'--define \'APP_VERSION="{app_version}"\'', # This requires app changes
    ]
    
    if not is_uninstaller:
        # Add these options only for the main application
        cmd.extend([
            '--collect-all=whisper',  # Ensure all whisper models are included
            '--collect-all=torch',    # Ensure all torch dependencies are included
            '--add-data', 'prompt_templates:prompt_templates', # Include the prompt_templates folder
            '--add-data', f'{VERSION_FILE}:.' # Include the VERSION file in the root of the bundle
        ])
    else:
        # Add the main application path as a data file for the uninstaller
        main_app_path = os.path.join('dist', 'secure-note')
        if os.path.exists(main_app_path):
            cmd.extend([
                '--add-data', f'{main_app_path}:.',
                '--add-data', f'{VERSION_FILE}:.' # Include the VERSION file in the root of the bundle
            ])
    
    # Add the main script
    cmd.append(script_name)
    
    # Run PyInstaller
    subprocess.run(cmd, check=True)

def main():
    print("Cleaning previous build artifacts...")
    clean_build_dirs()
    
    current_version = get_version()
    print(f"Current version: {current_version}")

    app_version_for_build = current_version
    increment_level = None

    if "--increment-patch" in sys.argv:
        increment_level = "patch"
    elif "--increment-minor" in sys.argv:
        increment_level = "minor"
    elif "--increment-major" in sys.argv:
        increment_level = "major"

    if increment_level:
        new_version = increment_version(current_version, level=increment_level)
        set_version(new_version)
        print(f"Incremented {increment_level} version to: {new_version}")
        app_version_for_build = new_version
    else:
        print("Development build. Version not incremented.")
    
    print(f"Building with version: {app_version_for_build}")
    print("Building main application...")
    build_executable('src/launcher.py', 'secure-note', app_version_for_build)
    
    print("Building uninstaller...")
    build_executable('scripts/uninstaller_template.py', 'uninstall-secure-note', app_version_for_build, is_uninstaller=True)
    
    print("\nBuild complete!")
    print("Two executables have been created in the 'dist' directory:")
    print("1. secure-note - The main application")
    print("2. uninstall-secure-note - The uninstaller")

    # Zip the artifacts
    print(f"\nZipping artifacts for version {app_version_for_build}...")
    app_name = "secure-note"
    dist_dir = "dist"
    
    # Determine archive name based on whether version was incremented
    if increment_level is None: # This indicates a development build (no version increment flag was passed)
        archive_basename = f"{app_name}-{app_version_for_build}-test"
        print(f"Development build detected. Archive will be named: {archive_basename}.zip")
    else: # This is a release build (patch, minor, or major increment)
        archive_basename = f"{app_name}-{app_version_for_build}"
        print(f"Release build detected. Archive will be named: {archive_basename}.zip")

    # archive_dir_path is the name of the folder *inside* the zip file.
    # It should match the archive_basename for consistency.
    internal_folder_name = archive_basename 
    zip_file_path = os.path.join(dist_dir, f"{archive_basename}.zip")

    # Ensure the executables exist before trying to zip them
    main_app_exe_path = os.path.join(dist_dir, app_name)
    uninstaller_exe_path = os.path.join(dist_dir, f"uninstall-{app_name}")

    if not os.path.exists(main_app_exe_path):
        print(f"ERROR: Main application executable not found at {main_app_exe_path}. Skipping zipping.")
        return
    if not os.path.exists(uninstaller_exe_path):
        print(f"ERROR: Uninstaller executable not found at {uninstaller_exe_path}. Skipping zipping.")
        return

    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add main executable to the folder inside the zip
            zf.write(main_app_exe_path, os.path.join(internal_folder_name, os.path.basename(main_app_exe_path)))
            # Add uninstaller to the folder inside the zip
            zf.write(uninstaller_exe_path, os.path.join(internal_folder_name, os.path.basename(uninstaller_exe_path)))
            # Add the VERSION file to the folder inside the zip
            if os.path.exists(VERSION_FILE):
                 zf.write(VERSION_FILE, os.path.join(internal_folder_name, VERSION_FILE))
            else:
                print(f"Warning: {VERSION_FILE} not found. It will not be included in the zip.")

        print(f"Successfully created zip archive: {zip_file_path}")
        print(f"The archive contains a folder '{internal_folder_name}' with the executables and VERSION file.")

    except Exception as e:
        print(f"Error creating zip archive: {e}")

if __name__ == '__main__':
    main() 