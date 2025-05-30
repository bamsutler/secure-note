import os
import sys
import subprocess
import platform
import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime
import yaml

from src import config_service
from src import transcribe

# Assuming LoggingService is available and configured as in other modules
from src.logging_service import LoggingService
log = LoggingService.get_logger(__name__)

def check_command_exists(command):
    """Check if a command exists in the system PATH or if it's a special case like BlackHole"""
    if command == 'blackhole-2ch':
        # Check if BlackHole is installed via Homebrew
        try:
            result = subprocess.run(['brew', 'list', 'blackhole-2ch'], 
                                 capture_output=True, text=True, check=False)
            return result.returncode == 0
        except Exception:
            return False
    return shutil.which(command) is not None

def load_blackhole_extension():
    """Load the BlackHole kernel extension if it's not already loaded"""
    try:
        # Check if extension is already loaded
        result = subprocess.run(['kextstat', '-b', 'com.existential.audio.BlackHoleDriver'], 
                             capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True
            
        # If not loaded, try to load it
        print("\nBlackHole kernel extension is not loaded. Attempting to load it...")
        subprocess.run(['sudo', 'kextload', '/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver/Contents/MacOS/BlackHole2ch'], 
                     check=True)
        return True
    except Exception as e:
        print(f"Error loading BlackHole kernel extension: {e}")
        print("Please try loading it manually by running:")
        print("sudo kextload /Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver/Contents/MacOS/BlackHole2ch")
        return False

def request_system_settings_access():
    """Request access to System Settings and wait for user confirmation"""
    print("\nTo configure BlackHole, we need to access System Settings.")
    print("The system will now open System Settings and request permission.")
    print("Please follow these steps:")
    print("1. When System Settings opens, click 'OK' to grant permission")
    print("2. You may need to enter your password")
    print("3. After granting permission, return here and press Enter")
    
    # Open System Settings to trigger the permission request
    subprocess.run(['open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'])
    
    # Wait for user confirmation
    input("\nPress Enter after you have granted permission in System Settings...")

def setup_blackhole():
    """Guide the user through setting up BlackHole for system audio recording"""
    if platform.system() != 'Darwin':
        print("BlackHole setup is only supported on macOS")
        return False
    
    print("\nSetting up BlackHole for system audio recording...")
    print("This will help you configure your system to record audio output.")
    
    # Check if BlackHole is installed
    blackhole_was_installed = False
    if not check_command_exists('blackhole-2ch'):
        print("BlackHole is not installed. Installing now...")
        subprocess.run(['brew', 'install', 'blackhole-2ch'], check=True)
        blackhole_was_installed = True
    
    if blackhole_was_installed:
        print("\nIMPORTANT: BlackHole was just installed and requires a system restart.")
        print("Please save any work and restart your computer before continuing.")
        print("After restarting, run this application again.")
        input("Press Enter to exit...")
        sys.exit(0)
    
    # Try to load the kernel extension if it's not already loaded
    if not load_blackhole_extension():
        print("\nWARNING: Could not load BlackHole kernel extension.")
        print("Some features may not work correctly.")
        print("You may need to restart your computer to load the extension.")
    
    # Open Audio MIDI Setup
    print("\nOpening Audio MIDI Setup...")
    print("Please follow these steps:")
    print("1. In Audio MIDI Setup, click the '+' button in the bottom left")
    print("2. Select 'Create Multi-Output Device'")
    print("3. In the right panel, check both 'BlackHole 2ch' and your main output device")
    print("4. Close Audio MIDI Setup when done")
    
    # Open Audio MIDI Setup
    subprocess.run(['open', '-a', 'Audio MIDI Setup'])
    
    # Wait for user confirmation
    input("\nPress Enter after you have configured the Multi-Output Device...")
    
    print("\nBlackHole setup complete!")
    print("Your system is now configured to record audio output.")
    
    return True

def install_ollama():
    """Install Ollama if not present and set up the required model"""
    ollama_was_installed = False
    
    if not check_command_exists('ollama'):
        print("Installing Ollama...")
        # Use /bin/sh for broader compatibility
        install_command_base = ['sudo', '/bin/bash', '-c']
        install_script_curl = '$(curl -fsSL https://ollama.com/install.sh)'
        
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            try:
                subprocess.run(install_command_base + [install_script_curl], check=True)
            except FileNotFoundError:
                # Fallback if /bin/sh is not found, though highly unlikely
                print("Error: /bin/sh not found. Trying with 'sh' directly.")
                subprocess.run(['sh', '-c', install_script_curl], check=True) # Assumes sh is in PATH
        else:
            print("Warning: Ollama installation not supported on this platform via this script.")
            return False
        ollama_was_installed = True
    
    # Install the required model
    print("\nInstalling required Ollama model...")
    try:
        # Pull the model (this will download it if not present)
        subprocess.run(['ollama', 'pull', 'llama3.2:latest'], check=True)
        print("Model installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing model: {e}")
        if ollama_was_installed:
            print("\nIMPORTANT: Ollama was just installed and may require a system restart.")
            print("Please save any work and restart your computer before continuing.")
            print("After restarting, run this application again.")
            input("Press Enter to exit...")
            sys.exit(0)
        return False
    
    return True

def install_brew_dependencies():
    """Install Homebrew dependencies on macOS"""
    if platform.system() != 'Darwin':
        return
    
    # Check if Homebrew is installed
    if not check_command_exists('brew'):
        print("Installing Homebrew...")
        # Homebrew installation command requires sudo and /bin/bash.
        # The script will prompt for the user's password.
        install_script_curl = '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'
        homebrew_install_command = ['sudo', '/bin/bash', '-c', install_script_curl]
        
        try:
            print("Attempting to install Homebrew. This requires sudo privileges and may ask for your password.")
            subprocess.run(homebrew_install_command, check=True)
            print("Homebrew installation command submitted. Please follow any on-screen prompts from the Homebrew installer.")
        except subprocess.CalledProcessError as e:
            print(f"Homebrew installation failed. The Homebrew installer exited with an error: {e}")
            print("Please try installing Homebrew manually from https://brew.sh/ and then re-run this launcher.")
            sys.exit(1)
        except FileNotFoundError:
            # This error means that 'sudo' or '/bin/bash' itself was not found.
            print("Error: Command 'sudo' or '/bin/bash' not found. These are essential for Homebrew installation.")
            print("Please ensure they are installed and accessible in your system's PATH.")
            sys.exit(1)
        except Exception as e: # Catch any other unexpected error
            print(f"An unexpected error occurred while trying to run the Homebrew installer: {e}")
            sys.exit(1)
    
    # Install portaudio (required for PyAudio)
    if not check_command_exists('portaudio'):
        print("Installing portaudio...")
        subprocess.run(['brew', 'install', 'portaudio'], check=True)
    
    # Install ffmpeg (required for audio processing)
    if not check_command_exists('ffmpeg'):
        print("Installing ffmpeg...")
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
    
    # Install BlackHole
    print("Installing BlackHole audio driver...")
    subprocess.run(['brew', 'install', 'blackhole-2ch'], check=True)
    
    # Set up BlackHole
    setup_blackhole()

def uninstall_ollama():
    """Uninstall Ollama"""
    if check_command_exists('ollama'):
        print("Uninstalling Ollama...")
        try:
            # Stop any running Ollama processes
            if platform.system() == 'Darwin':
                subprocess.run(['pkill', 'ollama'], check=False)
            elif platform.system() == 'Linux':
                subprocess.run(['systemctl', 'stop', 'ollama'], check=False)
            
            # Remove Ollama binary and data
            ollama_path = shutil.which('ollama')
            if ollama_path:
                os.remove(ollama_path)
            
            # Remove Ollama data directory
            data_dir = Path.home() / '.ollama'
            if data_dir.exists():
                shutil.rmtree(data_dir)
            
            print("Ollama uninstalled successfully")
        except Exception as e:
            print(f"Error uninstalling Ollama: {e}")

def uninstall_brew_dependencies():
    """Uninstall Homebrew dependencies on macOS"""
    if platform.system() != 'Darwin':
        return
    
    if check_command_exists('brew'):
        print("Uninstalling portaudio...")
        try:
            subprocess.run(['brew', 'uninstall', 'portaudio'], check=True)
            print("portaudio uninstalled successfully")
        except Exception as e:
            print(f"Error uninstalling portaudio: {e}")
        
        print("Uninstalling ffmpeg...")
        try:
            subprocess.run(['brew', 'uninstall', 'ffmpeg'], check=True)
            print("ffmpeg uninstalled successfully")
        except Exception as e:
            print(f"Error uninstalling ffmpeg: {e}")
        
        print("Uninstalling BlackHole...")
        try:
            subprocess.run(['brew', 'uninstall', 'blackhole-2ch'], check=True)
            print("BlackHole uninstalled successfully")
        except Exception as e:
            print(f"Error uninstalling BlackHole: {e}")

def uninstall_application():
    """Uninstall the application files"""
    print("Uninstalling application...")
    
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
        try:
            # Instead of removing the executable directly, we'll create a cleanup script
            cleanup_script_path = os.path.join(app_dir, 'cleanup.sh')
            # Ensure the shebang uses /bin/sh
            cleanup_script_content = f'''#!/bin/sh
# Wait a moment for the main process to exit
sleep 2
# Remove the executable
rm -f "{sys.executable}"
# Remove this cleanup script
rm -f "$0"
'''
            with open(cleanup_script_path, 'w') as f:
                f.write(cleanup_script_content)
            # Make the cleanup script executable
            os.chmod(cleanup_script_path, 0o755)
            # Run the cleanup script in the background using /bin/sh
            try:
                subprocess.Popen(['/bin/bash', cleanup_script_path])
            except FileNotFoundError:
                print("Error: /bin/bash not found for cleanup script. Trying with 'bash' directly.")
                subprocess.Popen(['bash', cleanup_script_path])
            print(f"Cleanup script created to remove executable: {sys.executable}")
        except Exception as e:
            print(f"Error creating cleanup script: {e}")
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Remove any cached files
    cache_dir = Path.home() / '.cache' / 'secure-note'
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"Removed cache directory: {cache_dir}")
        except Exception as e:
            print(f"Error removing cache directory: {e}")

def uninstall():
    """Run the uninstallation process"""
    print("Starting uninstallation...")
    
    # Ask for confirmation
    response = input("Are you sure you want to uninstall Secure Note? (y/N): ")
    if response.lower() != 'y':
        print("Uninstallation cancelled")
        return
    
    # Uninstall components
    uninstall_ollama()
    uninstall_brew_dependencies()
    uninstall_application()
    
    print("\nUninstallation complete!")
    print("Note: Some files may need to be removed manually:")
    print("1. Any configuration files in your home directory")
    print("2. Any data files you created with the application")

def get_user_directory_choice(prompt, default_path):
    """Get directory choice from user with validation"""
    while True:
        print(f"\n{prompt}")
        print(f"Default: {default_path}")
        print("Press Enter to use default, or enter a new path:")
        user_path = input().strip()
        
        if not user_path:  # User pressed Enter
            return default_path
            
        # Convert to absolute path if relative
        abs_path = os.path.abspath(os.path.expanduser(user_path))
        
        # Check if directory exists
        if not os.path.exists(abs_path):
            try:
                os.makedirs(abs_path)
                print(f"Created directory: {abs_path}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                print("Please try again with a valid path.")
                continue
        
        # Check if directory is writable
        if not os.access(abs_path, os.W_OK):
            print(f"Directory is not writable: {abs_path}")
            print("Please choose a different location.")
            continue
            
        return abs_path

def setup_storage_directories():
    """Set up storage directories based on user input, or use existing config."""

    # Instance of ConfigurationService to access its new defaults for comparison or initial load behavior
    # This doesn't load a file yet, just sets up the service to know its default paths
    config = config_service.ConfigurationService()

    app_config_dir = Path(config.get('paths', 'app_config_dir'))
    user_specific_config_path = app_config_dir / "config.yaml"

    if user_specific_config_path.exists():
        print(f"\nExisting user-specific configuration found at: {user_specific_config_path}")
        try:
            with open(user_specific_config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            
            # Ensure essential directories from the user's config exist
            markdown_save_from_user_config = user_config.get('paths', {}).get('markdown_save')
            if markdown_save_from_user_config:
                os.makedirs(markdown_save_from_user_config, exist_ok=True)
                print(f"Ensured markdown directory from user config exists: {markdown_save_from_user_config}")
            else:
                # If not in user config, ensure the default markdown path from ConfigurationService exists
                default_md_path = config.get('paths', 'markdown_save')
                if default_md_path: os.makedirs(default_md_path, exist_ok=True)
            
            # Database and temp paths are now primarily managed by defaults in ConfigurationService
            # We just need to ensure the ~/.sec_note_app directory exists for them.
            app_config_dir.mkdir(parents=True, exist_ok=True)
            print(f"Ensured base application config directory exists: {app_config_dir}")
            return True
        except Exception as e:
            print(f"Error reading or processing existing user config at {user_specific_config_path}: {e}")
            print("Proceeding to ask for directory configuration.")

    print("\n=== Storage Directory Configuration ===")
    print("You can customize where your analyses are stored.")
    
    # New default for markdown_save to present to user
    default_markdown_dir_for_prompt = config.get('paths', 'markdown_save')
    
    analyses_dir_choice = get_user_directory_choice(
        "Where would you like to store your analysis markdown files?",
        default_markdown_dir_for_prompt
    )
    
    app_config_dir.mkdir(parents=True, exist_ok=True) # Ensure ~/.sec_note_app exists

    default_config = config.get_config()
    default_config['paths']['markdown_save'] = analyses_dir_choice
    
    with open(user_specific_config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"\nUser-specific configuration saved to: {user_specific_config_path}")
    os.makedirs(analyses_dir_choice, exist_ok=True)

    return True

def main():
    parser = argparse.ArgumentParser(description="Secure Note: Audio Transcription and Analysis")
    parser.add_argument('--uninstall', action='store_true', help="Uninstall the application and its dependencies")
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
        sys.exit(0)
    
    # Initialize ConfigurationService and log version early
    # This instance can be potentially reused if other parts of main need config
    cfg_service = config_service.ConfigurationService() 
    app_version = cfg_service.get_application_version()
    if app_version:
        log.info(f"Starting Secure Note Taker version: {app_version}")
    else:
        log.warning("Could not determine application version.")
    
    # --- Setup Section (only if not uninstalling) ---
    print("Performing initial setup checks...")
    # Install Homebrew dependencies (macOS only)
    if platform.system() == 'Darwin':
        install_brew_dependencies()
    
    # Install Ollama and required model
    if not install_ollama():
        print("\nOllama installation or model setup failed. Please check the logs.")
        print("The application may not function correctly without Ollama and its model.")
        if not input("Continue anyway? (y/n): ").lower() == 'y':
            sys.exit(1)
            
    # Setup storage directories (ensure this is done before transcribe.cli_main)
    setup_storage_directories()
    
    print("\nSetup checks complete. Launching main application...")
    # Initialize and start the main application logic from transcribe.py
    transcribe.cli_main()

if __name__ == '__main__':
    # Add freeze_support() for PyInstaller
    import multiprocessing
    multiprocessing.freeze_support()
    main() 