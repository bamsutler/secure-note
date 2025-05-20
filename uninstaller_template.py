import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def check_command_exists(command):
    """Check if a command exists in the system PATH"""
    return shutil.which(command) is not None

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
            # Find and remove the main application
            main_app = os.path.join(app_dir, 'secure-note')
            if os.path.exists(main_app):
                os.remove(main_app)
                print(f"Removed main application: {main_app}")
            
            # Remove this uninstaller
            os.remove(sys.executable)
            print(f"Removed uninstaller: {sys.executable}")
        except Exception as e:
            print(f"Error removing executables: {e}")
    
    # Remove any cached files
    cache_dir = Path.home() / '.cache' / 'secure-note'
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"Removed cache directory: {cache_dir}")
        except Exception as e:
            print(f"Error removing cache directory: {e}")

def main():
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
    
    print("Uninstallation complete!")
    print("Note: Some files may need to be removed manually:")
    print("1. Any configuration files in your home directory")
    print("2. Any data files you created with the application")

if __name__ == '__main__':
    main() 