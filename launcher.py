import os
import sys
import subprocess
import platform
import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime

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
        if platform.system() == 'Darwin':
            # macOS installation
            subprocess.run([
                '/bin/bash', '-c',
                '$(curl -fsSL https://ollama.com/install.sh)'
            ], check=True)
        elif platform.system() == 'Linux':
            # Linux installation
            subprocess.run([
                '/bin/bash', '-c',
                '$(curl -fsSL https://ollama.com/install.sh)'
            ], check=True)
        else:
            print("Warning: Ollama installation not supported on this platform")
            return False
        ollama_was_installed = True
    
    # Install the required model
    print("\nInstalling required Ollama model...")
    try:
        # Pull the model (this will download it if not present)
        subprocess.run(['ollama', 'pull', 'meta-llama/Llama-3-8B-Instruct'], check=True)
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
        subprocess.run([
            '/bin/bash', '-c',
            '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'
        ], check=True)
    
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
            cleanup_script = os.path.join(app_dir, 'cleanup.sh')
            with open(cleanup_script, 'w') as f:
                f.write(f'''#!/bin/bash
# Wait a moment for the main process to exit
sleep 2
# Remove the executable
rm -f "{sys.executable}"
# Remove this cleanup script
rm -f "$0"
''')
            # Make the cleanup script executable
            os.chmod(cleanup_script, 0o755)
            # Run the cleanup script in the background
            subprocess.Popen(['/bin/bash', cleanup_script])
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

def main():
    parser = argparse.ArgumentParser(description='Secure Note Application')
    parser.add_argument('--uninstall', action='store_true', help='Uninstall the application')
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
        return

    print("Checking and installing external dependencies...")
    
    # Install Homebrew dependencies (macOS only)
    install_brew_dependencies()
    
    # Install Ollama and its model
    if not install_ollama():
        print("Warning: Ollama installation failed or is not supported")
        print("Some features may not work correctly")
    
    print("\nStarting application...")
    
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Import and run the command-line interface
    sys.path.insert(0, app_dir)
    import transcribe
    
    # Initialize DB using the function from core_processing
    transcribe.core_processing.init_db(db_name=transcribe.DB_NAME)
    
    # Attempt to load the Whisper model once at startup
    try:
        transcribe.core_processing.load_global_whisper_model(transcribe.DEFAULT_WHISPER_MODEL)
    except Exception as e:
        transcribe.log.critical(f"Fatal: Could not load the global Whisper model on startup: {e}")
        transcribe.log.critical("The application cannot continue without a functional Whisper model.")
        transcribe.log.critical("Please check your Whisper installation, model files, and system compatibility (e.g., ffmpeg).")
        sys.exit(1)

    transcribe.log.info("--------------   PYNOTES --------------")
    transcribe.log.debug(f"Recordings, transcriptions, and analyses will be saved to '{transcribe.DB_NAME}'.")
    transcribe.log.debug(f"Markdown files will be saved to '{transcribe.MARKDOWN_SAVE_PATH}'.")
    transcribe.log.debug(f"Using Ollama model for analysis: '{transcribe.OLLAMA_MODEL_NAME}'")
    whisper_model_choice_default = transcribe.DEFAULT_WHISPER_MODEL 
    selected_device_id_session = None

    transcribe.log.info("\n--- Select Audio Device for Microphone ---")
    current_input_devices = transcribe.list_audio_devices()
    if not current_input_devices: 
        transcribe.log.warning("\nNo suitable input devices were found. Cannot start new recording.")
    
    try: 
        device_choice_input = input("\nEnter the PyAudio INDEX of the audio device, 'd' for default, or 'q' to exit: ").strip().lower()
    except ValueError: 
        transcribe.log.warning("Invalid input. Enter a number (PyAudio Index), 'd', or 'b'.")
    except KeyboardInterrupt:
        transcribe.log.info("\nDevice selection interrupted. Returning to main menu.")
    except EOFError:
        transcribe.log.warning("\nEOF received during device selection. Returning to main menu.")
    except Exception as e: 
        transcribe.log.exception(f"An unexpected error during device selection: {e}")
    if device_choice_input == 'd': 
        selected_device_id_session = None # This will use default input for mic
        transcribe.log.info("Default input device will be used for the microphone.")
    elif device_choice_input == 'q':
        transcribe.log.info("Exiting application.")
        sys.exit(0)
    elif device_choice_input.isdigit():
        potential_id = int(device_choice_input)
        if any(dev['id'] == potential_id for dev in current_input_devices):
            selected_device_id_session = potential_id
    else:
        transcribe.log.warning("Invalid input. Enter a number (PyAudio Index), 'd', or 'b'.")

    while True:
        transcribe.log.info("\n--- Main Menu ---")
        action_choice = ""
        try:
            action_choice = input("Choose an action: (1) Re-analyze last recording, (2) Start new recording session, (q) Quit: ").strip().lower()
        except KeyboardInterrupt:
            transcribe.log.info("\nExiting application due to KeyboardInterrupt.")
            sys.exit(0)
        except EOFError:
            transcribe.log.info("\nExiting application due to EOF.")
            sys.exit(0)

        if action_choice == '1':
            last_record = transcribe.core_processing.get_last_transcription_for_reanalysis(db_name=transcribe.DB_NAME)
            if last_record and last_record['transcription']:
                transcribe.log.info(f"\nRe-analyzing transcription for recording ID: {last_record['id']}")
                transcribe.log.info(f"Original Transcription:\n'''{last_record['transcription']}'''")
                analysis_results = transcribe.core_processing.analyze_transcription_with_ollama(
                    last_record['transcription'],
                    ollama_api_url=transcribe.OLLAMA_API_URL,
                    ollama_model_name=transcribe.OLLAMA_MODEL_NAME
                )
                is_reanalysis_successful = (analysis_results and 
                                          analysis_results.get("full_markdown_response") not in transcribe.FAILED_ANALYSIS_SUMMARIES and 
                                          analysis_results.get("full_markdown_response", "").strip())
                if is_reanalysis_successful:
                    full_markdown_response = analysis_results.get('full_markdown_response', '')
                    title_from_reanalysis = analysis_results.get('title', 'Re-analysis')
                    current_time_reanalysis_obj = datetime.now()
                    transcribe.core_processing.update_db_with_new_analysis(last_record['id'], transcribe.OLLAMA_MODEL_NAME, full_markdown_response, db_name=transcribe.DB_NAME)
                    transcribe.log.info("\n--- Parsed Re-Analysis (Ollama Markdown) ---")
                    transcribe.log.info(f"Title: {title_from_reanalysis}")
                    transcribe.log.info(f"Response Preview (first 150 chars):\n{full_markdown_response[:150]}...")
                    transcribe.core_processing.save_markdown_file(
                        title_from_reanalysis,
                        full_markdown_response,
                        current_time_reanalysis_obj,
                        transcribe.MARKDOWN_SAVE_PATH
                    )
                else:
                    transcribe.log.warning("Failed to re-analyze the transcription meaningfully. Previous analysis in DB remains unchanged.")
                    if analysis_results: 
                         transcribe.log.warning(f"Re-analysis attempt raw response (not saved to DB): {analysis_results.get('full_markdown_response')}")
            else:
                transcribe.log.info("No previous transcription found in the database to re-analyze, or last transcription was empty.")
        elif action_choice == '2':
            transcribe.log.info(f"Starting new recording session with device index: {selected_device_id_session}")
            transcribe.capture_and_transcribe(mic_device_id_selected_by_user=selected_device_id_session, whisper_model_size=whisper_model_choice_default)
        elif action_choice == 'q':
            transcribe.log.info("Exiting application.")
            sys.exit(0)
        else:
            if action_choice:
                transcribe.log.warning("Invalid choice. Please enter '1', '2', or 'q'.")

if __name__ == '__main__':
    main() 