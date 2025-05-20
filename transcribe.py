import pyaudio # Replaced sounddevice
import numpy as np
import sys
import threading
from datetime import datetime
import io
import wave
import json # For payload to Ollama if used directly here, though mostly in core_processing
import logging
import gc # For garbage collection if ever explicitly needed
import os # For path operations

# Import shared logic from core_processing
import core_processing 

# --- Logger Setup ---
# The logger is now primarily configured in core_processing.
# We can get it here if needed, or just let core_processing handle logging for its functions.
log = logging.getLogger("transcribe_cli") # Use a specific logger for this CLI application
# If you want this CLI to also use RichHandler similar to core_processing:
if not log.handlers:
    log.setLevel(logging.INFO)
    from rich.logging import RichHandler
    handler = RichHandler(rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    log.addHandler(handler)

console = core_processing.console # Can reuse console from core_processing

# --- Use constants from core_processing ---
DB_NAME = core_processing.DB_NAME_DEFAULT
MARKDOWN_SAVE_PATH = core_processing.MARKDOWN_SAVE_PATH_DEFAULT
OLLAMA_API_URL = core_processing.OLLAMA_API_URL_DEFAULT
OLLAMA_MODEL_NAME = core_processing.OLLAMA_MODEL_NAME_DEFAULT
FAILED_ANALYSIS_SUMMARIES = core_processing.FAILED_ANALYSIS_SUMMARIES
DEFAULT_WHISPER_MODEL = core_processing.DEFAULT_WHISPER_MODEL

# --- PyAudio Specific Constants ---
FRAMES_PER_BUFFER = 1024
# PyAudio format - we want float32 to match previous sounddevice behavior
# and for compatibility with Whisper/NumPy processing downstream.
PYAUDIO_FORMAT = pyaudio.paFloat32 
CHANNELS = 1 # Each stream will be mono

# Global list to store audio chunks and an event to signal stopping
# These are specific to the live recording functionality of this CLI tool
mic_audio_chunks = [] # Renamed from audio_chunks
system_audio_chunks = [] # New for system audio
stop_event = threading.Event()

def pyaudio_mic_callback(in_data, frame_count, time_info, status_flags):
    """Callback for PyAudio stream for the microphone."""
    if status_flags:
        log.warning(f"PyAudio Mic Callback status flags: {status_flags}")
    try:
        numpy_data = np.frombuffer(in_data, dtype=np.float32)
        if not stop_event.is_set():
            mic_audio_chunks.append(numpy_data)
    except Exception as e:
        log.error(f"Error in PyAudio Mic callback: {e}")
    return (None, pyaudio.paContinue)

def pyaudio_system_callback(in_data, frame_count, time_info, status_flags):
    """Callback for PyAudio stream for system audio."""
    if status_flags:
        log.warning(f"PyAudio System Callback status flags: {status_flags}")
    try:
        numpy_data = np.frombuffer(in_data, dtype=np.float32)
        if not stop_event.is_set():
            system_audio_chunks.append(numpy_data)
    except Exception as e:
        log.error(f"Error in PyAudio System callback: {e}")
    return (None, pyaudio.paContinue)

def input_listener():
    """Listens for the 'stop' command from the user."""
    global stop_event
    log.info("Type 'stop' and press Enter to finish recording.")
    while not stop_event.is_set():
        try:
            command = input().strip().lower()
            if command == "stop":
                log.info("Stop command received.")
                stop_event.set()
                break
        except EOFError:
            log.warning("EOF received, stopping listener.")
            stop_event.set()
            break
        except KeyboardInterrupt:
            log.info("Keyboard interrupt in listener, stopping.")
            stop_event.set()
            break

def list_audio_devices():
    """Lists available audio input devices using PyAudio."""
    log.info("Available audio input devices (via PyAudio):")
    p = pyaudio.PyAudio()
    output_dev_info = p.get_default_output_device_info()
    log.info(f"Default output device: {output_dev_info.get('name')} (Index: {output_dev_info.get('index')})")
    input_devices = []
    try:
        num_devices = p.get_device_count()
        if num_devices == 0:
            log.warning("No audio devices found by PyAudio.")
            return input_devices

        for i in range(num_devices):
            dev_info = None
            try:
                dev_info = p.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels', 0) > 0:
                    host_api_info = p.get_host_api_info_by_index(dev_info.get('hostApi'))
                    host_api_name = host_api_info.get('name', 'N/A')
                    log.info(f"  ID {i}: {dev_info.get('name')} (Input Channels: {dev_info.get('maxInputChannels')}, Host API: {host_api_name}, Index: {dev_info.get('index')})")
                    input_devices.append({'id': i, 'name': dev_info.get('name')})
            except Exception as e_dev:
                name = dev_info.get('name') if dev_info else f"Device {i}"
                log.warning(f"  Could not query full details for {name}: {e_dev}")
                # Still try to add if it seems like an input device based on partial info
                if dev_info and dev_info.get('maxInputChannels', 0) > 0:
                    input_devices.append({'id': i, 'name': dev_info.get('name', f'Device {i}') + " (details partially unavailable)"})
    except Exception as e:
        log.error(f"Error listing PyAudio devices: {e}")
    finally:
        p.terminate()

    if not input_devices:
        log.warning("No input devices found. Ensure your microphone or loopback device is enabled and recognized.")
    return input_devices

def capture_and_transcribe(mic_device_id_selected_by_user=None, samplerate=16000, whisper_model_size=DEFAULT_WHISPER_MODEL):
    """
    Captures audio continuously from a selected microphone device AND the default system input device (assumed to be a loopback).
    Mixes the audio, then transcribes using core_processing.
    """
    global mic_audio_chunks, system_audio_chunks, stop_event
    mic_audio_chunks = []
    system_audio_chunks = []
    stop_event.clear()

    p = None
    mic_stream = None
    system_stream = None
    
    actual_mic_device_index = None
    actual_system_audio_device_index = None
    system_audio_device_name = "N/A"

    try:
        p = pyaudio.PyAudio()

        # 1. Determine Mic Device Index
        if mic_device_id_selected_by_user is None:
            try:
                default_mic_dev_info = p.get_default_input_device_info()
                actual_mic_device_index = default_mic_dev_info['index']
                log.debug(f"Using default PyAudio input device for microphone: {default_mic_dev_info['name']} (Index: {actual_mic_device_index})")
            except IOError:
                log.error("No default input device found for microphone, and none was selected. Please select a microphone from the menu.")
                if p: p.terminate()
                return
        else:
            actual_mic_device_index = mic_device_id_selected_by_user
            try:
                mic_dev_info = p.get_device_info_by_index(actual_mic_device_index)
                log.debug(f"Using selected PyAudio device for microphone: {mic_dev_info['name']} (Index: {actual_mic_device_index})")
            except IOError:
                log.error(f"Invalid PyAudio device index {actual_mic_device_index} for microphone.")
                if p: p.terminate()
                return
        
        # 2. Determine System Audio Device Index (Default Input, assumed to be a loopback)
        try:
            default_input_dev_info = p.get_default_input_device_info()
            actual_system_audio_device_index = default_input_dev_info['index']
            system_audio_device_name = default_input_dev_info['name']
            log.debug(f"Attempting to capture system audio from default input device: {system_audio_device_name} (Index: {actual_system_audio_device_index}).")
        except IOError:
            log.warning("No default input device found by PyAudio. Will not be able to capture system audio. Proceeding with microphone input only.")
            actual_system_audio_device_index = None # Ensure it's None

    except Exception as e:
        log.error(f"Error initializing PyAudio or device settings: {e}")
        if p: p.terminate()
        return

    try:
        core_processing.load_global_whisper_model(whisper_model_size)
    except Exception as e:
        log.error(f"Could not proceed with recording due to Whisper model loading error: {e}")
        if p: p.terminate()
        return

    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()
    log.debug(f"\nRecording at {samplerate} Hz (Channels: {CHANNELS}, Format: PyAudio Float32).")
    
    try:
        # Open Mic Stream
        log.debug(f"Opening microphone stream on device index {actual_mic_device_index}...")
        mic_stream = p.open(format=PYAUDIO_FORMAT,
                            channels=CHANNELS,
                            rate=samplerate,
                            input=True,
                            input_device_index=actual_mic_device_index,
                            frames_per_buffer=FRAMES_PER_BUFFER,
                            stream_callback=pyaudio_mic_callback)
        mic_stream.start_stream()
        log.debug("Microphone stream started.")

        # Open System Audio Stream (if different from mic and available)
        if actual_system_audio_device_index is not None:
            if actual_system_audio_device_index != actual_mic_device_index:
                log.debug(f"Opening system audio stream on device index {actual_system_audio_device_index} ({system_audio_device_name})...")
                try:
                    system_stream = p.open(format=PYAUDIO_FORMAT,
                                           channels=CHANNELS,
                                           rate=samplerate,
                                           input=True,
                                           input_device_index=actual_system_audio_device_index,
                                           frames_per_buffer=FRAMES_PER_BUFFER,
                                           stream_callback=pyaudio_system_callback)
                    system_stream.start_stream()
                    log.debug("System audio stream started.")
                except Exception as e_sys:
                    log.error(f"Could not start system audio stream on {system_audio_device_name} (Index {actual_system_audio_device_index}): {e_sys}")
                    system_stream = None # Ensure it's None if opening failed
            else:
                log.warning(f"Microphone device and system audio (default input) device are the same (Index {actual_mic_device_index}). System audio will be a duplicate of mic audio.")
                # No second stream needed; mic_audio_chunks will be duplicated later.
        
        log.info("Recording... Type 'stop' and press Enter to finish.")
        
        while not stop_event.is_set():
            active_mic = mic_stream and mic_stream.is_active()
            active_system = system_stream and system_stream.is_active()

            if not active_mic and (actual_system_audio_device_index is None or actual_system_audio_device_index == actual_mic_device_index or not active_system) :
                log.warning("Microphone stream stopped or no streams were active.")
                stop_event.set()
                break
            if actual_system_audio_device_index is not None and actual_system_audio_device_index != actual_mic_device_index and not active_system and system_stream:
                 log.warning("System audio stream stopped.")
                 # Potentially allow continuing with mic only, or stop all. For now, stop all.
                 stop_event.set()
                 break
            
            # Check if any stream was intended but is not active
            intended_mic = mic_stream is not None
            intended_system = system_stream is not None

            if (intended_mic and not active_mic) and (intended_system and not active_system):
                log.warning("Both microphone and system audio streams stopped unexpectedly.")
                stop_event.set()
                break
            if intended_mic and not active_mic and not intended_system: # only mic was intended
                 log.warning("Microphone stream stopped unexpectedly.")
                 stop_event.set()
                 break
            # If only system was intended and stopped (less likely with current logic but for completeness)
            if intended_system and not active_system and not intended_mic :
                 log.warning("System audio stream stopped unexpectedly.")
                 stop_event.set()
                 break

            stop_event.wait(timeout=0.1)

    except Exception as e:
        log.error(f"An error occurred during PyAudio recording: {e}")
        stop_event.set() # Ensure callback stops appending and listener exits
    finally:
        stop_event.set() 
        if listener_thread.is_alive():
            log.info("Waiting for input listener to complete...")
            listener_thread.join(timeout=1.5) 
            if listener_thread.is_alive():
                log.warning("Input listener did not stop cleanly.")

        if mic_stream is not None:
            try:
                if mic_stream.is_active():
                    mic_stream.stop_stream()
                mic_stream.close()
                log.info("Microphone stream stopped and closed.")
            except Exception as e_stream_close:
                log.error(f"Error closing microphone PyAudio stream: {e_stream_close}")
        
        if system_stream is not None:
            try:
                if system_stream.is_active():
                    system_stream.stop_stream()
                system_stream.close()
                log.info("System audio stream stopped and closed.")
            except Exception as e_stream_close:
                log.error(f"Error closing system PyAudio stream: {e_stream_close}")

        if p is not None:
            p.terminate()
            log.info("PyAudio instance terminated.")

    # --- Process and Mix Audio Data ---
    mic_audio_np = np.concatenate(mic_audio_chunks, axis=0) if mic_audio_chunks else np.array([], dtype=np.float32)
    system_audio_np = np.array([], dtype=np.float32)

    if actual_system_audio_device_index is not None:
        if actual_system_audio_device_index != actual_mic_device_index:
            if system_audio_chunks:
                system_audio_np = np.concatenate(system_audio_chunks, axis=0)
            else: # system stream was intended but no data
                 log.warning("System audio stream was attempted but no data was captured.")
        elif mic_audio_chunks: # Mic and System are the same device
            log.info("Using microphone audio as system audio (since devices are identical).")
            system_audio_np = mic_audio_np.copy() # Duplicate mic data
    
    mic_audio_available = mic_audio_np.size > 0
    system_audio_available = system_audio_np.size > 0

    if not mic_audio_available and not system_audio_available:
        log.warning("No audio was recorded from any source.")
        return

    mixed_audio_np = None
    if mic_audio_available and system_audio_available:
        log.info("Mixing microphone and system audio...")
        len_mic = len(mic_audio_np)
        len_system = len(system_audio_np)
        max_len = max(len_mic, len_system)

        if len_mic < max_len:
            mic_audio_np = np.pad(mic_audio_np, (0, max_len - len_mic), 'constant')
        if len_system < max_len:
            system_audio_np = np.pad(system_audio_np, (0, max_len - len_system), 'constant')
        
        mixed_audio_np = 0.5 * mic_audio_np + 0.5 * system_audio_np
        log.info(f"Mixed audio created. Length: {len(mixed_audio_np)} samples.")
    elif mic_audio_available:
        log.info("Using only microphone audio as system audio was not available or not captured.")
        mixed_audio_np = mic_audio_np
    elif system_audio_available: # This case implies mic was not available but system was
        log.info("Using only system audio as microphone audio was not available or not captured.")
        mixed_audio_np = system_audio_np
    else: # Should be caught by the earlier check
        log.warning("No audio data available for processing after attempting to mix.")
        return
        
    # Clear global lists
    mic_audio_chunks = [] 
    system_audio_chunks = []

    audio_data_flat = mixed_audio_np # mixed_audio_np is already flat and float32

    temp_audio_file = None
    try:
        # audio_data_flat is already float32 from the PyAudio callback conversion
        wav_bytes_for_db = core_processing.convert_float32_to_wav_bytes(audio_data_flat, samplerate)
        if not wav_bytes_for_db:
            log.error("Failed to convert recorded audio to WAV bytes. Cannot proceed.")
            return
        
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        # Use a more unique temp file name to avoid potential collisions if not cleaned up properly
        temp_file_name = f"temp_rec_pyaudio_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.wav"
        temp_audio_file = os.path.join(temp_dir, temp_file_name)
        
        with wave.open(temp_audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS) # CHANNELS is 1
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16)) # For WAV, common to save as Int16
            wf.setframerate(samplerate)
            # convert_float32_to_wav_bytes already creates int16 bytes
            wf.writeframes(wav_bytes_for_db) 
        log.info(f"Temporary audio file saved for transcription: {temp_audio_file}")

        log.info("Transcribing with Whisper (via core_processing)...")
        transcribed_text = core_processing.transcribe_audio_file(temp_audio_file, whisper_model_size)
        current_time_obj = datetime.now()
        log.info("\n--- Transcription ---")
        log.info(transcribed_text if transcribed_text else "[Transcription was empty]")

        # Save raw transcription to Markdown
        if transcribed_text:
            raw_transcription_md_path = core_processing.save_transcription_to_markdown(
                transcribed_text,
                current_time_obj,
                MARKDOWN_SAVE_PATH
            )
            if raw_transcription_md_path:
                log.info(f"Raw transcription also saved to: {raw_transcription_md_path}")
            else:
                log.warning("Failed to save raw transcription to a separate Markdown file.")

        analysis_results = None
        is_analysis_successful = False
        ollama_response_md_to_save = ''
        ollama_model_for_db = ''

        if transcribed_text:
            analysis_results = core_processing.analyze_transcription_with_ollama(
                transcribed_text,
                ollama_api_url=OLLAMA_API_URL,
                ollama_model_name=OLLAMA_MODEL_NAME
            )
            if analysis_results and analysis_results.get("full_markdown_response") not in FAILED_ANALYSIS_SUMMARIES and analysis_results.get("full_markdown_response", "").strip():
                is_analysis_successful = True
                log.info("\n--- Ollama Analysis (Raw Markdown) ---")
                log.info(f"Title: {analysis_results.get('title', 'N/A')}")
                log.info(f"Response Preview (first 150 chars):\n{analysis_results.get('full_markdown_response', 'N/A')[:150]}...")
                core_processing.save_markdown_file(
                    analysis_results.get('title', 'Meeting Analysis'),
                    analysis_results.get('full_markdown_response', ''),
                    current_time_obj,
                    MARKDOWN_SAVE_PATH
                )
                ollama_response_md_to_save = analysis_results.get('full_markdown_response', '')
                ollama_model_for_db = OLLAMA_MODEL_NAME
            else:
                log.warning("Ollama analysis was not successful or returned an empty/failed response.")
                if analysis_results: log.warning(f"Analysis response: {analysis_results.get('full_markdown_response')}")
        else:
            log.info("Transcription was empty. Skipping Ollama analysis.")

        current_time_iso = current_time_obj.isoformat()
        core_processing.save_to_db(current_time_iso, wav_bytes_for_db, samplerate, whisper_model_size, transcribed_text,
                                   ollama_model_for_db, ollama_response_md_to_save, db_name=DB_NAME)

    except Exception as e:
        log.exception(f"An error occurred during transcription, analysis, or saving: {e}")
        if 'transcribed_text' in locals() and transcribed_text and 'wav_bytes_for_db' in locals() and wav_bytes_for_db:
            try:
                log.warning("Attempting to save partial data due to error...")
                current_time_err_obj = datetime.now()
                current_time_err_iso = current_time_err_obj.isoformat()
                ollama_resp_on_err = analysis_results.get('full_markdown_response', '') if analysis_results else ''
                ollama_model_on_err = OLLAMA_MODEL_NAME if analysis_results and is_analysis_successful else ''
                if analysis_results and ollama_resp_on_err:
                     core_processing.save_markdown_file(
                        analysis_results.get('title', 'Error Analysis'),
                        ollama_resp_on_err,
                        current_time_err_obj,
                        MARKDOWN_SAVE_PATH
                    )
                core_processing.save_to_db(current_time_err_iso, wav_bytes_for_db, samplerate, whisper_model_size, transcribed_text,
                                           ollama_model_on_err, ollama_resp_on_err, db_name=DB_NAME)
            except Exception as db_err:
                log.exception(f"Failed to save partial data to DB after error: {db_err}")
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
                log.info(f"Temporary audio file {temp_audio_file} deleted.")
            except OSError as e:
                log.error(f"Error deleting temporary audio file {temp_audio_file}: {e}")

if __name__ == "__main__":
    # Initialize DB using the function from core_processing
    core_processing.init_db(db_name=DB_NAME)
    
    # Attempt to load the Whisper model once at startup using core_processing function
    try:
        core_processing.load_global_whisper_model(DEFAULT_WHISPER_MODEL)
    except Exception as e:
        log.critical(f"Fatal: Could not load the global Whisper model on startup: {e}")
        log.critical("The application cannot continue without a functional Whisper model.")
        log.critical("Please check your Whisper installation, model files, and system compatibility (e.g., ffmpeg).")
        sys.exit(1)

    log.info("--------------   PYNOTES --------------")
    log.debug(f"Recordings, transcriptions, and analyses will be saved to '{DB_NAME}'.")
    log.debug(f"Markdown files will be saved to '{MARKDOWN_SAVE_PATH}'.")
    log.debug(f"Using Ollama model for analysis: '{OLLAMA_MODEL_NAME}'")
    whisper_model_choice_default = DEFAULT_WHISPER_MODEL 
    selected_device_id_session = None

    log.info("\n--- Select Audio Device for Microphone ---")
    current_input_devices = list_audio_devices()
    if not current_input_devices: 
        log.warning("\nNo suitable input devices were found. Cannot start new recording.")
    
    try: 
        device_choice_input = input("\nEnter the PyAudio INDEX of the audio device, 'd' for default, or 'q' to exit: ").strip().lower()
    except ValueError: 
        log.warning("Invalid input. Enter a number (PyAudio Index), 'd', or 'b'.")
    except KeyboardInterrupt:
        log.info("\nDevice selection interrupted. Returning to main menu.")
    except EOFError:
        log.warning("\nEOF received during device selection. Returning to main menu.")
    except Exception as e: 
        log.exception(f"An unexpected error during device selection: {e}")
    if device_choice_input == 'd': 
        selected_device_id_session = None # This will use default input for mic
        log.info("Default input device will be used for the microphone.")
        potential_id = int(device_choice_input)
        # PyAudio device IDs are their indices
        if any(dev['id'] == potential_id for dev in current_input_devices):
            selected_device_id_session = potential_id
    elif device_choice_input == 'q':
        log.info("Exiting application.")
        sys.exit(0)
    #get the index of the device and set the selected_device_id_session to that index
    elif device_choice_input.isdigit():
        potential_id = int(device_choice_input)
        if any(dev['id'] == potential_id for dev in current_input_devices):
            selected_device_id_session = potential_id
    else:
        log.warning("Invalid input. Enter a number (PyAudio Index), 'd', or 'b'.")
    

    while True:
        log.info("\n--- Main Menu ---")
        action_choice = ""
        try:
            action_choice = input("Choose an action: (1) Re-analyze last recording, (2) Start new recording session, (q) Quit: ").strip().lower()
        except KeyboardInterrupt:
            log.info("\nExiting application due to KeyboardInterrupt.")
            sys.exit(0)
        except EOFError:
            log.info("\nExiting application due to EOF.")
            sys.exit(0)

        if action_choice == '1':
            last_record = core_processing.get_last_transcription_for_reanalysis(db_name=DB_NAME)
            if last_record and last_record['transcription']:
                log.info(f"\nRe-analyzing transcription for recording ID: {last_record['id']}")
                log.info(f"Original Transcription:\n'''{last_record['transcription']}'''")
                analysis_results = core_processing.analyze_transcription_with_ollama(
                    last_record['transcription'],
                    ollama_api_url=OLLAMA_API_URL,
                    ollama_model_name=OLLAMA_MODEL_NAME
                )
                is_reanalysis_successful = (analysis_results and 
                                          analysis_results.get("full_markdown_response") not in FAILED_ANALYSIS_SUMMARIES and 
                                          analysis_results.get("full_markdown_response", "").strip())
                if is_reanalysis_successful:
                    full_markdown_response = analysis_results.get('full_markdown_response', '')
                    title_from_reanalysis = analysis_results.get('title', 'Re-analysis')
                    current_time_reanalysis_obj = datetime.now()
                    core_processing.update_db_with_new_analysis(last_record['id'], OLLAMA_MODEL_NAME, full_markdown_response, db_name=DB_NAME)
                    log.info("\n--- Parsed Re-Analysis (Ollama Markdown) ---")
                    log.info(f"Title: {title_from_reanalysis}")
                    log.info(f"Response Preview (first 150 chars):\n{full_markdown_response[:150]}...")
                    core_processing.save_markdown_file(
                        title_from_reanalysis,
                        full_markdown_response,
                        current_time_reanalysis_obj,
                        MARKDOWN_SAVE_PATH
                    )
                else:
                    log.warning("Failed to re-analyze the transcription meaningfully. Previous analysis in DB remains unchanged.")
                    if analysis_results: 
                         log.warning(f"Re-analysis attempt raw response (not saved to DB): {analysis_results.get('full_markdown_response')}")
            else:
                log.info("No previous transcription found in the database to re-analyze, or last transcription was empty.")
        elif action_choice == '2':
            log.info(f"Starting new recording session with device index: {selected_device_id_session}")
            capture_and_transcribe(mic_device_id_selected_by_user=selected_device_id_session, whisper_model_size=whisper_model_choice_default)
        elif action_choice == 'q':
            log.info("Exiting application.")
            sys.exit(0)
        else:
            if action_choice:
                log.warning("Invalid choice. Please enter '1', '2', or 'q'.")
