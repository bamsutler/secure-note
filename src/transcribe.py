import sys
import threading
from datetime import datetime
import logging
import os 
import queue

from src.config_service import ConfigurationService
from src.logging_service import LoggingService
from src.menu_aware_log_handler import MenuAwareRichHandler
from src.audio_service import AudioService 
from src.analysis_service import AnalysisService 
from src.storage_service import StorageService 
from src.transcription_service import WhisperProvider 
# Import CLI interface
from src import cli_interface

# Create and configure our custom handler
menu_handler = MenuAwareRichHandler()
menu_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

# Configure root logger to use our handler
LoggingService.setup_root_logger(level=logging.INFO, handler=menu_handler)
log = LoggingService.get_logger(__name__)

config = ConfigurationService()
audio_service = AudioService() 
analysis_service = AnalysisService() 
storage_service = StorageService() 
whisper_provider = WhisperProvider(config_service=config)

# --- Use constants from core_processing ---
FAILED_ANALYSIS_SUMMARIES = config.get('error_messages', 'failed_analysis')
DEFAULT_WHISPER_MODEL = config.get('models', 'whisper', 'default')

# --- PyAudio Specific Constants ---
CHANNELS = config.get('audio', 'channels')
stop_event = threading.Event()

# Add a queue for processing tasks
processing_queue = queue.Queue()
processing_thread = None
is_processing = False

def input_listener():
    """Listens for the 'q' command from the user."""
    global stop_event
    menu_handler.update_menu(cli_interface.get_menu_text(is_recording=True))
    while not stop_event.is_set():
        try:
            command = input().strip().lower()
            if command == "q":
                log.info("Quit command received.")
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

def process_audio_in_background(audio_data, samplerate, whisper_model_size):
    """Process audio data in the background."""
    global is_processing
    try:
        is_processing = True
        temp_audio_file = None
        try:
            # Convert audio data to WAV bytes
            wav_bytes_for_db = audio_service.convert_float32_to_wav_bytes(audio_data, samplerate)
            if not wav_bytes_for_db:
                log.error("Failed to convert recorded audio to WAV bytes.")
                return

            # Save temporary WAV file using StorageService
            base_temp_filename = f"rec_pyaudio_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
            temp_audio_file = storage_service.save_temporary_audio_wav(
                wav_bytes=wav_bytes_for_db,
                base_filename=base_temp_filename,
                samplerate=samplerate,
                channels=CHANNELS # CHANNELS is a global in transcribe.py
            )

            if not temp_audio_file:
                log.error("Failed to save temporary audio WAV file via StorageService.")
                return
            
            # Transcribe audio
            log.info("Transcribing with Whisper via WhisperProvider...")
            transcribed_text = whisper_provider.transcribe(
                audio_file_path=temp_audio_file, 
                model_name=whisper_model_size
            )
            current_time_obj = datetime.now()

            if transcribed_text:
                # Save initial recording and transcription to DB
                record_id = storage_service.save_initial_recording_and_transcription(
                    timestamp=current_time_obj,
                    audio_wav_bytes=wav_bytes_for_db,
                    samplerate=samplerate,
                    whisper_model_used=whisper_model_size,
                    transcription=transcribed_text
                )

                if not record_id:
                    log.error("Failed to save initial recording and transcription to database. Aborting further processing for this audio.")
                    # Clean up temp file if saving to DB failed before analysis
                    if temp_audio_file:
                        storage_service.delete_file(temp_audio_file)
                    return # Critical error, cannot proceed without a DB record ID
                
                log.info(f"Initial data saved to DB with record ID: {record_id}")

                # Process with AnalysisService
                log.info("Processing with AnalysisService for structured analysis...")
                analysis_results = analysis_service.analyze_transcription(transcription=transcribed_text)
                
                # Assemble markdown using cli_interface
                final_markdown_to_save, is_analysis_successful, title_from_analysis = cli_interface.assemble_analysis_markdown(
                    analysis_results, 
                    default_title="Analysis"
                )

                # Let's strip H1 here if present for `response_for_db`.
                # This needs to be done BEFORE calling save_markdown_content
                response_body_for_db = final_markdown_to_save
                if final_markdown_to_save.startswith(f"# {title_from_analysis}\n\n"):
                    response_body_for_db = final_markdown_to_save[len(f"# {title_from_analysis}\n\n"):]

                if is_analysis_successful:
                    # Save analysis
                    markdown_path = storage_service.save_markdown_content(
                        file_and_h1_title=title_from_analysis,
                        body_markdown_content=response_body_for_db, # Pass the H1-stripped body
                        timestamp_obj=current_time_obj,
                        type="analysis"
                    )
                    if markdown_path:
                        log.info(f"Analysis saved to: {markdown_path}")
                        log.info(f"\nAnalysis Preview (first 200 chars of assembled markdown):\n{final_markdown_to_save[:200]}...")
                else:
                    log.warning("Analysis was not successful or returned empty/error content for key sections.")
                    log.debug(f"Full analysis_results from AnalysisService: {analysis_results}")
                    # Log the generated markdown even if not "successful" for debugging
                    log.info(f"Generated markdown (even if not saved as primary):\n{final_markdown_to_save[:500]}...")

                model_used_for_db = analysis_results.get('model_used', 'default_analysis_model')

                # Update the existing DB record with analysis results
                update_success = storage_service.update_analysis(
                    recording_id=record_id,
                    llm_model_used=model_used_for_db, 
                    analysis_markdown=response_body_for_db, # This is H1 stripped
                    title_for_file=title_from_analysis # Store the H1/file title
                )
                if update_success:
                    log.info(f"Analysis for record ID {record_id} successfully updated in the database.")
                else:
                    log.error(f"Failed to update analysis in database for record ID {record_id}.")
                # The log message "Processing complete and saved to DB." might be slightly misleading now, 
                # as it's two steps, but it generally conveys completion.
            else:
                log.warning("No transcription generated from audio.")

        except Exception as e:
            log.error(f"Error in background processing: {e}")
        finally:
            if temp_audio_file: # temp_audio_file will be None if saving failed
                deleted = storage_service.delete_file(temp_audio_file)
                if not deleted:
                    # Log is already done by delete_file on error
                    pass # Or add specific log here if needed
    finally:
        is_processing = False

def background_processor():
    """Background thread to process audio data."""
    while True:
        try:
            task = processing_queue.get()
            if task is None:  # Poison pill to stop the thread
                break
            
            task_type = task[0]
            task_data = task[1]

            if task_type == 'new_recording':
                audio_data, samplerate, whisper_model_size = task_data
                process_audio_in_background(audio_data, samplerate, whisper_model_size)
            elif task_type == 'analyze_existing':
                # task_data is expected to be a dictionary (record_data)
                process_existing_for_analysis_in_background(task_data)
            else:
                log.warning(f"Unknown task type in processing queue: {task_type}")

            processing_queue.task_done()
        except Exception as e:
            log.error(f"Error in background processor: {e}")

def start_background_processor():
    """Start the background processing thread."""
    global processing_thread
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=background_processor, daemon=True)
        processing_thread.start()

def stop_background_processor():
    """Stop the background processing thread."""
    global processing_thread
    if processing_thread and processing_thread.is_alive():
        processing_queue.put(None)  # Poison pill
        processing_thread.join()

def process_existing_for_analysis_in_background(record_data: dict):
    """Processes an existing transcription (from DB record) for analysis in the background."""
    global is_processing # Consider if this global is still needed or if it can be task-specific
    try:
        # is_processing = True # Potentially set a flag specific to this task type if needed
        record_id = record_data.get('id')
        transcription_text = record_data.get('transcription')
        original_timestamp_obj = record_data.get('timestamp') # This should be a datetime object
        # whisper_model_used = record_data.get('whisper_model_used') # Not directly used for re-analysis, but available

        if not all([record_id, transcription_text, original_timestamp_obj]):
            log.error(f"Missing critical data for background analysis of record. Data: {record_data}")
            return

        log.info(f"Background: Starting analysis for existing record ID: {record_id} (Timestamp: {original_timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')})")
        log.debug(f"Background: Transcription for ID {record_id} (first 100 chars): {transcription_text[:100]}...")

        analysis_results = analysis_service.analyze_transcription(transcription=transcription_text)
        
        title_for_md = f"Analysis for Recording ID {record_id} ({original_timestamp_obj.strftime('%Y%m%d_%H%M%S')})"
        final_markdown_to_save, is_analysis_successful, title_from_analysis = cli_interface.assemble_analysis_markdown(
            analysis_results, 
            default_title=title_for_md
        )

        new_llm_model_used = analysis_results.get('model_used', 'background_analysis_model')

        if is_analysis_successful:
            log.info(f"Background: Analysis successful for record ID: {record_id}. Title: {title_from_analysis}")
            # Strip H1 for DB save if present
            response_body_for_db = final_markdown_to_save
            if final_markdown_to_save.startswith(f"# {title_from_analysis}\n\n"):
                response_body_for_db = final_markdown_to_save[len(f"# {title_from_analysis}\n\n"):]

            update_success = storage_service.update_analysis(
                recording_id=record_id,
                llm_model_used=new_llm_model_used,
                analysis_markdown=response_body_for_db, # This is H1 stripped
                title_for_file=title_from_analysis # Store the H1/file title
            )
            if update_success:
                log.info(f"Background: Database updated for record ID: {record_id}.")
            else:
                log.error(f"Background: Failed to update database for record ID: {record_id}.")

            # Save the new analysis to a markdown file
            # Use the original timestamp of the recording for the markdown filename
            md_path = storage_service.save_markdown_content(
                file_and_h1_title=title_from_analysis, # Use the title from assembly (might include ID)
                body_markdown_content=response_body_for_db, # CORRECTED: Pass the H1-stripped body
                timestamp_obj=original_timestamp_obj, # Original recording timestamp
                type="analysis" # Standard analysis type
            )
            if md_path:
                log.info(f"Background: Analysis for record ID {record_id} also saved to: {md_path}")
            else:
                log.warning(f"Background: Failed to save analysis markdown file for record ID: {record_id}.")
        else:
            log.warning(f"Background: Analysis not successful or returned empty for record ID: {record_id}.")
            log.debug(f"Background: Full analysis_results for ID {record_id}: {analysis_results}")
            # Even if not successful, update the DB to note an attempt was made, or save a placeholder?
            # Current behavior: only updates DB/saves file on success. This seems reasonable.

    except Exception as e:
        log.error(f"Error in background processing for existing record ID {record_data.get('id', 'Unknown')}: {e}")
    finally:
        # is_processing = False # Reset task-specific flag if used
        pass

def capture_and_transcribe(mic_device_id_selected_by_user=None, samplerate=16000, whisper_model_size=DEFAULT_WHISPER_MODEL):
    """
    Captures audio continuously from a selected microphone device AND the default system input device
    using AudioService. Mixes the audio, then queues it for background processing.
    The task put on the queue is now a tuple: ('new_recording', (mixed_audio_np, samplerate, whisper_model_size))
    """

    global stop_event 
    stop_event.clear()

    # Model loading is handled by WhisperProvider's transcribe method or ensured at startup.
    # No explicit call to whisper_provider.load_model here anymore.

    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()
    
    log.info(f"Attempting to start recording via AudioService at {samplerate} Hz.")
    log.info(f"Mic device index for AudioService: {mic_device_id_selected_by_user if mic_device_id_selected_by_user is not None else 'Default'}")
    
    recording_started = audio_service.start_recording(
        samplerate=samplerate,
        mic_device_index=mic_device_id_selected_by_user,
        include_system_audio=True 
    )

    if not recording_started:
        log.error("AudioService failed to start recording. Check logs from AudioService.")
        stop_event.set() 
        if listener_thread.is_alive():
            listener_thread.join(timeout=1.0)
        menu_handler.update_menu(cli_interface.get_menu_text(is_recording=False)) 
        return

    log.info("Recording started via AudioService. Waiting for 'q' or stream errors.")
    
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=0.2) 

    except Exception as e: 
        log.error(f"An unexpected error occurred during the recording wait loop: {e}")
        stop_event.set()
    finally:
        log.info("Stop event received or wait loop exited. Stopping recording via AudioService...")
        stop_event.set() 
        
        if listener_thread.is_alive():
            log.info("Waiting for input listener to complete...")
            listener_thread.join(timeout=1.5) 
            if listener_thread.is_alive():
                log.warning("Input listener did not stop cleanly.")

        mixed_audio_np, actual_samplerate = audio_service.stop_recording()
        
        if actual_samplerate != samplerate:
            log.warning(f"Recording was expected at {samplerate}Hz but AudioService returned {actual_samplerate}Hz. Using {actual_samplerate}Hz.")
            samplerate = actual_samplerate 

    if mixed_audio_np is not None and mixed_audio_np.size > 0:
        log.info("Audio captured via AudioService. Starting background processing...")
        processing_queue.put(('new_recording', (mixed_audio_np, samplerate, whisper_model_size)))
    else:
        log.warning("No audio data to process from AudioService.")

    menu_handler.update_menu(cli_interface.get_menu_text(is_recording=False))

def process_existing_transcription():
    """Doctor option: 
    1. Checks for recordings with transcriptions but no analysis and queues them.
    2. Then, proceeds to re-analyze the single most recent recording interactively.
    """
    log.info(cli_interface.REPROCESSING_LAST_TRANSCRIPTION_HEADER)
    menu_handler.update_menu(cli_interface.get_menu_text(is_reprocessing=True))

    current_operation_cancelled = False
    try:
        # Part 1: Check for and queue unanalyzed records
        unanalyzed_records = storage_service.get_recordings_without_analysis()
        if unanalyzed_records:
            log.info(f"Found {len(unanalyzed_records)} recording(s) that have transcriptions but are missing analysis.")
            if cli_interface.confirm_process_unanalyzed(len(unanalyzed_records)):
                log.info(cli_interface.QUEUEING_UNANALYZED_RECORDS_INFO_TEMPLATE.format(len(unanalyzed_records)))
                for record_data in unanalyzed_records:
                    processing_queue.put(('analyze_existing', record_data))
                log.info("All identified records have been queued for background analysis.")
            else:
                log.info("Skipping batch analysis of unanalyzed records.")
        else:
            log.info(cli_interface.NO_UNANALYZED_RECORDS_FOUND_INFO)
        
        log.info(cli_interface.DOCTOR_QUEUE_COMPLETE_INFO)

        # --- BEGIN NEW LOGIC: Check for missing markdown files ---
        log.info("\n--- Checking for missing Markdown files for analyzed records ---")
        try:
            records_with_analysis = storage_service.get_records_with_analysis()
            if not records_with_analysis:
                log.info("No records with analysis found in the database. Skipping missing file check.")
            else:
                markdown_dir = storage_service.markdown_save_path
                try:
                    disk_files = [f for f in os.listdir(markdown_dir) if f.endswith('.md')]
                    log.info(f"Found {len(disk_files)} markdown files in {markdown_dir}.")
                except FileNotFoundError:
                    log.warning(f"Markdown directory {markdown_dir} not found. Cannot check for missing files.")
                    disk_files = [] # Ensure disk_files is defined
                except Exception as e:
                    log.error(f"Error listing files in {markdown_dir}: {e}. Cannot check for missing files.")
                    disk_files = [] # Ensure disk_files is defined

                missing_files_found = 0
                regenerated_files_count = 0

                for record in records_with_analysis:
                    record_id = record.get('id')
                    analysis_md_content = record.get('analysis_markdown') # This is the body
                    record_timestamp_obj = record.get('timestamp') # datetime object
                    title_for_file_from_db = record.get('title_for_file')

                    if not analysis_md_content or not record_timestamp_obj or not title_for_file_from_db:
                        log.warning(f"Record ID {record_id} is missing critical data (body, timestamp, or title_for_file). Skipping file check for it.")
                        continue

                    # Use the definitive title from the database for filename generation
                    expected_filename = storage_service._generate_filename(
                        title_prefix=title_for_file_from_db,
                        timestamp_obj=record_timestamp_obj,
                        extension=".md"
                    )
                    
                    # Check against files in the main markdown_notes directory for originals
                    # and the recovered directory
                    expected_filepath_original = markdown_dir / expected_filename
                    expected_filepath_recovered = markdown_dir / "recovered" / expected_filename

                    # os.listdir gives filenames, not full paths, so check against expected_filename
                    if expected_filename not in disk_files and not os.path.exists(expected_filepath_recovered):
                        missing_files_found += 1
                        log.warning(f"Missing markdown file for Record ID: {record_id}. Expected original: {expected_filepath_original} or recovered: {expected_filepath_recovered}")
                        
                        log.info(f"Attempting to regenerate file for Record ID: {record_id} using stored title '{title_for_file_from_db}'...")
                        
                        saved_path = storage_service.save_markdown_content(
                            file_and_h1_title=title_for_file_from_db, 
                            body_markdown_content=analysis_md_content, # Body from DB
                            timestamp_obj=record_timestamp_obj,
                            type="analysis_recovered" # Saves to recovered/ subfolder
                        )
                        if saved_path:
                            log.info(f"Successfully regenerated and saved missing markdown file for Record ID {record_id} to: {saved_path}")
                            regenerated_files_count += 1
                        else:
                            log.error(f"Failed to regenerate markdown file for Record ID {record_id}.")
                    
                if missing_files_found == 0:
                    log.info("No missing markdown files found for analyzed records.")
                else:
                    log.info(f"Found {missing_files_found} missing markdown file(s). Attempted to regenerate {regenerated_files_count}.")

        except Exception as e:
            log.error(f"Error during missing markdown file check: {e}")
        # --- END NEW LOGIC ---

        # menu_handler.update_menu(cli_interface.get_menu_text()) # Reset menu for the next part

        # Part 2: Interactively re-analyze the single most recent recording (original behavior)
        log.info("\n--- Interactive Re-analysis of Last Recording ---")
        last_recording_data = storage_service.get_last_transcription_for_reanalysis()

        if not last_recording_data or not last_recording_data.get('transcription'):
            log.warning(cli_interface.NO_TRANSCRIPTION_FOUND_IN_DB_WARNING)
            # menu_handler.update_menu(cli_interface.get_menu_text()) # Already handled in finally
            return # Nothing to interactively re-analyze

        transcription_text = last_recording_data['transcription']
        recording_id = last_recording_data['id']
        original_whisper_model = last_recording_data.get('whisper_model_used', 'N/A')
        original_llm_model = last_recording_data.get('llm_model_used', 'N/A')
        original_timestamp_obj = last_recording_data.get('timestamp') # Already a datetime object from DB layer
        
        original_analysis_preview = (last_recording_data.get('analysis_markdown', '')[:100] + '...') \
            if last_recording_data.get('analysis_markdown') else 'N/A'

        log.info(f"Found last transcription (ID: {recording_id}, Timestamp: {original_timestamp_obj.strftime('%Y-%m-%d %H:%M:%S') if original_timestamp_obj else 'N/A'}) for interactive re-analysis:")
        log.info(f"  Original Whisper Model: {original_whisper_model}")
        log.info(f"  Original LLM Model: {original_llm_model}")
        log.info(f"  Original Analysis (preview): {original_analysis_preview}")
        log.info(f"  Transcription Text (preview):\n{transcription_text[:200]}...")
        
        # User previously removed the confirmation step here. Re-analysis will proceed directly.
        # if not cli_interface.confirm_reanalysis():
        #     log.info(cli_interface.REANALYSIS_CANCELLED_INFO)
        #     # menu_handler.update_menu(cli_interface.get_menu_text()) # Already handled in finally
        #     return

        log.info("\nProceeding with interactive re-analysis...")
        analysis_results = analysis_service.analyze_transcription(transcription=transcription_text)

        title_for_md_interactive = f"Re-analysis of ID {recording_id} ({datetime.now().strftime('%Y%m%d_%H%M%S')})"
        final_markdown_to_save_reanalysis, is_reanalysis_successful, title_from_reanalysis = cli_interface.assemble_analysis_markdown(
            analysis_results,
            default_title=title_for_md_interactive
        )
        
        current_time_reanalysis_obj = datetime.now() # Timestamp for this specific re-analysis action
        new_llm_model_used = analysis_results.get('model_used', 'default_reanalysis_model')

        if is_reanalysis_successful:
            log.info("--- Interactive Re-Analysis Results ---")
            log.info(f"Title: {title_from_reanalysis}")
            log.info(f"Assembled Markdown Preview (first 150 chars):\n{final_markdown_to_save_reanalysis[:150]}...")
            
            response_body_for_db = final_markdown_to_save_reanalysis
            if final_markdown_to_save_reanalysis.startswith(f"# {title_from_reanalysis}\n\n"):
                response_body_for_db = final_markdown_to_save_reanalysis[len(f"# {title_from_reanalysis}\n\n"):]

            update_success = storage_service.update_analysis(
                recording_id=recording_id,
                llm_model_used=new_llm_model_used,
                analysis_markdown=response_body_for_db, # H1 stripped body
                title_for_file=title_from_reanalysis # The title for H1 and filename
            )

            if update_success:
                log.info(f"Interactive re-analysis for recording ID {recording_id} successfully updated in the database.")
                new_md_path = storage_service.save_markdown_content(
                    file_and_h1_title=title_from_reanalysis, 
                    body_markdown_content=response_body_for_db, # H1 stripped body
                    timestamp_obj=current_time_reanalysis_obj, 
                    type="analysis_reprocessed" # Saves to reprocessed/ subfolder
                )
                if new_md_path:
                    log.info(f"Interactive re-analysis also saved to new markdown file: {new_md_path}")
            else:
                log.error(f"Failed to update interactive re-analysis in database for recording ID {recording_id}.")
        else:
            log.warning(cli_interface.REANALYSIS_FAILED_WARNING)
            log.info(f"Generated markdown (even if not saved as primary for interactive re-analysis):\n{final_markdown_to_save_reanalysis[:500]}...")

    except KeyboardInterrupt:
        log.info(cli_interface.OPERATION_CANCELLED_INFO)
        current_operation_cancelled = True # Flag that this specific operation was cancelled
    except Exception as e:
        log.error(f"Error during 'Doctor' operation (processing existing transcriptions): {e}")
    finally:
        # Only reset menu if the operation wasn't cancelled by 'q' inside a prompt that exits the app
        # This means if cli_interface.confirm_process_unanalyzed or cli_interface.confirm_reanalysis were to sys.exit(),
        # this wouldn't run. But they raise KeyboardInterrupt, which is handled.
        if not current_operation_cancelled:
             menu_handler.update_menu(cli_interface.get_menu_text()) # Reset to main menu after doctor ops
        # If it was cancelled, the main loop's menu update will handle it, or it might be exiting.

def cli_main():
    start_background_processor()
    
    try:
        try:
            # Use the new ensure_model_loaded method for startup check
            whisper_provider.ensure_model_loaded(model_name=DEFAULT_WHISPER_MODEL)
            # The log message about model loading is now handled by ensure_model_loaded or _load_model.
        except Exception as e:
            log.critical(f"Fatal: Could not load/verify the Whisper model ('{DEFAULT_WHISPER_MODEL}') via WhisperProvider on startup: {e}")
            log.critical("The application cannot continue without a functional Whisper model.")
            log.critical("Please check your Whisper installation, model files, and system compatibility (e.g., ffmpeg).")
            sys.exit(1)

        
        log.info("#" * 80 + " Secure Note version " + config.get_application_version() + "#" * 80) 
        
        # Use the new function from cli_interface to select the audio device
        # It handles logging, printing device lists, and user input.
        # It will call sys.exit if user quits or no devices are found.
        selected_device_id_session, _ = cli_interface.select_audio_device(audio_service, log, sys)
        # current_input_devices is returned but not strictly needed here anymore as selection is complete.

        while True:
            menu_handler.update_menu(cli_interface.get_menu_text(is_recording=False))
            action_choice = ""
            try:
                action_choice = input().strip().lower()
            except Exception as e:
                log.error(f"Error getting action choice: {e}")
                continue

            if action_choice == 'd':
                process_existing_transcription()
            elif action_choice == 'r':
                log.info(f"Starting new recording session with device index: {selected_device_id_session}")
                capture_and_transcribe(mic_device_id_selected_by_user=selected_device_id_session, whisper_model_size=DEFAULT_WHISPER_MODEL)
            elif action_choice == 'q':
                log.info(cli_interface.EXITING_APP_INFO)
                break
            else:
                if action_choice: 
                    log.warning(cli_interface.INVALID_CHOICE_WARNING)
    finally:
        log.info("Shutting down application...")
        stop_background_processor()
        if audio_service: # Ensure audio_service object exists
            audio_service.close() # Explicitly close audio service resources
        
        # Explicitly clear screen and show a final message if possible before exit
        if 'menu_handler' in globals() and hasattr(menu_handler, 'console'):
            # Check if the console is still available, as it might be affected during shutdown
            try:
                if menu_handler.console:
                    menu_handler.console.clear()
                    # Use a direct print for the very last message if logging might fail
                    # print(cli_interface.APPLICATION_SHUTDOWN_INFO)
                    # Or try logging, but it might fail here as well
                    log.info(cli_interface.APPLICATION_SHUTDOWN_INFO) 
            except Exception as e:
                # If clearing or logging fails here, just proceed to exit
                print(f"Error during final cleanup: {e}. Exiting.") # Fallback to simple print
        sys.exit(0)

if __name__ == "__main__":
    cli_main()
