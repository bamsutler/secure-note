import os
from datetime import datetime

# --- Menu & Prompts ---
RECORDING_PROMPT = "Type 'q' to stop recording: "
REPROCESSING_PROMPT = "Processing... Type 'q' to attempt to cancel (may not be immediate): " # For reprocessing state
MAIN_MENU_TEXT = """Choose an action:
(r) Start new recording session
(d) Doctor repair recordings without analysis
(q) Quit: """

DEVICE_SELECTION_PROMPT = "\nEnter the PyAudio INDEX of the audio device, 'd' for default, or 'q' to exit: "
CONFIRM_REANALYSIS_PROMPT = "Proceed with re-analysis? (y/n): "

# --- Messages ---
NO_DEVICES_FOUND_WARNING = "\nNo suitable input devices were found. Cannot start new recording."
INVALID_DEVICE_INPUT_WARNING = "Invalid input. Defaulting to system default device if available."
INVALID_DEVICE_INDEX_WARNING = "Invalid device index {}. Defaulting to system default if available."
DEFAULT_DEVICE_SELECTED_INFO = "Default input device will be used for the microphone."
SELECTED_DEVICE_INFO = "Selected device index for microphone: {}"
INVALID_CHOICE_WARNING = "Invalid choice. Please enter 'r', 'd', or 'q'."
EXITING_APP_INFO = "Exiting application."
APPLICATION_SHUTDOWN_INFO = "Application has shut down."
OPERATION_CANCELLED_INFO = "\nOperation cancelled by user."
REANALYSIS_FAILED_WARNING = "Failed to re-analyze the transcription meaningfully or all sections were empty."
NO_PREVIOUS_TRANSCRIPTION_INFO = "No previous transcription found in the database to re-analyze, or last transcription was empty."
REPROCESSING_LAST_TRANSCRIPTION_HEADER = "--- Reprocessing Last Transcription from Database ---"
NO_TRANSCRIPTION_FOUND_IN_DB_WARNING = "No suitable transcription found in the database for re-analysis."
REANALYSIS_CANCELLED_INFO = "Re-analysis cancelled by user."

# Doctor option specific messages
FOUND_UNANALYZED_RECORDS_PROMPT_TEMPLATE = "Found {} recording(s) with transcriptions but no analysis. Would you like to queue them for analysis? This may take some time. (y/n): "
QUEUEING_UNANALYZED_RECORDS_INFO_TEMPLATE = "Queueing {} recording(s) for background analysis."
NO_UNANALYZED_RECORDS_FOUND_INFO = "No recordings found that require pending analysis."
DOCTOR_QUEUE_COMPLETE_INFO = "Background analysis queueing complete. The doctor will now check the very last recording for interactive re-analysis."

def get_menu_text(is_recording=False, is_reprocessing=False):
    """Get the current menu text."""
    if is_recording:
        return RECORDING_PROMPT
    if is_reprocessing:
        return REPROCESSING_PROMPT # Or a specific message for reprocessing
    return MAIN_MENU_TEXT

def format_audio_device_list(devices):
    """Formats the list of audio devices for display and returns a list of formatted strings."""
    if not devices:
        return ["No input devices found by AudioService. Ensure your microphone or loopback device is enabled and recognized."]
    
    formatted_list = ["Available audio input devices (via AudioService):"]
    for i, dev_info in enumerate(devices):
        formatted_list.append(
            f"  ID {dev_info['id']}: {dev_info['name']} (Host API: {dev_info['host_api_name']}, Max Input Channels: {dev_info['max_input_channels']}, Default SR: {dev_info['default_sample_rate']})"
        )
    return formatted_list

def select_audio_device(audio_service, log, sys_module):
    """
    Handles the audio input device selection process.
    Lists input devices, shows default output, prompts user, and returns selection.

    Args:
        audio_service: Instance of AudioService.
        log: Logger instance.
        sys_module: The sys module (for sys.exit).

    Returns:
        tuple: (selected_device_id, list_of_input_devices) or calls sys.exit.
               selected_device_id is None for default, or an int for a specific device.
    """
    log.info("\n--- Select Audio Device for Microphone ---")
    input_devices = audio_service.list_input_devices()

    if not input_devices:
        log.warning(NO_DEVICES_FOUND_WARNING) # Constant from this module
        # No devices, so return None and an empty list, let caller decide if fatal
        # return None, [] 
        # Decided to make this fatal as per original logic in cli_main where it warns and then proceeds to ask for input anyway, which is odd.
        # If no input devices, it's better to be clear that recording cannot start.
        log.critical("No input devices detected. The application cannot proceed with recording.")
        log.critical("Please ensure your microphone or loopback audio device is connected and enabled.")
        sys_module.exit(1)

    # Display input devices
    formatted_input_devices = format_audio_device_list(input_devices)
    display_messages(log.info, formatted_input_devices)

    # Display default output device using AudioService
    default_output_dev_info = audio_service.get_default_output_device_info()
    if default_output_dev_info:
        log.info(f"Default output device: {default_output_dev_info.get('name')} (Index: {default_output_dev_info.get('index')})")
    else:
        log.warning("Could not retrieve default output device information.")

    selected_device_id_session = None
    device_choice_input = ''
    try:
        device_choice_input = input(DEVICE_SELECTION_PROMPT).strip().lower()
    except ValueError:
        log.warning(INVALID_DEVICE_INPUT_WARNING)
    except Exception as e:
        log.exception(f"An unexpected error occurred during device selection: {e}. Exiting.")
        sys_module.exit(1)

    if device_choice_input == 'd':
        selected_device_id_session = None
        log.info(DEFAULT_DEVICE_SELECTED_INFO)
    elif device_choice_input == 'q':
        log.info(EXITING_APP_INFO)
        sys_module.exit(0)
    elif device_choice_input.isdigit():
        potential_id = int(device_choice_input)
        if any(dev['id'] == potential_id for dev in input_devices):
            selected_device_id_session = potential_id
            log.info(SELECTED_DEVICE_INFO.format(selected_device_id_session))
        else:
            log.warning(INVALID_DEVICE_INDEX_WARNING.format(potential_id))
            selected_device_id_session = None  # Fallback to default
    else:
        log.warning(INVALID_DEVICE_INPUT_WARNING)
        selected_device_id_session = None  # Fallback to default
    
    return selected_device_id_session, input_devices # Return devices for potential future use, though not strictly needed now.

def confirm_reanalysis() -> bool:
    """Prompts the user to confirm re-analysis."""
    while True:
        try:
            choice = input(CONFIRM_REANALYSIS_PROMPT).strip().lower()
            if choice == 'y':
                return True
            elif choice == 'n':
                return False
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.") # Direct print for simple prompt
        except KeyboardInterrupt:
            # print(OPERATION_CANCELLED_INFO) # Already logged by caller in transcribe.py
            raise # Re-raise to be caught by the main loop in transcribe.py
        except Exception as e:
            # log.error(f"Unexpected error during confirmation: {e}") # Avoid log here, simple print
            print(f"An error occurred: {e}. Please try again.")

def confirm_process_unanalyzed(count: int) -> bool:
    """Prompts the user to confirm processing of unanalyzed records."""
    prompt_message = FOUND_UNANALYZED_RECORDS_PROMPT_TEMPLATE.format(count)
    while True:
        try:
            choice = input(prompt_message).strip().lower()
            if choice == 'y':
                return True
            elif choice == 'n':
                return False
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
        except KeyboardInterrupt:
            # Let the calling function handle KeyboardInterrupt (e.g., log cancellation)
            raise
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.") # Simple feedback for input errors

def assemble_analysis_markdown(analysis_results, default_title="Analysis"):
    """
    Assembles the full markdown response from structured analysis_results.
    Returns a tuple: (final_markdown_to_save, is_successful, title_from_analysis).
    """
    full_markdown_response_parts = []
    title_from_analysis = analysis_results.get('title', default_title)
    is_successful = False # Default to false

    if analysis_results.get("error"):
        error_message = analysis_results.get('error')
        # Use a generic error message or a more specific one if available
        markdown_content = f"# {title_from_analysis}\n\nAnalysis encountered an error: {error_message}"
        # Handle specific error cases, e.g., empty transcription
        if "Transcription was empty" in error_message:
            pass # Or add specific markdown part, but the error message itself is usually sufficient
        full_markdown_response_parts.append(markdown_content)
    else:
        # Construct markdown from parts
        # The main title H1 will be added based on title_from_analysis before saving.
        # Here we just assemble the sections.
        if analysis_results.get('summary'):
            full_markdown_response_parts.append(f"## Summary\n{analysis_results.get('summary')}")
        
        if analysis_results.get('key_topics'):
            full_markdown_response_parts.append(f"## Key Topics\n{analysis_results.get('key_topics')}")

        if analysis_results.get('action_items'):
            full_markdown_response_parts.append(f"## Action Items\n{analysis_results.get('action_items')}")

        if analysis_results.get('open_questions'):
            full_markdown_response_parts.append(f"## Open Questions\n{analysis_results.get('open_questions')}")
        
        # Determine success based on content
        # Primary success: a non-error summary is present.
        # Secondary success: if no summary, but other key sections are present and not error placeholders.
        summary_content = analysis_results.get("summary", "")
        key_topics_content = analysis_results.get("key_topics", "")
        
        is_summary_valid = summary_content and "Could not generate summary" not in summary_content
        is_key_topics_valid = key_topics_content and "Could not generate key_topics" not in key_topics_content
        
        is_successful = is_summary_valid or is_key_topics_valid # Or add other sections to this logic

    full_markdown_output = "\n\n".join(full_markdown_response_parts) if full_markdown_response_parts else ""

    # Ensure title is not problematic before prepending
    if not title_from_analysis or "Transcription Empty" in title_from_analysis or "Analysis Failed" in title_from_analysis:
        if analysis_results.get("error"): # If there was an error, title might reflect that
            pass # Keep the error-infused title
        elif not full_markdown_output.strip(): # If no content and no error, title should reflect that.
             title_from_analysis = "Empty Analysis Result" 
             full_markdown_output = "No analysis content generated."
        # else, use the default_title or a generic one if title_from_analysis is bad

    # Prepend the main title to the assembled markdown content
    if title_from_analysis and not analysis_results.get("error"): # Avoid double H1 if error message already includes it
        final_markdown_to_save = f"# {title_from_analysis}\n\n{full_markdown_output}"
    elif analysis_results.get("error"): # Error message likely already formatted with a title
        final_markdown_to_save = full_markdown_output 
    else: # Fallback, e.g. if title was bad and no error
        final_markdown_to_save = f"# {default_title}\n\n{full_markdown_output}"


    if not full_markdown_response_parts and not analysis_results.get("error"):
        # This case handles if all LLM calls returned empty strings but no explicit "error" was set
        is_successful = False
        if not final_markdown_to_save.strip() or final_markdown_to_save == f"# {title_from_analysis}\n\n": # Check if effectively empty
           final_markdown_to_save = f"# {title_from_analysis}\n\nNo meaningful content generated." if title_from_analysis else "No meaningful content generated."

    return final_markdown_to_save, is_successful, title_from_analysis

def display_messages(log_func, messages):
    """Helper to log multiple messages."""
    if isinstance(messages, str):
        log_func(messages)
    elif isinstance(messages, list):
        for msg in messages:
            log_func(msg) 