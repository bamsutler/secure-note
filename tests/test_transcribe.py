import unittest
from unittest.mock import patch, MagicMock, call
import queue
from datetime import datetime
import numpy as np

# Modules to test (or parts of it)
from src.transcribe import process_audio_in_background, start_background_processor, stop_background_processor, processing_queue as transcribe_processing_queue

# Mocked global instances from transcribe.py
# These will be patched in the TestTranscribeApp class or individual tests.

class TestTranscribeApp(unittest.TestCase):

    @patch('src.transcribe.LoggingService')
    @patch('src.transcribe.ConfigurationService')
    @patch('src.transcribe.AudioService')
    @patch('src.transcribe.AnalysisService')
    @patch('src.transcribe.StorageService')
    @patch('src.transcribe.WhisperProvider')
    @patch('src.transcribe.cli_interface') # For assemble_analysis_markdown
    @patch('src.transcribe.datetime') # For datetime.now()
    def setUp(self, mock_datetime, mock_cli_interface, mock_whisper_provider_class, mock_storage_service_class, 
              mock_analysis_service_class, mock_audio_service_class, mock_config_service_class, mock_logging_service_class):
        
        # Setup mock for LoggingService and its logger
        self.mock_log = MagicMock()
        mock_logging_service_class.get_logger.return_value = self.mock_log
        # Patch the 'log' object specifically in the 'transcribe' module's scope for its functions
        # This requires knowing how 'log' is obtained in transcribe.py (e.g., transcribe.log)
        # For now, we assume functions in transcribe.py use a logger obtained via LoggingService.get_logger
        # If 'transcribe.log' is a global, it needs a different patch target.

        # Mock ConfigurationService instance and its 'get' method
        self.mock_config_service = MagicMock()
        mock_config_service_class.return_value = self.mock_config_service
        self.mock_config_service.get.side_effect = self._config_get_side_effect

        # Instantiate and assign mock services that are globally used in transcribe.py
        self.mock_audio_service = MagicMock()
        mock_audio_service_class.return_value = self.mock_audio_service
        
        self.mock_analysis_service = MagicMock()
        mock_analysis_service_class.return_value = self.mock_analysis_service
        
        self.mock_storage_service = MagicMock()
        mock_storage_service_class.return_value = self.mock_storage_service
        
        self.mock_whisper_provider = MagicMock()
        mock_whisper_provider_class.return_value = self.mock_whisper_provider

        self.mock_cli_interface = mock_cli_interface
        self.mock_datetime = mock_datetime
        
        # Reset shared state for background processing if necessary
        transcribe_processing_queue.queue.clear() # Clear items from previous tests
        # If there's a global 'stop_event' in transcribe.py, mock it or re-initialize it here.
        # e.g., with @patch('transcribe.stop_event', new_callable=threading.Event) or similar
        
        # Set up is_processing flag used in transcribe.py
        # This is a global in transcribe.py, so we patch it there.
        self.is_processing_patcher = patch('src.transcribe.is_processing', False)
        self.mock_is_processing = self.is_processing_patcher.start()


    def tearDown(self):
        self.is_processing_patcher.stop()
        # Ensure background processor is stopped if it was started by a test
        # This might involve checking 'transcribe.processing_thread'
        # For now, assume tests manage this explicitly or it's handled by mock Queue


    def _config_get_side_effect(self, *args):
        if args == ('error_messages', 'failed_analysis'):
            return "Analysis failed."
        if args == ('models', 'whisper', 'default'):
            return 'tiny-test'
        if args == ('audio', 'channels'):
            return 1
        # Add more config mocks as needed by functions in transcribe.py
        return MagicMock()

    @patch('src.transcribe.log', new_callable=MagicMock) # Explicitly patch the 'log' object used in the function
    def test_process_audio_in_background_success(self, mock_log_for_function):
        with patch('src.transcribe.audio_service', self.mock_audio_service), \
             patch('src.transcribe.storage_service', self.mock_storage_service), \
             patch('src.transcribe.whisper_provider', self.mock_whisper_provider), \
             patch('src.transcribe.analysis_service', self.mock_analysis_service), \
             patch('src.transcribe.cli_interface', self.mock_cli_interface), \
             patch('src.transcribe.datetime', self.mock_datetime), \
             patch('src.transcribe.CHANNELS', 1):

            # --- Arrange ---
            test_audio_data = np.random.rand(16000).astype(np.float32)
            test_samplerate = 16000
            test_whisper_model = "tiny-test"
    
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            self.mock_datetime.now.return_value = mock_now
    
            self.mock_audio_service.convert_float32_to_wav_bytes.return_value = b"wav_bytes"
            self.mock_storage_service.save_temporary_audio_wav.return_value = "/tmp/temp_audio.wav"
            self.mock_whisper_provider.transcribe.return_value = "This is a test transcription."
            # self.mock_storage_service.save_raw_transcription.return_value = "/path/to/raw_transcription.md" # No longer called
            
            # Mock for save_initial_recording_and_transcription
            mock_record_id = 123
            self.mock_storage_service.save_initial_recording_and_transcription.return_value = mock_record_id

            mock_analysis_results = {
                'title': 'Test Analysis', 'summary': 'Summary of test.', 
                'action_items': ['Action 1'], 'sentiment': 'neutral',
                'raw_llm_response': '{\"title\": \"Test Analysis\"}',
                'model_used': 'gpt-test', 'success': True
            }
            self.mock_analysis_service.analyze_transcription.return_value = mock_analysis_results
    
            # assemble_analysis_markdown returns: (markdown_string, success_bool, title_string)
            # The summary part for the DB is extracted from markdown_string
            expected_markdown_output = "# Test Analysis\n\nSummary of test."
            self.mock_cli_interface.assemble_analysis_markdown.return_value = (expected_markdown_output, True, "Test Analysis")
            self.mock_storage_service.save_markdown_content.return_value = "/path/to/analysis.md"
            self.mock_storage_service.delete_file.return_value = True
            self.mock_storage_service.update_analysis.return_value = True # Mock for update_analysis

            # --- Act ---
            process_audio_in_background(test_audio_data, test_samplerate, test_whisper_model)
    
            # --- Assert ---
            self.mock_audio_service.convert_float32_to_wav_bytes.assert_called_once_with(test_audio_data, test_samplerate)
            self.mock_storage_service.save_temporary_audio_wav.assert_called_once_with(
                wav_bytes=b"wav_bytes", 
                base_filename=unittest.mock.ANY, 
                samplerate=test_samplerate, 
                channels=1
            )
            self.mock_whisper_provider.transcribe.assert_called_once_with(
                audio_file_path="/tmp/temp_audio.wav", 
                model_name=test_whisper_model
            )
            # self.mock_storage_service.save_raw_transcription.assert_called_once_with( # No longer called
            #     text_content="This is a test transcription.",
            #     timestamp_obj=mock_now
            # )
            self.mock_storage_service.save_initial_recording_and_transcription.assert_called_once_with(
                timestamp=mock_now,
                audio_wav_bytes=b"wav_bytes",
                samplerate=test_samplerate,
                whisper_model_used=test_whisper_model,
                transcription="This is a test transcription."
            )
            self.mock_analysis_service.analyze_transcription.assert_called_once_with(transcription="This is a test transcription.")
            self.mock_cli_interface.assemble_analysis_markdown.assert_called_once_with(
                mock_analysis_results, default_title="Analysis"
            )
            self.mock_storage_service.save_markdown_content.assert_called_once_with(
                file_and_h1_title="Test Analysis",
                body_markdown_content="Summary of test.", # Content with H1 stripped
                timestamp_obj=mock_now,
                type="analysis"
            )
            # self.mock_storage_service.save_recording_and_analysis.assert_called_once_with( # No longer called
            #     timestamp=mock_now,
            #     audio_wav_bytes=b"wav_bytes",
            #     samplerate=test_samplerate,
            #     whisper_model_used=test_whisper_model,
            #     transcription="This is a test transcription.",
            #     llm_model_used='gpt-test',
            #     analysis_markdown="Summary" # Body after stripping H1
            # )
            self.mock_storage_service.update_analysis.assert_called_once_with(
                recording_id=mock_record_id,
                llm_model_used='gpt-test',
                analysis_markdown="Summary of test.", # Extracted summary
                title_for_file="Test Analysis" # Add title_for_file
            )
            self.mock_storage_service.delete_file.assert_called_once_with("/tmp/temp_audio.wav")

    @patch('src.transcribe.log', new_callable=MagicMock)
    def test_process_audio_in_background_transcription_fails(self, mock_log_for_function):
        with patch('src.transcribe.audio_service', self.mock_audio_service), \
             patch('src.transcribe.storage_service', self.mock_storage_service), \
             patch('src.transcribe.whisper_provider', self.mock_whisper_provider), \
             patch('src.transcribe.analysis_service', self.mock_analysis_service), \
             patch('src.transcribe.datetime', self.mock_datetime), \
             patch('src.transcribe.CHANNELS', 1):

            test_audio_data = np.random.rand(16000).astype(np.float32)
            test_samplerate = 16000
            test_whisper_model = "tiny-test"

            self.mock_audio_service.convert_float32_to_wav_bytes.return_value = b"wav_bytes"
            self.mock_storage_service.save_temporary_audio_wav.return_value = "/tmp/temp_audio.wav"
            self.mock_whisper_provider.transcribe.return_value = None # Simulate transcription failure

            process_audio_in_background(test_audio_data, test_samplerate, test_whisper_model)

            self.mock_whisper_provider.transcribe.assert_called_once()
            self.mock_analysis_service.analyze_transcription.assert_not_called()
            self.mock_storage_service.save_raw_transcription.assert_not_called()
            self.mock_storage_service.save_markdown_content.assert_not_called()
            self.mock_storage_service.save_recording_and_analysis.assert_not_called()
            self.mock_storage_service.delete_file.assert_called_once_with("/tmp/temp_audio.wav")
            mock_log_for_function.warning.assert_any_call("No transcription generated from audio.")
            
    # More tests for other functions in transcribe.py (e.g., capture_and_transcribe, cli_main) will go here.
    # They will require more complex mocking, especially for cli_main's input loop and sys.exit.

    # Example for testing the background queue mechanism (simplified)
    @patch('src.transcribe.process_audio_in_background') # Mock the actual processing function
    def test_background_processor_picks_task_from_queue(self, mock_process_audio):
        # Ensure transcribe.py's queue is used for this test
        # Clear the queue before starting, as it's a global module queue
        while not transcribe_processing_queue.empty():
            try:
                transcribe_processing_queue.get_nowait()
            except queue.Empty:
                break

        with patch('src.transcribe.processing_queue', transcribe_processing_queue): # Ensure we are using the right queue if imported differently
            start_background_processor() # Starts the thread
    
            # Define the data payload for the 'new_recording' task
            audio_np_array = np.array([1.0])
            sample_rate_val = 16000
            model_name_val = "test_model"
            audio_data_payload = (audio_np_array, sample_rate_val, model_name_val)
            
            # Structure the task as (task_type, task_data)
            task_to_queue = ('new_recording', audio_data_payload)
            transcribe_processing_queue.put(task_to_queue)
    
            # Give the background thread some time to process
            transcribe_processing_queue.join() # Wait for task to be processed
    
            mock_process_audio.assert_called_once_with(audio_np_array, sample_rate_val, model_name_val)
            
            stop_background_processor() # Stops the thread by putting None and joining

    @patch('src.transcribe.log', new_callable=MagicMock)
    @patch('src.transcribe.menu_handler', new_callable=MagicMock) # Mock menu_handler used in input_listener and capture_and_transcribe
    @patch('src.transcribe.input_listener') # Mock the input_listener function directly
    @patch('src.transcribe.threading.Thread') # To mock listener_thread = threading.Thread(...)
    @patch('src.transcribe.stop_event') # Mock the global stop_event from transcribe.py
    def test_capture_and_transcribe_success(self, mock_stop_event, mock_thread_class, mock_input_listener_func, mock_menu_handler, mock_log_func):
        with patch('src.transcribe.audio_service', self.mock_audio_service), \
             patch('src.transcribe.processing_queue', transcribe_processing_queue), \
             patch('src.transcribe.cli_interface', self.mock_cli_interface): # For get_menu_text

            # --- Arrange ---
            test_mic_device_id = 1
            test_whisper_model = "tiny-test-capture"
            expected_samplerate = 16000
            mock_audio_data = np.random.rand(expected_samplerate).astype(np.float32)

            # Configure mocks
            mock_stop_event.is_set.side_effect = [False, False, True] # Simulate running then stopping
            self.mock_audio_service.start_recording.return_value = True # Recording starts successfully
            self.mock_audio_service.stop_recording.return_value = (mock_audio_data, expected_samplerate)
            self.mock_cli_interface.get_menu_text.return_value = "Test Menu"
            mock_listener_thread_instance = MagicMock()
            mock_thread_class.return_value = mock_listener_thread_instance

            # Ensure queue is clear before test
            while not transcribe_processing_queue.empty():
                transcribe_processing_queue.get_nowait()

            # --- Act ---
            # Import here to use the class-level mocks for global services if not already patched
            from src.transcribe import capture_and_transcribe 
            capture_and_transcribe(mic_device_id_selected_by_user=test_mic_device_id, samplerate=expected_samplerate, whisper_model_size=test_whisper_model)

            # --- Assert ---
            mock_stop_event.clear.assert_called_once()
            mock_thread_class.assert_called_once_with(target=mock_input_listener_func, daemon=True)
            mock_listener_thread_instance.start.assert_called_once()
            
            self.mock_audio_service.start_recording.assert_called_once_with(
                samplerate=expected_samplerate,
                mic_device_index=test_mic_device_id,
                include_system_audio=True
            )
            mock_log_func.info.assert_any_call("Audio captured via AudioService. Starting background processing...")
            self.mock_audio_service.stop_recording.assert_called_once()
            
            # Check if item was put on the queue
            self.assertFalse(transcribe_processing_queue.empty())
            queued_task = transcribe_processing_queue.get_nowait()
            
            # Assert the structure and content of the queued task
            self.assertIsInstance(queued_task, tuple, "Queued task should be a tuple")
            self.assertEqual(len(queued_task), 2, "Queued task should have two elements (type, data)")
            
            task_type, task_data = queued_task
            self.assertEqual(task_type, 'new_recording', "Task type should be 'new_recording'")
            
            self.assertIsInstance(task_data, tuple, "Task data should be a tuple")
            self.assertEqual(len(task_data), 3, "Task data tuple should have three elements (audio, rate, model)")
            
            queued_audio_data, queued_samplerate, queued_model_name = task_data
            
            self.assertIsInstance(queued_audio_data, np.ndarray, "Queued audio data should be a numpy array")
            self.assertTrue(np.array_equal(queued_audio_data, mock_audio_data), "Queued audio data does not match expected")
            self.assertEqual(queued_samplerate, expected_samplerate, "Queued samplerate does not match expected")
            self.assertEqual(queued_model_name, test_whisper_model, "Queued model name does not match expected")

            mock_menu_handler.update_menu.assert_any_call("Test Menu") # Check if menu was updated
            mock_listener_thread_instance.join.assert_called_with(timeout=1.5)

    @patch('src.transcribe.log', new_callable=MagicMock)
    @patch('src.transcribe.menu_handler', new_callable=MagicMock)
    @patch('src.transcribe.input_listener')
    @patch('src.transcribe.threading.Thread')
    @patch('src.transcribe.stop_event')
    def test_capture_and_transcribe_start_recording_fails(self, mock_stop_event, mock_thread_class, mock_input_listener_func, mock_menu_handler, mock_log_func):
        with patch('src.transcribe.audio_service', self.mock_audio_service), \
             patch('src.transcribe.cli_interface', self.mock_cli_interface): 

            # --- Arrange ---
            self.mock_audio_service.start_recording.return_value = False # Recording fails to start
            mock_listener_thread_instance = MagicMock()
            mock_thread_class.return_value = mock_listener_thread_instance
            self.mock_cli_interface.get_menu_text.return_value = "Test Menu"

            # --- Act ---
            from src.transcribe import capture_and_transcribe
            capture_and_transcribe(mic_device_id_selected_by_user=0, whisper_model_size="test")

            # --- Assert ---
            mock_stop_event.clear.assert_called_once()
            mock_thread_class.assert_called_once()
            mock_listener_thread_instance.start.assert_called_once()
            self.mock_audio_service.start_recording.assert_called_once()
            mock_log_func.error.assert_any_call("AudioService failed to start recording. Check logs from AudioService.")
            self.mock_audio_service.stop_recording.assert_not_called() # Should not be called if start fails
            self.assertTrue(transcribe_processing_queue.empty()) # No item should be queued
            mock_menu_handler.update_menu.assert_any_call("Test Menu")
            mock_stop_event.set.assert_called() # Ensure stop_event is set on failure

    @patch('src.transcribe.log', new_callable=MagicMock)
    @patch('src.transcribe.menu_handler', new_callable=MagicMock)
    @patch('src.transcribe.input_listener')
    @patch('src.transcribe.threading.Thread')
    @patch('src.transcribe.stop_event')
    def test_capture_and_transcribe_no_audio_data_returned(self, mock_stop_event, mock_thread_class, mock_input_listener_func, mock_menu_handler, mock_log_func):
        with patch('src.transcribe.audio_service', self.mock_audio_service), \
             patch('src.transcribe.cli_interface', self.mock_cli_interface): 

            # --- Arrange ---
            mock_stop_event.is_set.side_effect = [False, False, True]
            self.mock_audio_service.start_recording.return_value = True
            self.mock_audio_service.stop_recording.return_value = (None, 16000) # No audio data
            mock_listener_thread_instance = MagicMock()
            mock_thread_class.return_value = mock_listener_thread_instance
            self.mock_cli_interface.get_menu_text.return_value = "Test Menu"

            # --- Act ---
            from src.transcribe import capture_and_transcribe
            capture_and_transcribe(mic_device_id_selected_by_user=0, whisper_model_size="test")

            # --- Assert ---
            self.mock_audio_service.start_recording.assert_called_once()
            self.mock_audio_service.stop_recording.assert_called_once()
            mock_log_func.warning.assert_any_call("No audio data to process from AudioService.")
            self.assertTrue(transcribe_processing_queue.empty()) # No item should be queued
            mock_menu_handler.update_menu.assert_any_call("Test Menu")

    @patch('src.transcribe.log', new_callable=MagicMock)
    @patch('src.transcribe.menu_handler', new_callable=MagicMock) # Mock menu_handler
    @patch('src.transcribe.stop_event') # Mock global stop_event
    @patch('builtins.input') # Mock built-in input()
    def test_input_listener_quit_command(self, mock_input, mock_stop_event, mock_menu_handler, mock_log_func):
        with patch('src.transcribe.cli_interface', self.mock_cli_interface): # For get_menu_text
            # --- Arrange ---
            # Simulate input 'q' then stop_event.is_set() becomes True to exit loop
            mock_input.side_effect = ["q"]
            mock_stop_event.is_set.side_effect = [False, True] # Loop once, then exit
            self.mock_cli_interface.get_menu_text.return_value = "Recording Menu"

            # --- Act ---
            from src.transcribe import input_listener
            input_listener()

            # --- Assert ---
            mock_menu_handler.update_menu.assert_called_once_with("Recording Menu")
            mock_input.assert_called_once()
            mock_stop_event.set.assert_called_once() # 'q' should call stop_event.set()
            mock_log_func.info.assert_any_call("Quit command received.")

    @patch('src.transcribe.log', new_callable=MagicMock)
    @patch('src.transcribe.menu_handler', new_callable=MagicMock)
    @patch('src.transcribe.stop_event')
    @patch('builtins.input')
    def test_input_listener_eof_error(self, mock_input, mock_stop_event, mock_menu_handler, mock_log_func):
        with patch('src.transcribe.cli_interface', self.mock_cli_interface):
            # --- Arrange ---
            mock_input.side_effect = EOFError("Simulated EOF")
            # is_set will be checked once, then error occurs, loop should break.
            mock_stop_event.is_set.return_value = False 
            self.mock_cli_interface.get_menu_text.return_value = "Recording Menu"

            # --- Act ---
            from src.transcribe import input_listener
            input_listener()

            # --- Assert ---
            mock_menu_handler.update_menu.assert_called_once_with("Recording Menu")
            mock_input.assert_called_once()
            mock_stop_event.set.assert_called_once() # EOFError should call stop_event.set()
            mock_log_func.warning.assert_any_call("EOF received, stopping listener.")

    @patch('src.transcribe.log', new_callable=MagicMock)
    @patch('src.transcribe.menu_handler', new_callable=MagicMock)
    @patch('src.transcribe.stop_event')
    @patch('builtins.input')
    def test_input_listener_keyboard_interrupt(self, mock_input, mock_stop_event, mock_menu_handler, mock_log_func):
        with patch('src.transcribe.cli_interface', self.mock_cli_interface):
            # --- Arrange ---
            mock_input.side_effect = KeyboardInterrupt("Simulated Ctrl+C")
            mock_stop_event.is_set.return_value = False
            self.mock_cli_interface.get_menu_text.return_value = "Recording Menu"

            # --- Act ---
            from src.transcribe import input_listener
            input_listener()

            # --- Assert ---
            mock_menu_handler.update_menu.assert_called_once_with("Recording Menu")
            mock_input.assert_called_once()
            mock_stop_event.set.assert_called_once() # KeyboardInterrupt should call stop_event.set()
            mock_log_func.info.assert_any_call("Keyboard interrupt in listener, stopping.")

    @patch('src.transcribe.log', new_callable=MagicMock)
    # No longer patching 'open' as process_existing_transcription gets data from DB for re-analysis
    def test_process_existing_transcription_success(self, mock_log_func):
        """Test successful re-analysis of a transcription obtained from the database."""
        with patch('src.transcribe.cli_interface', self.mock_cli_interface), \
             patch('src.transcribe.storage_service', self.mock_storage_service), \
             patch('src.transcribe.analysis_service', self.mock_analysis_service), \
             patch('src.transcribe.datetime', self.mock_datetime):

            # --- Arrange ---
            mock_now_reanalysis = datetime(2023, 2, 1, 10, 0, 0)
            mock_original_timestamp = datetime(2023, 1, 1, 12, 0, 0)
            self.mock_datetime.now.return_value = mock_now_reanalysis
            
            mock_record_from_db = {
                'id': 303, 'transcription': "This is the existing transcription text from DB", 
                'timestamp': mock_original_timestamp,
                'whisper_model_used': 'base', 'llm_model_used': 'old_llm', 
                'analysis_markdown': '# Old Analysis'
            }
            self.mock_storage_service.get_recordings_without_analysis.return_value = []
            self.mock_storage_service.get_last_transcription_for_reanalysis.return_value = mock_record_from_db

            mock_analysis_results = {
                'title': 'Re-Analysis Title', 'summary': 'Re-analyzed summary.', 
                'action_items': ['New Action'], 'sentiment': 'positive',
                'raw_llm_response': '{\\"title\\": \\"Re-Analysis Title\\"}',
                'model_used': 'gpt-re-test', 'success': True
            }
            self.mock_analysis_service.analyze_transcription.return_value = mock_analysis_results
            
            expected_markdown_to_save = "# Re-Analysis Title\\\\n\\\\nRe-analyzed content."
            # The actual title used for assemble_analysis_markdown will include a timestamp
            # So we mock based on that, and it should be "Re-Analysis Title"
            self.mock_cli_interface.assemble_analysis_markdown.return_value = (expected_markdown_to_save, True, "Re-Analysis Title")
            self.mock_storage_service.save_markdown_content.return_value = "/path/to/new_reanalysis.md"
            self.mock_storage_service.update_analysis.return_value = True

            # --- Act ---
            from src.transcribe import process_existing_transcription
            process_existing_transcription()

            # --- Assert ---
            self.mock_storage_service.get_last_transcription_for_reanalysis.assert_called_once()
            self.mock_analysis_service.analyze_transcription.assert_called_once_with(transcription="This is the existing transcription text from DB")
            self.mock_cli_interface.assemble_analysis_markdown.assert_called_once_with(
                mock_analysis_results, 
                default_title=unittest.mock.ANY 
            )
            self.mock_storage_service.update_analysis.assert_called_once_with(
                recording_id=303,
                llm_model_used='gpt-re-test',
                analysis_markdown=expected_markdown_to_save, # Should match the full markdown passed when stripping doesn't occur due to \\n
                title_for_file="Re-Analysis Title" # Add title_for_file
            )
            self.mock_storage_service.save_markdown_content.assert_called_once_with(
                file_and_h1_title="Re-Analysis Title",
                body_markdown_content=expected_markdown_to_save,
                timestamp_obj=mock_now_reanalysis, 
                type="analysis_reprocessed"
            )
            mock_log_func.info.assert_any_call(f"Interactive re-analysis also saved to new markdown file: /path/to/new_reanalysis.md")

    @patch('src.transcribe.log', new_callable=MagicMock)
    # No longer patching 'open' as process_existing_transcription gets data from DB for re-analysis
    def test_process_existing_transcription_empty_db_transcription(self, mock_log_func):
        """Test re-analysis when the last DB record has an empty transcription (should not proceed to analysis)."""
        with patch('src.transcribe.cli_interface', self.mock_cli_interface), \
             patch('src.transcribe.storage_service', self.mock_storage_service), \
             patch('src.transcribe.analysis_service', self.mock_analysis_service):
            # --- Arrange ---
            mock_empty_transcription_record = {
                'id': 101, 'transcription': "", 'timestamp': datetime.now(), # Empty transcription
                'whisper_model_used': 'base', 'llm_model_used': 'old_llm', 
                'analysis_markdown': '# Old Analysis'
            }
            self.mock_storage_service.get_recordings_without_analysis.return_value = []
            self.mock_storage_service.get_last_transcription_for_reanalysis.return_value = mock_empty_transcription_record
            
            # Set the warning message attribute on the mock_cli_interface
            self.mock_cli_interface.NO_TRANSCRIPTION_FOUND_IN_DB_WARNING = "No suitable transcription found in DB for re-analysis."

            # --- Act ---
            from src.transcribe import process_existing_transcription
            process_existing_transcription()

            # --- Assert ---
            self.mock_storage_service.get_recordings_without_analysis.assert_called_once()
            self.mock_storage_service.get_last_transcription_for_reanalysis.assert_called_once()
            
            # Assert that analyze_transcription is NOT called due to empty transcription string
            self.mock_analysis_service.analyze_transcription.assert_not_called()
            
            # Assert that the specific warning is logged
            mock_log_func.warning.assert_any_call("No suitable transcription found in DB for re-analysis.")
            
            # assemble_analysis_markdown should not be called if analysis_service.analyze_transcription is not called
            self.mock_cli_interface.assemble_analysis_markdown.assert_not_called()
            
            self.mock_storage_service.update_analysis.assert_not_called()
            self.mock_storage_service.save_markdown_content.assert_not_called()

    @patch('src.transcribe.log', new_callable=MagicMock)
    # No longer patching 'open'
    def test_process_existing_transcription_reanalysis_fails(self, mock_log_func):
        """Test re-analysis when analysis_service.analyze_transcription itself indicates failure."""
        with patch('src.transcribe.cli_interface', self.mock_cli_interface), \
             patch('src.transcribe.storage_service', self.mock_storage_service), \
             patch('src.transcribe.analysis_service', self.mock_analysis_service), \
             patch('src.transcribe.datetime', self.mock_datetime): 
            # --- Arrange ---
            mock_record_from_db = {
                'id': 202, 'transcription': "Valid transcription text from DB", 'timestamp': datetime.now(),
                'whisper_model_used': 'base', 'llm_model_used': 'old_llm', 
                'analysis_markdown': '# Old Analysis'
            }
            self.mock_storage_service.get_recordings_without_analysis.return_value = []
            self.mock_storage_service.get_last_transcription_for_reanalysis.return_value = mock_record_from_db

            mock_analysis_results = {'success': False, 'error': 'LLM Error'} 
            self.mock_analysis_service.analyze_transcription.return_value = mock_analysis_results
            
            self.mock_cli_interface.assemble_analysis_markdown.return_value = ("# Re-analysis Failed\\\\n\\\\nError details.", False, "Re-analysis Failed")
            self.mock_cli_interface.REANALYSIS_FAILED_WARNING = "Re-analysis did not succeed."

            # --- Act ---
            from src.transcribe import process_existing_transcription
            process_existing_transcription()

            # --- Assert ---
            self.mock_storage_service.get_last_transcription_for_reanalysis.assert_called_once()
            self.mock_analysis_service.analyze_transcription.assert_called_once_with(transcription="Valid transcription text from DB")
            self.mock_cli_interface.assemble_analysis_markdown.assert_called_once_with(
                mock_analysis_results, 
                default_title=unittest.mock.ANY
            )
            mock_log_func.warning.assert_any_call("Re-analysis did not succeed.")
            self.mock_storage_service.update_analysis.assert_not_called()
            self.mock_storage_service.save_markdown_content.assert_not_called()
            
            # Check for the specific log message about generated markdown
            logged_info_calls = [call_args[0][0] for call_args in mock_log_func.info.call_args_list]
            self.assertTrue(any("Generated markdown (even if not saved as primary for interactive re-analysis)" in call_str for call_str in logged_info_calls))

    # Test related to file reading, which is no longer the primary path for re-analysis.
    # Commenting out as the function process_existing_transcription now primarily uses DB records.
    # @patch('src.transcribe.log', new_callable=MagicMock)
    # @patch('builtins.open', new_callable=unittest.mock.mock_open) # Correct patch target for builtins.open
    # def test_process_existing_transcription_file_read_error(self, mock_open_file, mock_log_func):
    #     with patch('src.transcribe.cli_interface', self.mock_cli_interface), \\
    #          patch('src.transcribe.storage_service', self.mock_storage_service), \\
    #          patch('src.transcribe.analysis_service', self.mock_analysis_service):
    #         # --- Arrange ---
    #         # This test would need to mock cli_interface.select_existing_transcription_file if that path was still active
    #         # For now, assuming that path is deprecated in favor of DB based re-analysis.
    #         # If a file path for re-analysis was still supported, this test would be valid.
    #         # selected_file_path = "/path/to/problematic_file.md"
    #         # self.mock_cli_interface.select_existing_transcription_file.return_value = selected_file_path
    #         # mock_open_file.side_effect = IOError("Cannot read file")

    #         # --- Act ---
    #         from src.transcribe import process_existing_transcription
    #         # process_existing_transcription() # Call with appropriate args if the file path was an arg

    #         # --- Assert ---
    #         # mock_open_file.assert_called_once_with(selected_file_path, 'r', encoding='utf-8')
    #         # mock_log_func.error.assert_any_call(f"Error reading transcription file: Cannot read file")
    #         self.mock_analysis_service.analyze_transcription.assert_not_called()

if __name__ == '__main__':
    unittest.main() 