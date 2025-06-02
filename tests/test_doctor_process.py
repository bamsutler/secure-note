import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime
import os # For os.listdir, os.path.exists mocks if needed directly
from pathlib import Path # Import Path

from src.transcribe import process_existing_transcription
# Import services and cli_interface for type hinting and direct mocking if instances are not passed
from src.storage_service import StorageService
from src.analysis_service import AnalysisService
from src import cli_interface # As a module

class TestDoctorProcess(unittest.TestCase):

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('src.transcribe.log') # Mock logger
    @patch('src.transcribe.analysis_service') # Mock the global analysis_service instance
    @patch('src.transcribe.cli_interface') # Mock the cli_interface module
    @patch('src.transcribe.storage_service') # Mock the global storage_service instance
    def test_regenerates_missing_file_using_db_title(self, mock_storage_service_global, mock_cli_interface_module,
                                                   mock_analysis_service_global, mock_log, 
                                                   mock_os_listdir, mock_os_path_exists):
        # Arrange
        mock_ss_instance = mock_storage_service_global
        mock_ss_instance.get_recordings_without_analysis.return_value = []
        mock_ss_instance.get_last_transcription_for_reanalysis.return_value = None

        record_id = 1
        record_ts = datetime(2023, 10, 26, 10, 0, 0)
        db_title = f"DB Title for ID {record_id}"
        db_body = "This is the analysis body from DB."
        records_with_analysis = [{
            'id': record_id, 'timestamp': record_ts, 'analysis_markdown': db_body, 
            'title_for_file': db_title, 'llm_model_used': 'test_model'
        }]
        mock_ss_instance.get_records_with_analysis.return_value = records_with_analysis

        # New setup: Use a concrete pathlib.Path object
        mock_markdown_dir_name = "mock_markdown_dir" 
        # Ensure Path is available (it's imported at the top of the file)
        mock_ss_instance.markdown_save_path = Path(mock_markdown_dir_name)

        expected_filename_str = f"{record_ts.strftime('%Y%m%d_%H%M%S')}_{db_title.replace(' ', '_')}.md"
        mock_ss_instance._generate_filename.return_value = expected_filename_str
        mock_os_listdir.return_value = []

        def print_and_return_false(*args, **kwargs):
            print(f"DIAGNOSTIC: mock_os_path_exists INVOCATION with args: {args}, kwargs: {kwargs}")
            return False
        mock_os_path_exists.side_effect = print_and_return_false

        process_existing_transcription()

        # --- Start of Diagnostic Assertions ---
        # 1. Verify os.listdir call
        mock_os_listdir.assert_called_once_with(mock_ss_instance.markdown_save_path)

        # 2. Check if the generic exception for os.listdir was hit
        expected_listdir_err_log_start = f"Error listing files in {str(mock_ss_instance.markdown_save_path)}:"
        listdir_error_logged = False
        for call_item in mock_log.error.call_args_list:
            args, _ = call_item
            if args and isinstance(args[0], str) and args[0].startswith(expected_listdir_err_log_start):
                listdir_error_logged = True
                print(f"DIAGNOSTIC: Logged error related to listdir: {args[0]}") # For test output
                break
        self.assertFalse(listdir_error_logged, 
                         f"os.listdir block appears to have raised an exception. Logged errors: {mock_log.error.call_args_list}")

        # NEW DIAGNOSTIC: Check if the 'continue' for missing critical data was hit
        missing_data_log_warning_found = False
        # The log message uses record_id, so we need to make sure it's defined if the loop doesn't run
        # However, get_records_with_analysis ensures there's at least one record for this test.
        expected_missing_data_log_substring = "is missing critical data" 
        for call_item in mock_log.warning.call_args_list:
            args, _ = call_item
            if args and isinstance(args[0], str) and expected_missing_data_log_substring in args[0]:
                missing_data_log_warning_found = True
                print(f"DIAGNOSTIC: Logged warning for missing critical data: {args[0]}") # For test output
                break
        self.assertFalse(missing_data_log_warning_found,
                         "Loop may have 'continue'd due to missing critical data in a record, preventing further checks for that record.")

        # 3. Verify storage_service._generate_filename call
        mock_ss_instance._generate_filename.assert_called_with(
            title_prefix=db_title, timestamp_obj=record_ts, extension=".md"
        )
        # expected_filename_str is the result of the above call, as per mock setup

        # 4. Verify os.path.exists call for the recovered file path
        self.assertTrue(mock_os_path_exists.called, "os.path.exists was not called after listdir and critical data checks.")
        
        # Construct the expected path string that os.path.exists should have been called with
        # Our mock for __truediv__ means str(markdown_dir / "recovered" / expected_filename_str) 
        # should be "mock_markdown_dir/recovered/THE_GENERATED_FILENAME"
        expected_path_str_for_exists_check = f"{str(mock_ss_instance.markdown_save_path)}/recovered/{expected_filename_str}"

        called_with_expected_path = False
        for call_obj in mock_os_path_exists.call_args_list:
            args, kwargs = call_obj # call_obj can be call(), call(arg), call(arg, kwarg=...)
            actual_arg_passed = None
            if args: # Path is the first positional argument
                actual_arg_passed = args[0]
            
            if actual_arg_passed is not None and str(actual_arg_passed) == expected_path_str_for_exists_check:
                called_with_expected_path = True
                break
        
        self.assertTrue(called_with_expected_path, 
                        f"os.path.exists was not called with the expected path string '{expected_path_str_for_exists_check}'. Actual calls: {mock_os_path_exists.call_args_list}")
        # --- End of Diagnostic Assertions ---

        mock_ss_instance.save_markdown_content.assert_called_once_with(
            file_and_h1_title=db_title, body_markdown_content=db_body,
            timestamp_obj=record_ts, type="analysis_recovered"
        )
        mock_log.info.assert_any_call(f"Attempting to regenerate file for Record ID: {record_id} using stored title '{db_title}'...")

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('src.transcribe.log')
    @patch('src.transcribe.analysis_service')
    @patch('src.transcribe.cli_interface')
    @patch('src.transcribe.storage_service')
    def test_skips_if_file_exists_in_main_directory(self, mock_storage_service_global, mock_cli_interface_module,
                                                    mock_analysis_service_global, mock_log, 
                                                    mock_os_listdir, mock_os_path_exists):
        mock_ss_instance = mock_storage_service_global
        mock_ss_instance.get_recordings_without_analysis.return_value = []
        mock_ss_instance.get_last_transcription_for_reanalysis.return_value = None
        mock_ss_instance.markdown_save_path = Path("mock_markdown_dir")

        record_id = 1
        record_ts = datetime(2023, 10, 26, 10, 0, 0)
        db_title = "Existing File Title"
        records_with_analysis = [{
            'id': record_id, 'timestamp': record_ts, 
            'analysis_markdown': "body", 'title_for_file': db_title
        }]
        mock_ss_instance.get_records_with_analysis.return_value = records_with_analysis

        expected_filename = f"{record_ts.strftime('%Y%m%d_%H%M%S')}_{db_title.replace(' ', '_')}.md"
        mock_ss_instance._generate_filename.return_value = expected_filename
        mock_os_listdir.return_value = [expected_filename]

        process_existing_transcription()

        mock_ss_instance.save_markdown_content.assert_not_called()
        mock_log.info.assert_any_call("No missing markdown files found for analyzed records.")
        mock_os_path_exists.assert_not_called()

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('src.transcribe.log')
    @patch('src.transcribe.analysis_service')
    @patch('src.transcribe.cli_interface')
    @patch('src.transcribe.storage_service')
    def test_skips_if_file_exists_in_recovered_directory(self, mock_storage_service_global, mock_cli_interface_module,
                                                        mock_analysis_service_global, mock_log, 
                                                        mock_os_listdir, mock_os_path_exists):
        mock_ss_instance = mock_storage_service_global
        mock_ss_instance.get_recordings_without_analysis.return_value = []
        mock_ss_instance.get_last_transcription_for_reanalysis.return_value = None
        mock_ss_instance.markdown_save_path = Path("mock_markdown_dir")

        record_id = 1
        record_ts = datetime(2023, 10, 26, 10, 0, 0)
        db_title = "Recovered File Title"
        records_with_analysis = [{
            'id': record_id, 'timestamp': record_ts, 
            'analysis_markdown': "body", 'title_for_file': db_title
        }]
        mock_ss_instance.get_records_with_analysis.return_value = records_with_analysis

        expected_filename = f"{record_ts.strftime('%Y%m%d_%H%M%S')}_{db_title.replace(' ', '_')}.md"
        mock_ss_instance._generate_filename.return_value = expected_filename
        mock_os_listdir.return_value = []
        
        def print_and_return_true(*args, **kwargs):
            print(f"DIAGNOSTIC (recovered_directory_test): mock_os_path_exists INVOCATION with args: {args}, kwargs: {kwargs}")
            return True
        mock_os_path_exists.side_effect = print_and_return_true

        process_existing_transcription()

        mock_ss_instance.save_markdown_content.assert_not_called()
        self.assertTrue(mock_os_path_exists.called)
        expected_path_to_check = str(Path("mock_markdown_dir") / "recovered" / expected_filename)
        
        called_with_expected_path = False
        for call_obj in mock_os_path_exists.call_args_list:
            args_call, _ = call_obj
            if args_call and str(args_call[0]) == expected_path_to_check:
                called_with_expected_path = True
                break
        self.assertTrue(called_with_expected_path, 
                        f"os.path.exists was not called with '{expected_path_to_check}'. Calls: {mock_os_path_exists.call_args_list}")

    # TODO: Add tests for the interactive re-analysis part.
    # TODO: Add tests for queueing of unanalyzed records.

    @patch('src.transcribe.datetime') # To control datetime.now() for consistent filenames/titles
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('src.transcribe.log')
    @patch('src.transcribe.analysis_service') 
    @patch('src.transcribe.cli_interface') 
    @patch('src.transcribe.storage_service') 
    def test_interactive_reanalysis_updates_db_and_saves_file(self, mock_storage_service_global, mock_cli_interface_module,
                                                              mock_analysis_service_global, mock_log, 
                                                              mock_os_listdir, mock_os_path_exists, mock_datetime):
        # Arrange
        mock_ss_instance = mock_storage_service_global
        mock_as_instance = mock_analysis_service_global # Direct use of the global mock
        mock_cli_module = mock_cli_interface_module # Direct use

        # Part 1: Skip unanalyzed records and missing file check for this test
        mock_ss_instance.get_recordings_without_analysis.return_value = []
        mock_ss_instance.get_records_with_analysis.return_value = []

        # Setup for interactive re-analysis part
        last_rec_id = 10
        last_rec_ts_original = datetime(2023, 1, 1, 12, 0, 0)
        last_rec_transcription = "This is the last transcription text."
        last_rec_data = {
            'id': last_rec_id, 
            'timestamp': last_rec_ts_original, 
            'transcription': last_rec_transcription,
            'whisper_model_used': 'tiny',
            'llm_model_used': 'old_llm',
            'analysis_markdown': 'Old analysis body.',
            'title_for_file': 'Old Title'
        }
        mock_ss_instance.get_last_transcription_for_reanalysis.return_value = last_rec_data

        # Mock current time for consistent naming in re-analysis
        current_time_for_reanalysis = datetime(2023, 10, 27, 15, 0, 0)
        mock_datetime.now.return_value = current_time_for_reanalysis

        # Mock analysis service and CLI interface for re-analysis
        mock_analysis_results = {'summary': 'Re-analyzed summary', 'model_used': 'new_llm_mock'}
        mock_as_instance.analyze_transcription.return_value = mock_analysis_results
        
        expected_reanalysis_title = f"Re-analysis of ID {last_rec_id} ({current_time_for_reanalysis.strftime('%Y%m%d_%H%M%S')})"
        reanalyzed_body_for_db = "Re-analyzed body for DB (H1 stripped)"
        reanalyzed_full_markdown = f"# {expected_reanalysis_title}\n\n{reanalyzed_body_for_db}"

        mock_cli_module.assemble_analysis_markdown.return_value = (
            reanalyzed_full_markdown, # final_markdown_to_save_reanalysis
            True,                     # is_reanalysis_successful
            expected_reanalysis_title # title_from_reanalysis
        )

        # Act
        process_existing_transcription()

        # Assert
        mock_as_instance.analyze_transcription.assert_called_once_with(transcription=last_rec_transcription)
        mock_cli_module.assemble_analysis_markdown.assert_called_once_with(
            mock_analysis_results,
            default_title=f"Re-analysis of ID {last_rec_id} ({current_time_for_reanalysis.strftime('%Y%m%d_%H%M%S')})"
        )
        
        mock_ss_instance.update_analysis.assert_called_once_with(
            recording_id=last_rec_id,
            llm_model_used=mock_analysis_results['model_used'],
            analysis_markdown=reanalyzed_body_for_db, # H1 stripped body
            title_for_file=expected_reanalysis_title
        )
        mock_ss_instance.save_markdown_content.assert_called_once_with(
            file_and_h1_title=expected_reanalysis_title,
            body_markdown_content=reanalyzed_body_for_db, # H1 stripped body
            timestamp_obj=current_time_for_reanalysis,
            type="analysis_reprocessed"
        )

    @patch('src.transcribe.processing_queue') # Mock the queue
    @patch('src.transcribe.log')
    @patch('src.transcribe.cli_interface')
    @patch('src.transcribe.storage_service')
    def test_queues_unanalyzed_records_if_confirmed(self, mock_storage_service_global, mock_cli_interface_module,
                                                     mock_log, mock_processing_queue):
        # Arrange
        mock_ss_instance = mock_storage_service_global
        mock_cli_module = mock_cli_interface_module

        # Skip missing file check and interactive re-analysis for this test
        mock_ss_instance.get_records_with_analysis.return_value = []
        mock_ss_instance.get_last_transcription_for_reanalysis.return_value = None

        # Setup records needing analysis
        unanalyzed_rec1_ts = datetime(2023, 5, 1, 10, 0, 0)
        unanalyzed_rec1 = {'id': 101, 'timestamp': unanalyzed_rec1_ts, 'transcription': 'Rec 1 trans', 'whisper_model_used': 'tiny'}
        unanalyzed_rec2_ts = datetime(2023, 5, 2, 11, 0, 0)
        unanalyzed_rec2 = {'id': 102, 'timestamp': unanalyzed_rec2_ts, 'transcription': 'Rec 2 trans', 'whisper_model_used': 'base'}
        mock_ss_instance.get_recordings_without_analysis.return_value = [unanalyzed_rec1, unanalyzed_rec2]

        mock_cli_module.confirm_process_unanalyzed.return_value = True # User confirms

        # Act
        process_existing_transcription()

        # Assert
        mock_cli_module.confirm_process_unanalyzed.assert_called_once_with(2) # Called with count of records
        
        expected_calls_to_queue = [
            call(('analyze_existing', unanalyzed_rec1)),
            call(('analyze_existing', unanalyzed_rec2))
        ]
        mock_processing_queue.put.assert_has_calls(expected_calls_to_queue, any_order=False)
        self.assertEqual(mock_processing_queue.put.call_count, 2)
        mock_log.info.assert_any_call("All identified records have been queued for background analysis.")

    @patch('src.transcribe.processing_queue')
    @patch('src.transcribe.log')
    @patch('src.transcribe.cli_interface')
    @patch('src.transcribe.storage_service')
    def test_does_not_queue_if_user_declines(self, mock_storage_service_global, mock_cli_interface_module,
                                            mock_log, mock_processing_queue):
        # Arrange
        mock_ss_instance = mock_storage_service_global
        mock_cli_module = mock_cli_interface_module
        mock_ss_instance.get_records_with_analysis.return_value = []
        mock_ss_instance.get_last_transcription_for_reanalysis.return_value = None

        unanalyzed_rec = {'id': 201, 'timestamp': datetime.now(), 'transcription': 'To be skipped'}
        mock_ss_instance.get_recordings_without_analysis.return_value = [unanalyzed_rec]
        
        mock_cli_module.confirm_process_unanalyzed.return_value = False # User declines

        # Act
        process_existing_transcription()

        # Assert
        mock_cli_module.confirm_process_unanalyzed.assert_called_once_with(1)
        mock_processing_queue.put.assert_not_called()
        mock_log.info.assert_any_call("Skipping batch analysis of unanalyzed records.")

if __name__ == '__main__':
    unittest.main() 