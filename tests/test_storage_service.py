import unittest
from unittest.mock import patch, MagicMock, call
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.storage_service import StorageService
from src.config_service import ConfigurationService

class TestStorageService(unittest.TestCase):

    # @patch('src.storage_service.config_service') # Removed - config is now injected
    def setUp(self):
        self.mock_config_service_instance = MagicMock(spec=ConfigurationService)

        self.db_name_for_setup = "file:test_storage_service_db?mode=memory&cache=shared"
        
        # --- Mock for markdown_save_path --- 
        self.mock_markdown_save_path_obj = MagicMock(spec=Path, name="MockMarkdownBasePath")
        self.mock_markdown_save_path_obj.mkdir = MagicMock(name="mkdir_base_markdown_path")
        self.mock_markdown_save_path_obj.__str__.return_value = "mock_markdown_notes"

        # Mocks for subdirectories (recovered, reprocessed)
        self.mock_recovered_path_obj = MagicMock(spec=Path, name="MockRecoveredPath")
        self.mock_recovered_path_obj.mkdir = MagicMock(name="mkdir_recovered_subpath")
        self.mock_recovered_path_obj.__str__.return_value = "mock_markdown_notes/recovered"

        self.mock_reprocessed_path_obj = MagicMock(spec=Path, name="MockReprocessedPath")
        self.mock_reprocessed_path_obj.mkdir = MagicMock(name="mkdir_reprocessed_subpath")
        self.mock_reprocessed_path_obj.__str__.return_value = "mock_markdown_notes/reprocessed"

        # Behavior for self.mock_markdown_save_path_obj / "subdir" or filename
        # This needs to handle both creating subdirectories AND creating final file paths
        def markdown_truediv_side_effect(other):
            other_str = str(other)
            if other_str == "recovered":
                # When self.markdown_save_path / "recovered" is called,
                # it should return self.mock_recovered_path_obj.
                # This mock_recovered_path_obj will then have its own __truediv__ for the filename.
                return self.mock_recovered_path_obj
            elif other_str == "reprocessed":
                return self.mock_reprocessed_path_obj
            else: 
                # This case is for self.markdown_save_path / "filename.md"
                final_file_path_mock = MagicMock(spec=Path, name=f"MockFinalFilePath_{other_str}")
                final_file_path_mock.__str__.return_value = str(self.mock_markdown_save_path_obj) + "/" + other_str
                # final_file_path_mock doesn't need mkdir, but its parent (self.mock_markdown_save_path_obj) does.
                return final_file_path_mock
        self.mock_markdown_save_path_obj.__truediv__ = MagicMock(side_effect=markdown_truediv_side_effect, name="truediv_base_markdown")

        # Configure __truediv__ for subfolder mocks to handle appending filename
        def subdir_truediv_side_effect(base_mock_path_obj, other_filename_str):
            final_file_path_mock = MagicMock(spec=Path, name=f"MockFinalFilePath_{base_mock_path_obj._extract_mock_name()}_{other_filename_str}")
            final_file_path_mock.__str__.return_value = str(base_mock_path_obj) + "/" + str(other_filename_str)
            return final_file_path_mock

        self.mock_recovered_path_obj.__truediv__ = MagicMock(
            side_effect=lambda other: subdir_truediv_side_effect(self.mock_recovered_path_obj, other),
            name="truediv_recovered"
        )
        self.mock_reprocessed_path_obj.__truediv__ = MagicMock(
            side_effect=lambda other: subdir_truediv_side_effect(self.mock_reprocessed_path_obj, other),
            name="truediv_reprocessed"
        )

        # --- Mock for temp_path --- 
        self.mock_temp_path_obj = MagicMock(spec=Path, name="MockTempPath")
        self.mock_temp_path_obj.mkdir = MagicMock(name="mkdir_temp_path")
        self.mock_temp_path_obj.__str__.return_value = "mock_audio_temp"
        self.mock_temp_path_obj.__truediv__ = MagicMock( # Basic truediv for temp path for joining filename
            side_effect=lambda other: subdir_truediv_side_effect(self.mock_temp_path_obj, other),
            name="truediv_temp"
        )

        def mock_config_get_side_effect(*args, **kwargs):
            if args == ('database', 'name'):
                return self.db_name_for_setup
            if args == ('paths', 'markdown_save'):
                return str(self.mock_markdown_save_path_obj)
            if args == ('paths', 'temp_path'):
                return str(self.mock_temp_path_obj)
            return MagicMock() # Default for other config gets

        self.mock_config_service_instance.get.side_effect = mock_config_get_side_effect
        
        # self.storage_service = StorageService() # Removed - instantiated in test or not at all if test doesn't need it
        # Reset mkdir mocks that were called during StorageService.__init__
        # These are on self.mock_..._path_obj which are NOT the same as what SUT sees if Path is mocked
        self.mock_markdown_save_path_obj.mkdir.reset_mock()
        self.mock_temp_path_obj.mkdir.reset_mock()
        # mkdir for recovered/reprocessed are NOT called in __init__

    def tearDown(self):
        # For shared in-memory DB, need to manually close the connection to allow it to be wiped for next test if truly isolated.
        # However, SQLite in-memory dbs are often connection-specific. If db_name_for_setup is connection-specific in-memory,
        # then new connection in StorageService() in setUp implies new DB.
        # If "file:...&cache=shared", it persists until all connections are closed.
        # To be safe, explicitly try to clean up tables if one connection object were reused (not the case here).
        conn = sqlite3.connect(self.db_name_for_setup, uri=True)
        try:
            conn.execute("DROP TABLE IF EXISTS recordings")
            conn.commit()
        finally:
            conn.close()
    
    def test_save_and_get_recording_transcription(self):
        """Test saving a new recording with transcription and retrieving it."""
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        ts = datetime.now()
        audio_bytes = b'fake_audio_data'
        samplerate = 16000
        whisper_model = "tiny"
        transcription_text = "This is a test transcription."

        record_id = storage_service.save_initial_recording_and_transcription(
            timestamp=ts,
            audio_wav_bytes=audio_bytes,
            samplerate=samplerate,
            whisper_model_used=whisper_model,
            transcription=transcription_text
        )
        self.assertIsNotNone(record_id, "save_initial_recording_and_transcription should return a record ID.")

        retrieved_record = storage_service.get_recording_by_id(record_id)
        self.assertIsNotNone(retrieved_record, "Should retrieve the saved record by ID.")
        self.assertEqual(retrieved_record['timestamp'], ts.isoformat())
        self.assertEqual(retrieved_record['audio_wav'], audio_bytes)
        self.assertEqual(retrieved_record['samplerate'], samplerate)
        self.assertEqual(retrieved_record['whisper_model_used'], whisper_model)
        self.assertEqual(retrieved_record['transcription'], transcription_text)
        self.assertIsNone(retrieved_record['llm_model_used']) # Initially null
        self.assertIsNone(retrieved_record['analysis_markdown']) # Initially null

    def test_get_recording_not_found(self):
        """Test retrieving a non-existent recording by ID."""
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        retrieved_record = storage_service.get_recording_by_id(99999) # Non-existent ID
        self.assertIsNone(retrieved_record)

    def test_update_and_get_analysis(self):
        """Test saving an initial recording and then updating it with analysis."""
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        ts = datetime.now()
        record_id = storage_service.save_initial_recording_and_transcription(
            timestamp=ts,
            audio_wav_bytes=b'audio',
            samplerate=16000,
            whisper_model_used="base",
            transcription="Initial text."
        )
        self.assertIsNotNone(record_id)

        llm_model = "test_llm_v1"
        analysis_md = "# Analysis\nThis is the detailed analysis."
        
        update_success = storage_service.update_analysis(
            recording_id=record_id,
            llm_model_used=llm_model,
            analysis_markdown=analysis_md,
            title_for_file="Test Title for Analysis ID" + str(record_id) # Added title
        )
        self.assertTrue(update_success, "update_analysis should return True on success.")

        retrieved_record = storage_service.get_recording_by_id(record_id)
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record['llm_model_used'], llm_model)
        self.assertEqual(retrieved_record['analysis_markdown'], analysis_md)
        self.assertEqual(retrieved_record['transcription'], "Initial text.") # Check transcription is still there

    def test_update_analysis_for_non_existent_record(self):
        """Test updating analysis for a record ID that does not exist."""
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        update_success = storage_service.update_analysis(
            recording_id=88888, # Non-existent
            llm_model_used="test_model",
            analysis_markdown="some analysis",
            title_for_file="Non Existent Title" # Added title
        )
        self.assertFalse(update_success, "update_analysis should return False for non-existent record ID.")

    def test_get_last_transcription_for_reanalysis(self):
        """Test fetching the last transcription eligible for re-analysis."""
        # Clear any existing records (new in-memory DB for each test method due to setUp)
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)

        # Scenario 1: No records
        self.assertIsNone(storage_service.get_last_transcription_for_reanalysis())

        # Scenario 2: Record with no transcription
        ts1 = datetime.now() - timedelta(minutes=10)
        id1 = storage_service.save_initial_recording_and_transcription(ts1, b'a1', 16000, 't1', None) # No transcription
        self.assertIsNone(storage_service.get_last_transcription_for_reanalysis())
        
        # Scenario 3: Record with empty transcription
        ts2 = datetime.now() - timedelta(minutes=8)
        id2 = storage_service.save_initial_recording_and_transcription(ts2, b'a2', 16000, 't2', "") # Empty transcription
        self.assertIsNone(storage_service.get_last_transcription_for_reanalysis())

        # Scenario 4: A valid record
        ts3 = datetime.now() - timedelta(minutes=5)
        transcription3 = "This is a valid transcription for record 3."
        id3 = storage_service.save_initial_recording_and_transcription(ts3, b'a3', 16000, 'w3', transcription3)
        last_record = storage_service.get_last_transcription_for_reanalysis()
        self.assertIsNotNone(last_record)
        self.assertEqual(last_record['id'], id3)
        self.assertEqual(last_record['transcription'], transcription3)

        # Scenario 5: Another valid record, should fetch this newer one
        ts4 = datetime.now() - timedelta(minutes=2)
        transcription4 = "This is a newer valid transcription for record 4."
        analysis4_md = "Pre-existing analysis for record 4"
        id4 = storage_service.save_initial_recording_and_transcription(ts4, b'a4', 16000, 'w4', transcription4)
        storage_service.update_analysis(id4, "llm_prev", analysis4_md, title_for_file="Test Title for Reanalysis ID4")
        last_record_after_update = storage_service.get_last_transcription_for_reanalysis()
        self.assertIsNotNone(last_record_after_update)
        self.assertEqual(last_record_after_update['id'], id4)
        self.assertEqual(last_record_after_update['transcription'], transcription4)
        self.assertEqual(last_record_after_update['analysis_markdown'], analysis4_md) # Check analysis is also fetched

    def test_get_recordings_without_analysis(self):
        """Test fetching records that have a transcription but no analysis."""
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        # Scenario 1: No records
        self.assertEqual(storage_service.get_recordings_without_analysis(), [])

        # Record 1: No transcription
        ts1 = datetime.now() - timedelta(days=1, minutes=10)
        id1 = storage_service.save_initial_recording_and_transcription(ts1, b'a1', 16000, 'w1', None)
        
        # Record 2: With transcription, no analysis (should be fetched)
        ts2 = datetime.now() - timedelta(days=1, minutes=5)
        transcription2 = "Record 2 needs analysis."
        id2 = storage_service.save_initial_recording_and_transcription(ts2, b'a2', 16000, 'w2', transcription2)

        # Record 3: With transcription and analysis (should NOT be fetched)
        ts3 = datetime.now() - timedelta(days=1, minutes=1)
        transcription3 = "Record 3 has analysis."
        id3 = storage_service.save_initial_recording_and_transcription(ts3, b'a3', 16000, 'w3', transcription3)
        storage_service.update_analysis(id3, "llm_done", "Analysis complete for record 3.", title_for_file="Test Title for Record 3")

        # Record 4: Empty transcription (should NOT be fetched)
        ts4 = datetime.now() - timedelta(days=1, minutes=15)
        id4 = storage_service.save_initial_recording_and_transcription(ts4, b'a4', 16000, 'w4', "")
        
        # Record 5: With transcription, analysis is an empty string (should be fetched as missing analysis)
        ts5 = datetime.now() - timedelta(days=1, minutes=3)
        transcription5 = "Record 5 needs analysis (analysis is empty string)."
        id5 = storage_service.save_initial_recording_and_transcription(ts5, b'a5', 16000, 'w5', transcription5)
        storage_service.update_analysis(id5, "llm_attempted_empty", "", title_for_file="Test Title for Record 5")

        unanalyzed_records = storage_service.get_recordings_without_analysis()
        self.assertEqual(len(unanalyzed_records), 2)
        
        unanalyzed_ids = [r['id'] for r in unanalyzed_records]
        self.assertIn(id2, unanalyzed_ids)
        self.assertIn(id5, unanalyzed_ids)
        
        for record in unanalyzed_records:
            if record['id'] == id2:
                self.assertEqual(record['transcription'], transcription2)
                self.assertEqual(record['timestamp'], ts2) # Timestamps are converted back to datetime by the method
            elif record['id'] == id5:
                self.assertEqual(record['transcription'], transcription5)
                self.assertEqual(record['timestamp'], ts5)

    def test_init_db_adds_title_for_file_column_and_backfills(self):
        # This test needs its own isolated DB to check schema creation and backfill        
        db_file_path_for_test = "file:init_db_test?mode=memory&cache=shared"

        # 1. Setup: Create DB with old schema + data
        conn_setup = sqlite3.connect(db_file_path_for_test, uri=True)
        cursor_setup = conn_setup.cursor()
        cursor_setup.execute("DROP TABLE IF EXISTS recordings") # Clean slate
        cursor_setup.execute("""
        CREATE TABLE recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, audio_wav BLOB NOT NULL,
            samplerate INTEGER NOT NULL, whisper_model_used TEXT NOT NULL, transcription TEXT,
            llm_model_used TEXT, analysis_markdown TEXT
        )""")
        ts1_iso = (datetime.now() - timedelta(days=1)).isoformat()
        cursor_setup.execute("INSERT INTO recordings (timestamp, audio_wav, samplerate, whisper_model_used, transcription, analysis_markdown) VALUES (?, ?, ?, ?, ?, ?)",
                           (ts1_iso, b'audio1', 16000, 'tiny', 'trans1', 'body1'))
        id1 = cursor_setup.lastrowid
        ts2_iso = (datetime.now() - timedelta(days=2)).isoformat()
        cursor_setup.execute("INSERT INTO recordings (timestamp, audio_wav, samplerate, whisper_model_used, transcription, analysis_markdown) VALUES (?, ?, ?, ?, ?, ?)",
                           (ts2_iso, b'audio2', 16000, 'tiny', 'trans2', None)) # Changed NULL to None
        conn_setup.commit()
        # conn_setup.close() # Defer closing this connection

        # 2. Action: Instantiate StorageService, targeting this specific DB
        # Mock config_service to point to this DB for the StorageService instance
        mock_config = MagicMock(spec=ConfigurationService)
        def temp_config_get_side_effect(section, key, default=None):
            if section == 'database' and key == 'name': return db_file_path_for_test
            if section == 'paths' and key == 'markdown_save': return self.mock_markdown_save_path_obj
            if section == 'paths' and key == 'temp_path': return self.mock_temp_path_obj
            return default
        mock_config.get.side_effect = temp_config_get_side_effect

        # with patch('src.storage_service.config_service', mock_config): # This is no longer needed due to DI
        with patch('src.storage_service.Path') as MockPathForInitTest:
            mock_path_init_instance = MagicMock()
            # SUT calls .parent.mkdir() and .mkdir() on Path objects in __init__
            mock_path_init_instance.parent.mkdir.return_value = None 
            mock_path_init_instance.mkdir.return_value = None
            MockPathForInitTest.return_value = mock_path_init_instance
            # Instantiate StorageService directly with the mocked config
            service_for_init_test = StorageService(config_service_instance=mock_config)

        # 3. Verification
        conn_verify = sqlite3.connect(db_file_path_for_test, uri=True)
        conn_verify.row_factory = sqlite3.Row
        cursor_verify = conn_verify.cursor()
        cursor_verify.execute("PRAGMA table_info(recordings)")
        columns = {row['name'] for row in cursor_verify.fetchall()}
        self.assertIn('title_for_file', columns)

        # Check backfill for record 1
        cursor_verify.execute("SELECT * FROM recordings WHERE id = ?", (id1,))
        row1_for_verify = cursor_verify.fetchone()
        self.assertIsNotNone(row1_for_verify, f"Record with ID {id1} was not found during verification phase.")
        record1_title = row1_for_verify['title_for_file']

        dt_obj1 = datetime.fromisoformat(ts1_iso)
        expected_dummy_title1 = f"Analysis for ID {id1} ({dt_obj1.strftime('%Y%m%d_%H%M%S')})"
        self.assertEqual(record1_title, expected_dummy_title1)

        # Check record 2 (no analysis_markdown, so title_for_file should be NULL)
        cursor_verify.execute("SELECT title_for_file FROM recordings WHERE timestamp = ?", (ts2_iso,))
        record2_title = cursor_verify.fetchone()['title_for_file']
        self.assertIsNone(record2_title)
        conn_verify.close()
        conn_setup.close() # Close the initial setup connection now

    def test_update_analysis_saves_title_for_file(self):
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        ts = datetime.now()
        record_id = storage_service.save_initial_recording_and_transcription(
            timestamp=ts, audio_wav_bytes=b'audio', samplerate=16000,
            whisper_model_used="base", transcription="Initial text."
        )
        self.assertIsNotNone(record_id)

        llm_model = "test_llm_v2"
        analysis_md_body = "This is the analysis body."
        file_title = "My Specific File Title"
        
        update_success = storage_service.update_analysis(
            recording_id=record_id,
            llm_model_used=llm_model,
            analysis_markdown=analysis_md_body,
            title_for_file=file_title # New argument
        )
        self.assertTrue(update_success)

        retrieved_record = storage_service.get_recording_by_id(record_id)
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record['llm_model_used'], llm_model)
        self.assertEqual(retrieved_record['analysis_markdown'], analysis_md_body)
        self.assertEqual(retrieved_record['title_for_file'], file_title) # Verify new field

    def test_get_records_with_analysis_retrieves_title_for_file(self):
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        ts = datetime.now()
        record_id = storage_service.save_initial_recording_and_transcription(
            timestamp=ts, audio_wav_bytes=b'audio', samplerate=16000,
            whisper_model_used="base", transcription="Text for analysis."
        )
        file_title = "Analysis Title For Get Test"
        analysis_body = "Analysis body content."
        storage_service.update_analysis(record_id, "llm_model_get", analysis_body, file_title)

        records = storage_service.get_records_with_analysis()
        self.assertEqual(len(records), 1)
        retrieved_record = records[0]
        self.assertEqual(retrieved_record['id'], record_id)
        self.assertEqual(retrieved_record['analysis_markdown'], analysis_body)
        self.assertEqual(retrieved_record['title_for_file'], file_title)
        self.assertIsInstance(retrieved_record['timestamp'], datetime) # Check timestamp conversion

    @patch('src.storage_service.Path')
    @patch('builtins.open')
    def test_save_markdown_content_creates_h1_and_subfolders(self, mock_open, MockPath):
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
    
        # print(f"DEBUG TEST: MockPath injected: {MockPath} (ID: {id(MockPath)})")
    
        # Path objects that will be returned by the side_effect for Path() calls in SUT __init__
        sut_init_markdown_path = MagicMock(spec=Path, name="sut_init_markdown_path_returned_by_side_effect")
        sut_init_markdown_path.mkdir = MagicMock(name="sut_init_markdown_path.mkdir")
        sut_init_temp_path = MagicMock(spec=Path, name="sut_init_temp_path_returned_by_side_effect")
        sut_init_temp_path.mkdir = MagicMock(name="sut_init_temp_path.mkdir")
        sut_init_db_path = MagicMock(spec=Path, name="sut_init_db_path_returned_by_side_effect")
        sut_init_db_path.parent = MagicMock(spec=Path, name="sut_init_db_path.parent")
        sut_init_db_path.parent.mkdir = MagicMock(name="sut_init_db_path.parent.mkdir")

        def path_constructor_side_effect(path_arg_str):
            # print(f"DEBUG TEST - Path CONSTRUCTOR side_effect: CALLED with '{path_arg_str}'")
            if path_arg_str == str(self.mock_markdown_save_path_obj):
                # print(f"DEBUG TEST - Path CONSTRUCTOR side_effect: returning sut_init_markdown_path for '{path_arg_str}'")
                return sut_init_markdown_path
            elif path_arg_str == str(self.mock_temp_path_obj):
                # print(f"DEBUG TEST - Path CONSTRUCTOR side_effect: returning sut_init_temp_path for '{path_arg_str}'")
                return sut_init_temp_path
            elif path_arg_str == self.db_name_for_setup and ":memory:" not in self.db_name_for_setup:
                # print(f"DEBUG TEST - Path CONSTRUCTOR side_effect: returning sut_init_db_path for '{path_arg_str}'")
                return sut_init_db_path
            # print(f"DEBUG TEST - Path CONSTRUCTOR side_effect: no match for '{path_arg_str}', returning NEW mock")
            return MagicMock(spec=Path, name=f"Path_SideEffect_Fallback_{path_arg_str}")

        MockPath.side_effect = path_constructor_side_effect
        # print(f"DEBUG TEST: MockPath.side_effect ASSIGNED.")

        # print(f"DEBUG TEST: About to instantiate StorageService...")
        storage_service = StorageService(config_service_instance=self.mock_config_service_instance)
        # print(f"DEBUG TEST: StorageService instantiated.")
        
        self.assertIs(storage_service.markdown_save_path, sut_init_markdown_path, "SUT markdown_save_path is not the expected mock object")
        self.assertIs(storage_service.temp_path, sut_init_temp_path, "SUT temp_path is not the expected mock object")
        if storage_service.db_path is not None:
            self.assertIs(storage_service.db_path, sut_init_db_path, "SUT db_path is not the expected mock object")
        else:
            self.assertTrue(":memory:" in self.db_name_for_setup, "db_path is None but db_name_for_setup was not :memory:")

        # Reset mkdir mocks that were called during StorageService.__init__
        # These are on the sut_init_* path objects.
        sut_init_markdown_path.mkdir.reset_mock()
        sut_init_temp_path.mkdir.reset_mock()
        if storage_service.db_path is not None: # Only if db_path object was created and had .parent.mkdir called
            sut_init_db_path.parent.mkdir.reset_mock()

        # --- Setup for save_markdown_content calls ---
        file_title = "My Test Note"
        ts_obj = datetime(2023, 10, 26, 14, 30, 0)
        expected_filename_part = "20231026_143000_My_Test_Note.md"
        body_content = "Line 1\\nLine 2" # This means body has literal backslash-n
        expected_full_content_std = f"# {file_title}\n\n{body_content}" # Actual newlines after H1, then body_content

        # Mocks for final file paths returned by __truediv__
        final_std_path_mock = MagicMock(spec=Path, name="final_std_path_mock")
        final_std_path_mock.__str__.return_value = str(self.mock_markdown_save_path_obj / expected_filename_part)
        
        final_rec_path_mock = MagicMock(spec=Path, name="final_rec_path_mock")
        final_rec_path_mock.__str__.return_value = str(self.mock_markdown_save_path_obj / "recovered" / expected_filename_part)
        
        final_rep_path_mock = MagicMock(spec=Path, name="final_rep_path_mock")
        final_rep_path_mock.__str__.return_value = str(self.mock_markdown_save_path_obj / "reprocessed" / expected_filename_part)

        # Mocks for subdirectories that are results of `sut_init_markdown_path / <subdir_name>`
        mock_recovered_subdir_path = MagicMock(spec=Path, name="mock_recovered_subdir_path")
        mock_recovered_subdir_path.mkdir = MagicMock(name="mock_recovered_subdir_path.mkdir")
        mock_reprocessed_subdir_path = MagicMock(spec=Path, name="mock_reprocessed_subdir_path")
        mock_reprocessed_subdir_path.mkdir = MagicMock(name="mock_reprocessed_subdir_path.mkdir")

        # Configure __truediv__ on sut_init_markdown_path (which is storage_service.markdown_save_path)
        def markdown_save_path_truediv_side_effect(segment):
            # print(f"DEBUG TEST - sut_init_markdown_path.__truediv__ called with: '{segment}'")
            if segment == "recovered":
                return mock_recovered_subdir_path
            elif segment == "reprocessed":
                return mock_reprocessed_subdir_path
            elif segment == expected_filename_part: # This is for type='analysis' (base path + filename)
                return final_std_path_mock
            # print(f"DEBUG TEST - sut_init_markdown_path.__truediv__ fallback for '{segment}'")
            return MagicMock(spec=Path, name=f"sut_init_markdown_path_truediv_fallback_{segment}")
        sut_init_markdown_path.__truediv__ = MagicMock(side_effect=markdown_save_path_truediv_side_effect)

        # Configure __truediv__ on the subdirectory mocks to return final file path mocks
        mock_recovered_subdir_path.__truediv__ = MagicMock(return_value=final_rec_path_mock, name="mock_recovered_subdir_path.__truediv__")
        mock_reprocessed_subdir_path.__truediv__ = MagicMock(return_value=final_rep_path_mock, name="mock_reprocessed_subdir_path.__truediv__")
        
        # --- Test 1: Standard analysis ---
        # print("DEBUG TEST: --- Starting Test 1: Standard Analysis ---")
        with patch.object(storage_service, '_generate_filename', return_value=expected_filename_part) as mock_gen_filename:
            path_std_returned_str = storage_service.save_markdown_content(file_title, body_content, ts_obj, "analysis")
            
            mock_gen_filename.assert_called_once_with(title_prefix=file_title, timestamp_obj=ts_obj, extension=".md")
            self.assertEqual(path_std_returned_str, str(final_std_path_mock))
            
            sut_init_markdown_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_recovered_subdir_path.mkdir.assert_not_called()
            mock_reprocessed_subdir_path.mkdir.assert_not_called()
            
            mock_open.assert_called_once_with(final_std_path_mock, 'w', encoding='utf-8')
            mock_file_handle.write.assert_called_once_with(expected_full_content_std)
        
        # Reset mocks for next test section
        mock_open.reset_mock(); mock_file_handle.reset_mock(); mock_gen_filename.reset_mock()
        sut_init_markdown_path.mkdir.reset_mock()
        mock_recovered_subdir_path.mkdir.reset_mock()
        mock_reprocessed_subdir_path.mkdir.reset_mock()
        sut_init_markdown_path.__truediv__.reset_mock(side_effect=True, return_value=True)
        mock_recovered_subdir_path.__truediv__.reset_mock(side_effect=True, return_value=True)
        mock_reprocessed_subdir_path.__truediv__.reset_mock(side_effect=True, return_value=True)

        # --- Test 2: Recovered ---
        # print("DEBUG TEST: --- Starting Test 2: Recovered ---")
        # Re-establish side effect for sut_init_markdown_path.__truediv__ as it was reset
        sut_init_markdown_path.__truediv__.side_effect = markdown_save_path_truediv_side_effect
        # RE-ESTABLISH __truediv__ for mock_recovered_subdir_path
        mock_recovered_subdir_path.__truediv__ = MagicMock(return_value=final_rec_path_mock, name="mock_recovered_subdir_path.__truediv___restored_for_test2")

        with patch.object(storage_service, '_generate_filename', return_value=expected_filename_part) as mock_gen_filename:
            path_rec_returned_str = storage_service.save_markdown_content(file_title, body_content, ts_obj, "analysis_recovered")
            mock_gen_filename.assert_called_once_with(title_prefix=file_title, timestamp_obj=ts_obj, extension=".md")
            self.assertEqual(path_rec_returned_str, str(final_rec_path_mock))

            sut_init_markdown_path.mkdir.assert_not_called()
            mock_recovered_subdir_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_reprocessed_subdir_path.mkdir.assert_not_called()
            
            mock_open.assert_called_once_with(final_rec_path_mock, 'w', encoding='utf-8')
            mock_file_handle.write.assert_called_once_with(expected_full_content_std)

        # Reset mocks
        mock_open.reset_mock(); mock_file_handle.reset_mock(); mock_gen_filename.reset_mock()
        sut_init_markdown_path.mkdir.reset_mock()
        mock_recovered_subdir_path.mkdir.reset_mock()
        mock_reprocessed_subdir_path.mkdir.reset_mock()
        sut_init_markdown_path.__truediv__.reset_mock(side_effect=True, return_value=True)
        mock_recovered_subdir_path.__truediv__.reset_mock(side_effect=True, return_value=True)
        mock_reprocessed_subdir_path.__truediv__.reset_mock(side_effect=True, return_value=True)

        # --- Test 3: Reprocessed ---
        # print("DEBUG TEST: --- Starting Test 3: Reprocessed ---")
        # Re-establish side effect for sut_init_markdown_path.__truediv__
        sut_init_markdown_path.__truediv__.side_effect = markdown_save_path_truediv_side_effect
        # RE-ESTABLISH __truediv__ for mock_reprocessed_subdir_path
        mock_reprocessed_subdir_path.__truediv__ = MagicMock(return_value=final_rep_path_mock, name="mock_reprocessed_subdir_path.__truediv___restored_for_test3")

        with patch.object(storage_service, '_generate_filename', return_value=expected_filename_part) as mock_gen_filename:
            path_rep_returned_str = storage_service.save_markdown_content(file_title, body_content, ts_obj, "analysis_reprocessed")
            mock_gen_filename.assert_called_once_with(title_prefix=file_title, timestamp_obj=ts_obj, extension=".md")
            self.assertEqual(path_rep_returned_str, str(final_rep_path_mock))
            
            sut_init_markdown_path.mkdir.assert_not_called()
            mock_recovered_subdir_path.mkdir.assert_not_called()
            mock_reprocessed_subdir_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            mock_open.assert_called_once_with(final_rep_path_mock, 'w', encoding='utf-8')
            mock_file_handle.write.assert_called_once_with(expected_full_content_std)

if __name__ == '__main__':
    unittest.main() 