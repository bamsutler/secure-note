import unittest
from unittest.mock import patch, MagicMock
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from src.storage_service import StorageService
# ConfigurationService might be needed if we want to mock its instance specifically
# from src.config_service import ConfigurationService 

class TestStorageService(unittest.TestCase):

    @patch('src.storage_service.Path')
    @patch('src.storage_service.config_service')
    def setUp(self, mock_config_service_module, MockPathClass):
        def mock_config_get_side_effect(*args, **kwargs):
            if args == ('database', 'name'):
                return "file::memory:?cache=shared"
            if args == ('paths', 'markdown_save'):
                return "mock_markdown_notes"
            if args == ('paths', 'temp_path'):
                return "mock_audio_temp"
            # Add default for audio related configs if StorageService starts using them directly
            # For now, StorageService only uses 'get_pyaudio_sample_size' from AudioService,
            # which is a static method and doesn't rely on AudioService's own config for that call.
            return MagicMock()

        mock_config_service_module.get.side_effect = mock_config_get_side_effect
        
        mock_path_instance = MagicMock()
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path_instance.mkdir = MagicMock()
        mock_path_instance.__truediv__ = lambda self, other: mock_path_instance # For path1 / path2
        mock_path_instance.exists.return_value = True # Assume paths "exist" for mkdir(exist_ok=True)
        MockPathClass.return_value = mock_path_instance

        self.storage_service = StorageService()
        # init_db is called by StorageService constructor

    def tearDown(self):
        # StorageService uses in-memory DB, connection managed internally.
        # No explicit close needed here.
        pass

    def test_save_and_get_recording_transcription(self):
        """Test saving a new recording with transcription and retrieving it."""
        ts = datetime.now()
        audio_bytes = b'fake_audio_data'
        samplerate = 16000
        whisper_model = "tiny"
        transcription_text = "This is a test transcription."

        record_id = self.storage_service.save_initial_recording_and_transcription(
            timestamp=ts,
            audio_wav_bytes=audio_bytes,
            samplerate=samplerate,
            whisper_model_used=whisper_model,
            transcription=transcription_text
        )
        self.assertIsNotNone(record_id, "save_initial_recording_and_transcription should return a record ID.")

        retrieved_record = self.storage_service.get_recording_by_id(record_id)
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
        retrieved_record = self.storage_service.get_recording_by_id(99999) # Non-existent ID
        self.assertIsNone(retrieved_record)

    def test_update_and_get_analysis(self):
        """Test saving an initial recording and then updating it with analysis."""
        ts = datetime.now()
        record_id = self.storage_service.save_initial_recording_and_transcription(
            timestamp=ts,
            audio_wav_bytes=b'audio',
            samplerate=16000,
            whisper_model_used="base",
            transcription="Initial text."
        )
        self.assertIsNotNone(record_id)

        llm_model = "test_llm_v1"
        analysis_md = "# Analysis\nThis is the detailed analysis."
        
        update_success = self.storage_service.update_analysis(
            recording_id=record_id,
            llm_model_used=llm_model,
            analysis_markdown=analysis_md
        )
        self.assertTrue(update_success, "update_analysis should return True on success.")

        retrieved_record = self.storage_service.get_recording_by_id(record_id)
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record['llm_model_used'], llm_model)
        self.assertEqual(retrieved_record['analysis_markdown'], analysis_md)
        self.assertEqual(retrieved_record['transcription'], "Initial text.") # Check transcription is still there

    def test_update_analysis_for_non_existent_record(self):
        """Test updating analysis for a record ID that does not exist."""
        update_success = self.storage_service.update_analysis(
            recording_id=88888, # Non-existent
            llm_model_used="test_model",
            analysis_markdown="some analysis"
        )
        self.assertFalse(update_success, "update_analysis should return False for non-existent record ID.")

    def test_get_last_transcription_for_reanalysis(self):
        """Test fetching the last transcription eligible for re-analysis."""
        # Clear any existing records (new in-memory DB for each test method due to setUp)
        
        # Scenario 1: No records
        self.assertIsNone(self.storage_service.get_last_transcription_for_reanalysis())

        # Scenario 2: Record with no transcription
        ts1 = datetime.now() - timedelta(minutes=10)
        id1 = self.storage_service.save_initial_recording_and_transcription(ts1, b'a1', 16000, 't1', None) # No transcription
        self.assertIsNone(self.storage_service.get_last_transcription_for_reanalysis())
        
        # Scenario 3: Record with empty transcription
        ts2 = datetime.now() - timedelta(minutes=8)
        id2 = self.storage_service.save_initial_recording_and_transcription(ts2, b'a2', 16000, 't2', "") # Empty transcription
        self.assertIsNone(self.storage_service.get_last_transcription_for_reanalysis())

        # Scenario 4: A valid record
        ts3 = datetime.now() - timedelta(minutes=5)
        transcription3 = "This is a valid transcription for record 3."
        id3 = self.storage_service.save_initial_recording_and_transcription(ts3, b'a3', 16000, 'w3', transcription3)
        last_record = self.storage_service.get_last_transcription_for_reanalysis()
        self.assertIsNotNone(last_record)
        self.assertEqual(last_record['id'], id3)
        self.assertEqual(last_record['transcription'], transcription3)

        # Scenario 5: Another valid record, should fetch this newer one
        ts4 = datetime.now() - timedelta(minutes=2)
        transcription4 = "This is a newer valid transcription for record 4."
        analysis4_md = "Pre-existing analysis for record 4"
        id4 = self.storage_service.save_initial_recording_and_transcription(ts4, b'a4', 16000, 'w4', transcription4)
        self.storage_service.update_analysis(id4, "llm_prev", analysis4_md) # Add analysis to it

        last_record_again = self.storage_service.get_last_transcription_for_reanalysis()
        self.assertIsNotNone(last_record_again)
        self.assertEqual(last_record_again['id'], id4)
        self.assertEqual(last_record_again['transcription'], transcription4)
        self.assertEqual(last_record_again['analysis_markdown'], analysis4_md) # Check analysis is also fetched

    def test_get_recordings_without_analysis(self):
        """Test fetching records that have a transcription but no analysis."""
        # Scenario 1: No records
        self.assertEqual(self.storage_service.get_recordings_without_analysis(), [])

        # Record 1: No transcription
        ts1 = datetime.now() - timedelta(days=1, minutes=10)
        id1 = self.storage_service.save_initial_recording_and_transcription(ts1, b'a1', 16000, 'w1', None)
        
        # Record 2: With transcription, no analysis (should be fetched)
        ts2 = datetime.now() - timedelta(days=1, minutes=5)
        transcription2 = "Record 2 needs analysis."
        id2 = self.storage_service.save_initial_recording_and_transcription(ts2, b'a2', 16000, 'w2', transcription2)

        # Record 3: With transcription and analysis (should NOT be fetched)
        ts3 = datetime.now() - timedelta(days=1, minutes=1)
        transcription3 = "Record 3 has analysis."
        id3 = self.storage_service.save_initial_recording_and_transcription(ts3, b'a3', 16000, 'w3', transcription3)
        self.storage_service.update_analysis(id3, "llm_done", "Analysis complete for record 3.")

        # Record 4: Empty transcription (should NOT be fetched)
        ts4 = datetime.now() - timedelta(days=1, minutes=15)
        id4 = self.storage_service.save_initial_recording_and_transcription(ts4, b'a4', 16000, 'w4', "")
        
        # Record 5: With transcription, analysis is an empty string (should be fetched as missing analysis)
        ts5 = datetime.now() - timedelta(days=1, minutes=3)
        transcription5 = "Record 5 needs analysis (analysis is empty string)."
        id5 = self.storage_service.save_initial_recording_and_transcription(ts5, b'a5', 16000, 'w5', transcription5)
        self.storage_service.update_analysis(id5, "llm_attempted_empty", "") # Analysis is empty string

        unanalyzed_records = self.storage_service.get_recordings_without_analysis()
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


if __name__ == '__main__':
    unittest.main() 