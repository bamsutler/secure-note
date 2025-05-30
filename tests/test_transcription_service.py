import unittest
from unittest.mock import patch, MagicMock
import os

from src.transcription_service import WhisperProvider
from src.config_service import ConfigurationService

class TestTranscriptionService(unittest.TestCase):

    def setUp(self):
        # Mock ConfigurationService to avoid dependency on actual config files
        self.mock_config_service = MagicMock(spec=ConfigurationService)
        self.mock_config_service.get.side_effect = self._get_config_side_effect

        # Patch the ConfigurationService globally or where WhisperProvider can access it
        # For simplicity, if WhisperProvider instantiates it, we might need to patch its instantiation
        # Assuming WhisperProvider might use a global config or get it passed, for now, we prepare the mock.
        # If WhisperProvider creates its own ConfigService, this patch needs to be more targeted.
        self.whisper_provider = WhisperProvider(config_service=self.mock_config_service)

    def _get_config_side_effect(self, *args, **kwargs):
        # print(f"Mock config_service.get called with: {args}") 
        if args == ('models', 'whisper', 'default'):
            return 'tiny'
        if args == ('models', 'whisper', 'path'):
            # Return a dummy path, as we are mocking model loading
            return '/dummy/model/path' 
        # Add other specific config mocks if WhisperProvider uses them
        return MagicMock() # Default mock for any other config get

    @patch('src.transcription_service.whisper.load_model')
    def test_ensure_model_loaded_loads_model_if_not_loaded(self, mock_load_model):
        # Test that ensure_model_loaded calls load_model if no model is loaded
        WhisperProvider._model_instance = None # Ensure model is not loaded (class attribute)
        WhisperProvider._loaded_model_name = None # Ensure model name is not set (class attribute)
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance

        self.whisper_provider.ensure_model_loaded(model_name='tiny')

        mock_load_model.assert_called_once_with('tiny')
        self.assertEqual(WhisperProvider._model_instance, mock_model_instance) # Check class attribute

    @patch('src.transcription_service.whisper.load_model')
    def test_ensure_model_loaded_uses_existing_model_if_loaded(self, mock_load_model):
        # Test that ensure_model_loaded does not call load_model if a model is already loaded
        mock_existing_model = MagicMock()
        WhisperProvider._model_instance = mock_existing_model # Use class attribute
        WhisperProvider._loaded_model_name = 'tiny' # Use class attribute

        self.whisper_provider.ensure_model_loaded(model_name='tiny')

        mock_load_model.assert_not_called()
        self.assertEqual(WhisperProvider._model_instance, mock_existing_model) # Check class attribute

    @patch('src.transcription_service.whisper.load_model')
    def test_ensure_model_loaded_reloads_if_different_model_name_requested(self, mock_load_model):
        # Test that ensure_model_loaded reloads if a different model name is requested
        mock_existing_model = MagicMock()
        WhisperProvider._model_instance = mock_existing_model # Use class attribute
        WhisperProvider._loaded_model_name = 'tiny' # Use class attribute

        new_mock_model_instance = MagicMock()
        mock_load_model.return_value = new_mock_model_instance

        self.whisper_provider.ensure_model_loaded(model_name='base')
        mock_load_model.assert_called_once_with('base')
        self.assertEqual(WhisperProvider._model_instance, new_mock_model_instance) # Check class attribute
        self.assertEqual(WhisperProvider._loaded_model_name, 'base') # Check class attribute

    @patch.object(WhisperProvider, '_load_model') # Patch _load_model to control model instance directly
    @patch('src.transcription_service.os.path.exists') # Mock os.path.exists
    def test_transcribe_successful(self, mock_path_exists, mock_internal_load_model):
        # Test successful transcription
        mock_path_exists.return_value = True # Simulate audio file exists
        
        # Mock the model and its transcribe method
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {'text': 'This is a test transcription.'}
        
        # Configure _load_model to return our mock_model
        mock_internal_load_model.return_value = mock_model

        audio_file_path = "dummy/path/to/audio.wav"
        model_name = "tiny"
        
        transcription = self.whisper_provider.transcribe(audio_file_path, model_name)

        mock_internal_load_model.assert_called_once_with(model_name)
        mock_model.transcribe.assert_called_once_with(audio_file_path, fp16=False)
        self.assertEqual(transcription, "This is a test transcription.")

    @patch.object(WhisperProvider, '_load_model') # Patch _load_model
    @patch('src.transcription_service.os.path.exists')
    def test_transcribe_file_not_found(self, mock_path_exists, mock_internal_load_model):
        # Test transcription when audio file does not exist
        mock_path_exists.return_value = False # Simulate audio file does not exist
        
        # Even if file doesn't exist, _load_model is called before the check
        # So, we mock the model instance that _load_model would return
        mock_model_instance = MagicMock()
        mock_internal_load_model.return_value = mock_model_instance

        audio_file_path = "non_existent_audio.wav"
        model_name = "tiny"

        transcription = self.whisper_provider.transcribe(audio_file_path, model_name)
        
        mock_internal_load_model.assert_called_once_with(model_name)
        mock_path_exists.assert_called_once_with(audio_file_path)
        # The actual transcription method on the model should not be called
        mock_model_instance.transcribe.assert_not_called()
        self.assertIsNone(transcription)

    @patch.object(WhisperProvider, '_load_model') # Patch _load_model
    @patch('src.transcription_service.os.path.exists')
    def test_transcribe_model_transcription_fails(self, mock_path_exists, mock_internal_load_model):
        # Test transcription when the model's transcribe method fails
        mock_path_exists.return_value = True

        mock_model = MagicMock()
        mock_model.transcribe.return_value = None # Simulate transcription failure
        mock_internal_load_model.return_value = mock_model # _load_model returns this model

        audio_file_path = "dummy/path/to/audio.wav"
        model_name = "tiny"

        transcription = self.whisper_provider.transcribe(audio_file_path, model_name)

        mock_internal_load_model.assert_called_once_with(model_name)
        mock_model.transcribe.assert_called_once_with(audio_file_path, fp16=False)
        self.assertIsNone(transcription)

    @patch.object(WhisperProvider, '_load_model') # Patch _load_model
    @patch('src.transcription_service.os.path.exists')
    def test_transcribe_model_transcription_returns_empty_text(self, mock_path_exists, mock_internal_load_model):
        # Test transcription when the model's transcribe method returns empty text
        mock_path_exists.return_value = True

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {'text': ''} # Simulate empty transcription
        mock_internal_load_model.return_value = mock_model # _load_model returns this model

        audio_file_path = "dummy/path/to/audio.wav"
        model_name = "tiny"

        transcription = self.whisper_provider.transcribe(audio_file_path, model_name)

        mock_internal_load_model.assert_called_once_with(model_name)
        mock_model.transcribe.assert_called_once_with(audio_file_path, fp16=False)
        self.assertEqual(transcription, "")

    @patch('src.transcription_service.whisper.load_model')
    def test_ensure_model_loaded_handles_load_exception(self, mock_load_model):
        # Test that ensure_model_loaded handles exceptions during model loading
        WhisperProvider._model_instance = None # Use class attribute
        WhisperProvider._loaded_model_name = None # Use class attribute
        mock_load_model.side_effect = Exception("Model load failed")

        with self.assertRaises(Exception) as context:
             self.whisper_provider.ensure_model_loaded(model_name='tiny')
        self.assertTrue("Model load failed" in str(context.exception))
        mock_load_model.assert_called_once_with('tiny')
        self.assertIsNone(WhisperProvider._model_instance) # Check class attribute

    # @patch('src.transcription_service.whisper.load_model') # Ensure this test remains commented
    # def test_transcribe_audio_file_not_found(self, mock_load_model):
    #     # Test transcription when audio file does not exist
    #     # This test needs to be completed or was a leftover from previous edits.
    #     # For now, ensuring it remains commented as per the original structure before the faulty reapply.
    #     pass # Placeholder if it were to be uncommented


if __name__ == '__main__':
    unittest.main() 