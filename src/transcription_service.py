import whisper # Assuming whisper is installed
import os
from pathlib import Path

# Assuming ConfigurationService and LoggingService are in the same directory or accessible via PYTHONPATH
from src.config_service import ConfigurationService
from src.logging_service import LoggingService

log = LoggingService.get_logger(__name__)

class WhisperProvider:
    """Handles loading and using the Whisper model for transcription."""
    _model_instance = None
    _loaded_model_name = None

    def __init__(self, config_service: ConfigurationService = ConfigurationService()):
        """
        Initializes the WhisperProvider.
        Args:
            config_service (ConfigurationService): The configuration service instance.
        """

        self.config_service = config_service
        self.default_model_name = self.config_service.get('models', 'whisper', 'default', default='base')

    def _load_model(self, model_name_to_load: str):
        """Loads the Whisper model if not already loaded or if a different model is requested."""
        if WhisperProvider._model_instance is not None and WhisperProvider._loaded_model_name == model_name_to_load:
            log.debug(f"Whisper model '{model_name_to_load}' already loaded.")
            return WhisperProvider._model_instance
        
        try:
            log.info(f"Loading Whisper model '{model_name_to_load}' globally for TranscriptionService... (This may take a moment)")
            WhisperProvider._model_instance = whisper.load_model(model_name_to_load)
            WhisperProvider._loaded_model_name = model_name_to_load
            log.info(f"Global Whisper model '{model_name_to_load}' loaded for TranscriptionService.")
            return WhisperProvider._model_instance
        except Exception as e:
            log.error(f"Error loading global Whisper model '{model_name_to_load}' for TranscriptionService: {e}")
            WhisperProvider._model_instance = None
            WhisperProvider._loaded_model_name = None
            raise RuntimeError(f"Whisper model '{model_name_to_load}' failed to load: {e}") # Re-raise to signal failure

    def ensure_model_loaded(self, model_name: str | None = None) -> bool:
        """
        Ensures that the specified Whisper model (or default if None) is loaded.
        This can be used as a startup check.
        Args:
            model_name (str | None): Specific Whisper model name to load. Defaults to configured default.
        Returns:
            bool: True if the model is loaded or was successfully loaded, False otherwise (though it usually raises on failure).
        Raises:
            RuntimeError: If the model fails to load.
        """
        target_model_name = model_name if model_name else self.default_model_name
        try:
            self._load_model(target_model_name)
            log.info(f"Whisper model '{target_model_name}' is confirmed to be loaded/loadable.")
            return True
        except RuntimeError as r_err: # Specifically catch and re-raise model loading errors
            log.error(f"Failed to ensure Whisper model '{target_model_name}' is loaded: {r_err}")
            raise # Re-raise the RuntimeError to signal critical failure
        except Exception as e:
            log.error(f"An unexpected error occurred while ensuring Whisper model '{target_model_name}' was loaded: {e}")
            # Wrap other exceptions in a RuntimeError to maintain consistent error signaling for critical failure
            raise RuntimeError(f"Unexpected error loading model '{target_model_name}': {e}")

    def transcribe(self, audio_file_path: str, model_name: str | None = None) -> str | None:
        """
        Transcribes an audio file using the Whisper model.
        Args:
            audio_file_path (str): Path to the audio file.
            model_name (str | None): Specific Whisper model name to use. Defaults to configured default.
        Returns:
            str | None: The transcribed text, or None on failure.
        """
        target_model_name = model_name if model_name else self.default_model_name
        
        try:
            instance = self._load_model(target_model_name)
            if not os.path.exists(audio_file_path):
                log.error(f"Audio file not found for transcription: {audio_file_path}")
                return None

            log.info(f"Transcribing audio file: {audio_file_path} with model '{target_model_name}'...")
            # In newer versions of Whisper, fp16 is often handled automatically based on device.
            # Forcing fp16=False might be for CPU explicitly. Check Whisper docs for best practice.
            result = instance.transcribe(audio_file_path, fp16=False) 
            transcribed_text = result["text"].strip()
            log.info(f"Transcription complete for {audio_file_path}.")
            return transcribed_text
        except RuntimeError as r_err: # Catch model loading errors specifically
            log.error(f"Cannot transcribe due to model loading error: {r_err}")
            # No need to re-raise here as the error is specific to transcription failing
            return None
        except Exception as e:
            log.error(f"Error during transcription of {audio_file_path} with model '{target_model_name}': {e}")
            return None

class TranscriptionService:
    """
    Service responsible for providing audio transcription capabilities.
    It uses a provider (e.g., WhisperProvider) to perform the actual transcription.
    """
    def __init__(self, provider: WhisperProvider | None = None, config_service_override: ConfigurationService | None = None):
        """
        Initializes the TranscriptionService.
        Args:
            provider (WhisperProvider | None): An instance of a transcription provider (e.g., WhisperProvider).
                                            If None, a WhisperProvider is instantiated.
            config_service_override (ConfigurationService | None): An optional ConfigurationService instance.
                                                        If provider is None, this config_service (or a new one) 
                                                        will be passed to the new WhisperProvider.
                                                        If provider is provided, this argument is ignored.
        """
        if provider is None:
            # If no provider, we need a config service for the new WhisperProvider
            current_config_service = config_service_override if config_service_override else ConfigurationService()
            self.provider = WhisperProvider(config_service=current_config_service)
            log.info("TranscriptionService initialized with default WhisperProvider.")
        else:
            self.provider = provider
            log.info(f"TranscriptionService initialized with provided {type(provider).__name__}.")

    def transcribe_audio(self, audio_file_path: str, whisper_model_size: str | None = None) -> str | None:
        """
        Transcribes the given audio file using the configured provider.
        Args:
            audio_file_path (str): The path to the audio file to be transcribed.
            whisper_model_size (str | None): The specific Whisper model size to use (e.g., 'base', 'small').
                                            If None, the provider's default will be used.
        Returns:
            str | None: The transcribed text, or None if transcription fails.
        """
        if not audio_file_path:
            log.error("Audio file path is required for transcription.")
            return None
        
        log.info(f"TranscriptionService: Request to transcribe '{audio_file_path}' with model '{whisper_model_size or 'default'}'.")
        try:
            transcribed_text = self.provider.transcribe(audio_file_path, model_name=whisper_model_size)
            if transcribed_text:
                log.info(f"TranscriptionService: Successfully transcribed '{audio_file_path}'.")
            else:
                log.warning(f"TranscriptionService: Transcription failed or returned empty for '{audio_file_path}'.")
            return transcribed_text
        except Exception as e:
            # Providers should ideally handle their own specific errors and log them.
            # This is a fallback.
            log.error(f"TranscriptionService: An unexpected error occurred during transcription of '{audio_file_path}': {e}")
            return None
