import yaml
import os
import logging
from pathlib import Path
import sys # Added for sys._MEIPASS

# Configure logger for this service
log = logging.getLogger(__name__) # Use __name__ for logger hierarchy

VERSION_FILE_NAME = "VERSION"

class ConfigurationService:
    _config = None
    _config_path = None

    def __init__(self, config_file_name="config.yaml", app_name="sec_note_app"):
        """
        Initializes the ConfigurationService.
        Args:
            config_file_name (str): The name of the configuration file.
            app_name (str): The name of the application, used for the home directory config path.
        """
        self._determine_config_path(config_file_name, app_name)
        self._load_config()

    def _determine_config_path(self, config_file_name, app_name):
        """Determines the config path, checking local directory first, then user's home app directory."""
        # 1. Check local directory (relative to where the script is run)
        # This assumes that if a local config.yaml exists, it's in the CWD.
        # If this service is always called from scripts within src/, and config.yaml is at root,
        # this path needs adjustment or to rely on the home directory version.
        # For now, maintaining consistency with original loader's CWD assumption for local.
        local_config_path = Path(config_file_name)

        # 2. Check application-specific directory in user's home
        home_config_dir = Path.home() / f".{app_name.lower().replace(' ', '_')}"
        home_config_path = home_config_dir / config_file_name

        if local_config_path.exists():
            self._config_path = local_config_path
            log.info(f"Using local configuration file: {self._config_path}")
        else:
            self._config_path = home_config_path
            log.info(f"Local config not found. Using home directory configuration path: {self._config_path}")
            # Ensure the directory exists if we're going to potentially write a default config there
            home_config_dir.mkdir(parents=True, exist_ok=True)


    def _load_config(self):
        """Loads configuration from the determined YAML file path."""
        try:
            if not self._config_path.exists():
                log.warning(f"Config file {self._config_path} not found. Loading default values.")
                self._config = self._get_default_config()
                self._save_default_config() # Save the defaults to the path
                return

            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            if self._config is None: # File was empty or invalid YAML
                log.warning(f"Config file {self._config_path} was empty or invalid. Loading default values.")
                self._config = self._get_default_config()
                self._save_default_config()
            else:
                log.info(f"Configuration loaded from {self._config_path}")

        except Exception as e:
            log.error(f"Error loading configuration from {self._config_path}: {e}. Using default values.")
            self._config = self._get_default_config()
            # Attempt to save defaults if loading failed catastrophically
            if not self._config_path.exists():
                 self._save_default_config()


    def _save_default_config(self):
        """Saves the current (default) configuration to the config path."""
        if self._config is None:
            log.error("Attempted to save an empty configuration. Defaulting first.")
            self._config = self._get_default_config()

        try:
            # Ensure parent directory exists
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            log.info(f"Default configuration saved to {self._config_path}")
        except Exception as e:
            log.error(f"Error saving default configuration to {self._config_path}: {e}")

    def _get_default_config(self):
        """Returns a dictionary of default configuration values."""
        home_dir = str(Path.home())
        app_docs_dir = Path(home_dir) / "Documents" / "SecureNote"
        app_config_dir = Path.home() / ".sec_note_app" # User-specific config and data
        
        default_db_name = "transcriptions_new.db"
        default_db_path = app_config_dir / default_db_name

        default_markdown_save_path = app_docs_dir / "MeetingNotes"
        default_temp_audio_path = app_config_dir / "audio_temp"

        # Default configuration structure
        default_config = {
            'database': {
                'name': str(default_db_path) # DB in ~/.sec_note_app/
            },
            'paths': {
                'markdown_save': str(default_markdown_save_path), # Analyses in ~/Documents/SecureNote/analyses
                'temp_path': str(default_temp_audio_path),      # Temp audio in ~/.sec_note_app/audio_temp
                'prompt_templates': "prompt_templates",
                'app_docs_dir': str(app_docs_dir),
                'app_config_dir': str(app_config_dir)
            },
            'models': {
                'whisper': {'default': 'base'},
                'ollama_api_url': 'http://localhost:11434/api/generate',
                'llm_model_name': 'gemma3:12b', # Default from original config.yaml
                'temperature': 0.1,
                'top_k': 64,
                'top_p': 0.95,
                'num_ctx': 100000
            },
            'audio': {
                'frames_per_buffer': 1024,
                'channels': 1,
                'format': 'paFloat32', # PyAudio format string, will be evaluated by getattr
                'allowed_extensions': ['wav', 'mp3', 'm4a', 'ogg', 'flac'],
                'samplerate': 16000
            },
            'error_messages': {
                'failed_analysis': [
                    "Ollama analysis failed. See logs for details.",
                    "Ollama returned empty response.",
                    "LLM output parsing failed or content was not in expected format.",
                    "Could not parse summary."
                ]
            }
        }

        return default_config

    def get_config(self):
        """Returns the entire configuration dictionary."""
        if self._config is None:
            self._load_config() # Ensure config is loaded
        return self._config

    def get(self, *keys, default=None):
        """
        Retrieves a configuration value using a sequence of keys.
        Args:
            *keys: A sequence of strings representing the path to the desired value.
            default: The value to return if the keys are not found.
        Returns:
            The configuration value or the default.
        """
        if self._config is None:
            self._load_config() # Ensure config is loaded

        value = self._config
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else: # Key path is deeper than the structure at this point
                    return default
            return value
        except KeyError:
            return default
        
    @staticmethod
    def get_application_version() -> str | None:
        """
        Reads the application version from the VERSION file.
        Handles running from source and from a PyInstaller bundle.
        """
        try:
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                # Running in a PyInstaller bundle
                # PyInstaller extracts data to sys._MEIPASS. The VERSION file should be at the root of that temp dir.
                base_path = Path(sys._MEIPASS)
                version_file_path = base_path / VERSION_FILE_NAME
            else:
                # Running as a normal script
                # Assume VERSION file is at the project root, relative to this src/config_service.py file
                # Path(__file__).resolve() is src/config_service.py
                # .parent is src/
                # .parent is project root
                base_path = Path(__file__).resolve().parent.parent 
                version_file_path = base_path / VERSION_FILE_NAME

            if version_file_path.exists():
                with open(version_file_path, "r") as f:
                    version = f.read().strip()
                    log.info(f"Application version '{version}' loaded from {version_file_path}")
                    return version
            else:
                log.error(f"Version file not found at: {version_file_path}")
                # Fallback: try to read from CWD if not found in expected locations
                # This might be useful if the script is run from the root directory directly
                # and not as part of a structured execution from within src/
                cwd_version_file_path = Path.cwd() / VERSION_FILE_NAME
                if cwd_version_file_path.exists():
                    log.info(f"Trying fallback to CWD for VERSION file: {cwd_version_file_path}")
                    with open(cwd_version_file_path, "r") as f:
                        version = f.read().strip()
                        log.info(f"Application version '{version}' loaded from CWD fallback {cwd_version_file_path}")
                        return version
                else:
                    log.error(f"Version file also not found at CWD: {cwd_version_file_path}")
                    return None
        except Exception as e:
            log.exception(f"Error reading version file: {e}") # Use log.exception to include stack trace
            return None