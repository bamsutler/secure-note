# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Modified `scripts/build.py` to include the `VERSION` file in the PyInstaller build for both the main application and the uninstaller. This ensures the application can access version information at runtime.
    - Updated the PyInstaller `--add-data` specifier for the `VERSION` file from `VERSION:VERSION` to `VERSION:.` to prevent it from being bundled as a directory.
- Updated `src/storage_service.py` (`_generate_filename` method) to:
    - Sanitize filenames more robustly.
    - Implement filename length checking against `MAX_FILENAME_LENGTH`.
    - Truncate filenames if they exceed the maximum allowed length.
    - Add logging for truncation events.
- Refactored version retrieval:
    - Moved `get_application_version` function into the `ConfigurationService` class in `src/config_service.py` as a method.
    - Updated `src/launcher.py` to instantiate `ConfigurationService` and call this method to log the application version at startup.

### Added
- Added a step to `scripts/build.py` to create a versioned zip archive (e.g., `secure-note-X.Y.Z.zip`) in the `dist` directory. This archive contains a folder with the main executable, the uninstaller, and the `VERSION` file.
    - The archive name will include `-test` (e.g., `secure-note-X.Y.Z-test.zip`) if the build is run without a version increment flag (i.e., a development build).

## [0.3.0] - YYYY-MM-DD
### Changed
- **Major refactor of application structure:**
    - Introduced `src/` directory for all core application code.
    - Old top-level scripts (`core_processing.py`, `summary.py`, `transcribe.py`, `launcher.py`) refactored into services within `src/` (e.g., `analysis_service.py`, `audio_service.py`, `config_service.py`, `storage_service.py`, `transcription_service.py`).
    - New main application entry point at `main.py`, using `src/launcher.py`.
- Build script (`build.py`) moved to `scripts/build.py` and enhanced for version incrementing.
- `uninstaller_template.py` moved to `scripts/` directory.
- Updated `makefile` for new build targets (`build-patch`, `build-minor`, `build-major`).
- Modified `.gitignore` and `requirements.txt`.

### Added
- **Versioning and Changelog:**
    - `VERSION` file to store application version.
    - `CHANGELOG.md` for tracking project changes.
    - Automated version incrementing in `scripts/build.py` controlled by `Makefile` targets.
- New application services under `src/`:
    - `analysis_service.py`: Handles text analysis (refactoring previous LLM and summary functionality).
    - `audio_service.py`: Manages audio recording and processing.
    - `cli_interface.py`: For command-line interactions.
    - `config_service.py`: Manages application configuration.
    - `environment_manager.py`: Manages environment setup.
    - `logging_service.py` and `menu_aware_log_handler.py`: Enhanced logging.
    - `storage_service.py`: Manages data storage.
    - `transcription_service.py`: Handles audio transcription.
- `prompt_templates/` directory with various templates for LLM interaction.
- Comprehensive test suite under `tests/` for new services.
- `conftest.py` for test configuration.

### Removed
- `transcription_server.py` (FastAPI server).
- Old top-level `build.py`, `core_processing.py`, `summary.py`, `transcribe.py`. 