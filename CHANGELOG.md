# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]
### Added
- Initial project setup. 