# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-06-01
### Changed
- Modified `scripts/build.py` to include the `VERSION` file in the PyInstaller build for both the main application and the uninstaller. This ensures the application can access version information at runtime.
    - Updated the PyInstaller `--add-data` specifier for the `VERSION` file from `VERSION:VERSION` to `VERSION:.` to prevent it from being bundled as a directory.
- Updated `src/storage_service.py`:
    - `_generate_filename` method:
        - Sanitize filenames more robustly.
        - Implement filename length checking against `MAX_FILENAME_LENGTH`.
        - Truncate filenames if they exceed the maximum allowed length.
        - Add logging for truncation events.
    - `init_db` method:
        - Added a new `title_for_file` TEXT column to the `recordings` table.
        - Implemented backfilling of this column for existing records that have `analysis_markdown` but no `title_for_file`, using a generated title.
    - `save_markdown_content` method:
        - Changed signature: `title` parameter renamed to `file_and_h1_title`.
        - Changed signature: `markdown_content` parameter renamed to `body_markdown_content` (expects H1 to be stripped by caller).
        - Now prepends `file_and_h1_title` as an H1 heading to `body_markdown_content` before saving.
        - Creates "recovered" and "reprocessed" subdirectories under the main markdown save path as needed.
    - `update_analysis` method:
        - Added new parameter `title_for_file` to store the intended filename/H1 title in the database.
- Refactored version retrieval:
    - Moved `get_application_version` function into the `ConfigurationService` class in `src/config_service.py` as a method.
    - Updated `src/launcher.py` to instantiate `ConfigurationService` and call this method to log the application version at startup.
- Updated tests (`tests/test_storage_service.py`, `tests/test_transcribe.py`) to reflect changes in `StorageService` method signatures and added `title_for_file` handling.

### Added
- Added a step to `scripts/build.py` to create a versioned zip archive (e.g., `secure-note-X.Y.Z.zip`) in the `dist` directory. This archive contains a folder with the main executable, the uninstaller, and the `VERSION` file.
    - The archive name will include `-test` (e.g., `secure-note-X.Y.Z-test.zip`) if the build is run without a version increment flag (i.e., a development build).
- Added new test file `tests/test_new_features.py` with initial tests for file recovery and interactive re-analysis logic in `src/transcribe.py` (`process_existing_transcription` function).
- Added new tests in `tests/test_storage_service.py`:
    - `test_init_db_adds_title_for_file_column_and_backfills`: Verifies schema migration and data backfilling for the new `title_for_file` column.
    - `test_update_analysis_saves_title_for_file`: Verifies `title_for_file` is correctly saved by `update_analysis`.
    - `test_get_records_with_analysis_retrieves_title_for_file`: Verifies `title_for_file` is correctly retrieved.
    - `test_save_markdown_content_creates_h1_and_subfolders`: Verifies H1 title usage and subfolder creation in `save_markdown_content`.
- Refactored `setUp` and `tearDown` in `tests/test_storage_service.py` for improved mocking of `Path` objects, `ConfigurationService` injection, and in-memory database handling.

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