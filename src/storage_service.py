import sqlite3
import os
from datetime import datetime
from pathlib import Path
import wave # For writing WAV files
import pyaudio # For paInt16 enum
import re # Added for filename sanitization and length check

# Assuming ConfigurationService and LoggingService are in the same directory or accessible via PYTHONPATH
from src.config_service import ConfigurationService # Use relative import if in a package
from src.logging_service import LoggingService
from src.audio_service import AudioService # For get_pyaudio_sample_size


log = LoggingService.get_logger(__name__)

MAX_FILENAME_LENGTH = 255 # Maximum length for a filename, adjust as needed

class StorageService:
    def __init__(self, config_service_instance=None):
        self.config = config_service_instance if config_service_instance else ConfigurationService()
        
        self.db_name = self.config.get('database', 'name', default='transcriptions_new.db')
        self.markdown_save_path = Path(self.config.get('paths', 'markdown_save', default='markdown_notes'))
        self.temp_path = Path(self.config.get('paths', 'temp_path', default='audio_temp'))
        
        # Ensure directories exist
        if ":memory:" not in self.db_name: 
            self.db_path = Path(self.db_name)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = None 
            
        self.markdown_save_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True) # Ensure temp path exists

        self.init_db()

    def _get_db_connection(self):
        """Establishes and returns a database connection."""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.row_factory = sqlite3.Row # Access columns by name
            return conn
        except sqlite3.Error as e:
            log.error(f"Error connecting to database '{self.db_name}': {e}")
            raise # Re-raise the exception so callers can handle it

    def init_db(self):
        """Initializes the SQLite database and creates/alters the recordings table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    audio_wav BLOB NOT NULL, -- Consider storing path if audio is too large for BLOB
                    samplerate INTEGER NOT NULL,
                    whisper_model_used TEXT NOT NULL,
                    transcription TEXT,
                    llm_model_used TEXT, -- For any LLM (Ollama or local)
                    analysis_markdown TEXT -- Stores the full markdown from LLM analysis
                )
                """)
                # Check and remove old columns if they exist (idempotent check)
                table_info = cursor.execute("PRAGMA table_info(recordings)").fetchall()
                column_names = [info[1] for info in table_info]
                
                # Add title_for_file column if it doesn't exist
                if 'title_for_file' not in column_names:
                    cursor.execute("ALTER TABLE recordings ADD COLUMN title_for_file TEXT")
                    log.info("Added 'title_for_file' column to 'recordings' table.")
                    conn.commit() # Commit after schema change before attempting backfill

                    # Backfill title_for_file for existing records with analysis_markdown
                    log.info("Attempting to backfill 'title_for_file' for existing records...")
                    cursor.execute("""
                        SELECT id, timestamp, analysis_markdown 
                        FROM recordings 
                        WHERE analysis_markdown IS NOT NULL AND analysis_markdown != '' AND title_for_file IS NULL
                    """)
                    records_to_backfill = cursor.fetchall()
                    backfilled_count = 0
                    for rec in records_to_backfill:
                        record_id = rec['id']
                        timestamp_str = rec['timestamp'] # This is ISO string
                        # analysis_md_body = rec['analysis_markdown'] # Body of markdown

                        # Try to parse timestamp_str to datetime object for formatting
                        try:
                            dt_obj = datetime.fromisoformat(timestamp_str)
                            formatted_time = dt_obj.strftime('%Y%m%d_%H%M%S')
                        except ValueError:
                            formatted_time = timestamp_str # Fallback to raw string if parsing fails
                        
                        # For backfilling, the actual H1 from analysis_markdown body isn't easily available without parsing.
                        # We'll use a generic title.
                        # The user requested "dummy titles" for backfilling.
                        dummy_title = f"Analysis for ID {record_id} ({formatted_time})"
                        
                        cursor.execute("UPDATE recordings SET title_for_file = ? WHERE id = ?", (dummy_title, record_id))
                        backfilled_count += 1
                    conn.commit()
                    if backfilled_count > 0:
                        log.info(f"Backfilled 'title_for_file' for {backfilled_count} records.")
                    else:
                        log.info("No records found requiring 'title_for_file' backfill, or backfill already done.")
                
                # Renaming example: 'ollama_model_used' to 'llm_model_used'
                # 'ollama_response_markdown' to 'analysis_markdown'
                # We'll assume the new schema from the CREATE TABLE statement above.
                # If altering from an existing DB structure used by core_processing.py, more complex migrations might be needed.
                # For this new service, we define the schema cleanly.

                # The following logic for dropping columns is kept for reference if adapting an old DB,
                # but with a new DB, it's not strictly necessary.
                cols_to_drop = ['model_used', 'ollama_model_used', 'ollama_response_markdown'] # Old names from core_processing
                cols_to_rename_map = {
                    'model_used': 'whisper_model_used', # Example: if original was just 'model_used' for whisper
                    'ollama_model_used': 'llm_model_used',
                    'ollama_response_markdown': 'analysis_markdown'
                }

                # Add new columns if they don't exist (idempotent)
                # This example assumes we are starting fresh with the schema defined in CREATE TABLE
                # If migrating an existing schema, you would check for specific columns and ADD or RENAME.
                # For instance, if 'whisper_model_used' was previously 'model_used':
                # if 'model_used' in column_names and 'whisper_model_used' not in column_names:
                #     cursor.execute("ALTER TABLE recordings RENAME COLUMN model_used TO whisper_model_used")
                #     log.info("Renamed column 'model_used' to 'whisper_model_used'.")

                conn.commit()
                log.info(f"Database '{self.db_name}' initialized/verified with 'recordings' table.")
        except sqlite3.Error as e:
            log.error(f"Database initialization error for '{self.db_name}': {e}")
            raise
    
    def save_initial_recording_and_transcription(
        self,
        timestamp: datetime,
        audio_wav_bytes: bytes,
        samplerate: int,
        whisper_model_used: str,
        transcription: str | None
    ) -> int | None:
        """Saves the initial recording data (audio, transcription details) to the database,
           leaving analysis fields null. Returns the new record ID."""
        iso_timestamp = timestamp.isoformat()
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO recordings (timestamp, audio_wav, samplerate, whisper_model_used, transcription, llm_model_used, analysis_markdown)
                VALUES (?, ?, ?, ?, ?, NULL, NULL)
                """, (iso_timestamp, audio_wav_bytes, samplerate, whisper_model_used, transcription))
                conn.commit()
                record_id = cursor.lastrowid
                log.info(f"Initial recording & transcription from {iso_timestamp} (ID: {record_id}) saved to DB '{self.db_name}'.")
                return record_id
        except sqlite3.Error as e:
            log.error(f"DB error in '{self.db_name}' saving initial recording from {iso_timestamp}: {e}")
            return None

    def get_last_transcription_for_reanalysis(self) -> dict | None:
        """Fetches the ID and transcription of the most recent recording with non-empty transcription."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # Fetch the latest record that has a transcription
                cursor.execute("""
                    SELECT id, transcription, whisper_model_used, llm_model_used, analysis_markdown
                    FROM recordings 
                    WHERE transcription IS NOT NULL AND transcription != '' 
                    ORDER BY id DESC LIMIT 1
                """)
                record = cursor.fetchone()
                if record:
                    return dict(record) # Convert sqlite3.Row to dict
                return None
        except sqlite3.Error as e:
            log.error(f"Database error in '{self.db_name}' while fetching last transcription: {e}")
            return None

    def get_recordings_without_analysis(self) -> list[dict]:
        """Fetches records that have a transcription but no analysis. 
           Returns a list of dicts, each with id, transcription, timestamp, and whisper_model_used."""
        records_to_process = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, transcription, whisper_model_used
                    FROM recordings
                    WHERE transcription IS NOT NULL AND transcription != ''
                      AND (analysis_markdown IS NULL OR analysis_markdown == '')
                    ORDER BY id ASC  -- Process older ones first
                """)
                raw_records = cursor.fetchall()
                for row in raw_records:
                    record = dict(row)
                    # Convert ISO timestamp string back to datetime object for use in file naming
                    try:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    except (ValueError, TypeError) as e:
                        log.warning(f"Could not parse timestamp '{record['timestamp']}' for record ID {record['id']}. Skipping this record for batch analysis. Error: {e}")
                        continue # Skip records with bad timestamps for this batch process
                    records_to_process.append(record)
                
                if records_to_process:
                    log.info(f"Found {len(records_to_process)} records missing analysis.")
                else:
                    log.info("No records found that are missing analysis.")
                return records_to_process
        except sqlite3.Error as e:
            log.error(f"Database error in '{self.db_name}' while fetching records without analysis: {e}")
            return [] # Return empty list on error

    def get_records_with_analysis(self) -> list[dict]:
        """Fetches records that have a non-empty analysis_markdown.
           Returns a list of dicts, each with id, timestamp, llm_model_used, analysis_markdown, and title_for_file."""
        records_with_analysis = []
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, llm_model_used, analysis_markdown, title_for_file
                    FROM recordings
                    WHERE analysis_markdown IS NOT NULL AND analysis_markdown != ''
                    ORDER BY id ASC
                """)
                raw_records = cursor.fetchall()
                for row in raw_records:
                    record = dict(row)
                    try:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    except (ValueError, TypeError) as e:
                        log.warning(f"Could not parse timestamp '{record['timestamp']}' for record ID {record['id']} with analysis. Skipping. Error: {e}")
                        continue
                    records_with_analysis.append(record)
                
                if records_with_analysis:
                    log.info(f"Found {len(records_with_analysis)} records with analysis.")
                else:
                    log.info("No records found with analysis.")
                return records_with_analysis
        except sqlite3.Error as e:
            log.error(f"Database error in '{self.db_name}' while fetching records with analysis: {e}")
            return []

    def get_recording_by_id(self, recording_id: int) -> dict | None:
        """Fetches a specific recording by its ID."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,))
                record = cursor.fetchone()
                if record:
                    return dict(record)
                return None
        except sqlite3.Error as e:
            log.error(f"Database error in '{self.db_name}' while fetching record ID {recording_id}: {e}")
            return None

    def update_analysis(self, recording_id: int, llm_model_used: str, analysis_markdown: str, title_for_file: str):
        """Updates an existing recording with new analysis data and the title used for the file."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE recordings
                SET llm_model_used = ?, analysis_markdown = ?, title_for_file = ?
                WHERE id = ?
                """, (llm_model_used, analysis_markdown, title_for_file, recording_id))
                conn.commit()
                if cursor.rowcount > 0:
                    log.info(f"Analysis and title for recording ID {recording_id} updated in database '{self.db_name}'.")
                    return True
                else:
                    log.warning(f"No record found with ID {recording_id} to update in '{self.db_name}'.")
                    return False
        except sqlite3.Error as e:
            log.error(f"Database error in '{self.db_name}' while updating analysis for ID {recording_id}: {e}")
            return False

    def _generate_filename(self, title_prefix: str, timestamp_obj: datetime, extension: str = ".md") -> str:
        """Generates a filename based on title prefix and timestamp, ensuring it's valid and not too long."""
        time_str = timestamp_obj.strftime("%Y%m%d_%H%M%S")
        
        # Sanitize title_prefix for filename
        # Remove invalid characters, replace spaces with underscores
        sane_title = re.sub(r'[^\w\s-]', '', title_prefix) # Keep alphanumeric, whitespace, hyphens
        sane_title = re.sub(r'[\s_]+', '_', sane_title).strip('_') # Replace whitespace/multiple underscores with single, strip leading/trailing

        if not sane_title: # If title becomes empty after sanitization
            sane_title = "untitled"

        base_filename_without_ext = f"{time_str}_{sane_title}"
        
        # Check length before adding extension
        # Max length for the base part = MAX_FILENAME_LENGTH - len(extension)
        max_base_len = MAX_FILENAME_LENGTH - len(extension)

        if len(base_filename_without_ext) > max_base_len:
            # Truncate the title part if the full name is too long
            # Calculate how much of the title we can keep
            # len(time_str) + 1 (for underscore) + len(truncated_title) <= max_base_len
            available_for_title = max_base_len - (len(time_str) + 1)
            
            if available_for_title <= 0:
                # This should not happen if time_str and extension are reasonable
                # But as a fallback, use a very short name
                log.warning(f"Filename for '{title_prefix}' is too long even for timestamp. Using minimal name.")
                sane_title = "longname" # Fallback short title
                base_filename_without_ext = f"{time_str}_{sane_title[:available_for_title]}" if available_for_title > 0 else time_str
            else:
                truncated_title = sane_title[:available_for_title]
                base_filename_without_ext = f"{time_str}_{truncated_title}"
            
            log.warning(f"Original title '{title_prefix}' resulted in a filename part that was too long. Truncated to: '{base_filename_without_ext + extension}'")

        final_filename = f"{base_filename_without_ext}{extension}"
        
        # Final check, though truncation should handle it.
        if len(final_filename) > MAX_FILENAME_LENGTH:
            # This would imply an issue with logic or very long extension
            log.error(f"CRITICAL: Filename '{final_filename}' still too long after processing. This should not happen.")
            # Fallback to a very generic name based on timestamp only
            final_filename = f"{time_str}{extension}"
            if len(final_filename) > MAX_FILENAME_LENGTH: # if timestamp + ext is too long (unlikely)
                 final_filename = final_filename[:MAX_FILENAME_LENGTH]

        return final_filename

    def save_markdown_content(self, file_and_h1_title: str, body_markdown_content: str, timestamp_obj: datetime, type: str = "analysis") -> str | None:
        """
        Saves markdown content to a file. The H1 header is derived from file_and_h1_title.
        Args:
            file_and_h1_title (str): The title for the markdown, used for filename generation AND as the H1 header.
            body_markdown_content (str): The body of the markdown content (without H1).
            timestamp_obj (datetime): Timestamp for filename generation.
            type (str): Type of content. Used to determine save sub-path and default title prefix if file_and_h1_title is empty.
        Returns:
            str | None: The full path to the saved file, or None on failure.
        """
        # Determine base path and a title prefix for filename generation if file_and_h1_title is empty
        # However, file_and_h1_title should ideally always be provided by the caller.
        if type == "analysis" or type == "analysis_reprocessed" or type == "analysis_recovered":
            base_path = self.markdown_save_path
            # If type is for subfolder, adjust base_path here. Example:
            if type == "analysis_recovered":
                 effective_base_path = base_path / "recovered"
            elif type == "analysis_reprocessed": # Example for reprocessed, if needed differently
                 effective_base_path = base_path / "reprocessed" 
            else: # Standard "analysis"
                 effective_base_path = base_path
        else:
            log.error(f"Unknown or unsupported markdown content type for file saving: {type}")
            return None
        
        # Ensure title_prefix for filename is never empty
        title_for_filename = file_and_h1_title
        if not title_for_filename:
            log.warning(f"file_and_h1_title was empty for type '{type}'. Using a default title for filename.")
            if type == "analysis_reprocessed":
                title_for_filename = "Re-analysis"
            elif type == "analysis_recovered":
                title_for_filename = "Recovered_Analysis"
            else:
                title_for_filename = "Analysis"
            # Append timestamp to default title to ensure some uniqueness if multiple are empty
            title_for_filename = f"{title_for_filename}_{timestamp_obj.strftime('%Y%m%d_%H%M%S')}"

        filename = self._generate_filename(title_prefix=file_and_h1_title, timestamp_obj=timestamp_obj, extension=".md")
        filepath = effective_base_path / filename
        
        # Construct the full markdown content with H1 header
        full_markdown_content = f"# {file_and_h1_title}\n\n{body_markdown_content}"

        try:
            effective_base_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(full_markdown_content)
            log.info(f"Markdown content ({type}) saved to: {filepath}")
            return str(filepath)
        except IOError as e:
            log.error(f"Error saving markdown file to {filepath}: {e}")
            return None

    def save_temporary_audio_wav(self, wav_bytes: bytes, base_filename: str, samplerate: int, channels: int) -> str | None:
        """
        Saves WAV audio bytes to a temporary file.

        Args:
            wav_bytes: The audio data in WAV byte format.
            base_filename: A base name for the file (e.g., timestamp-based).
            samplerate: The sample rate of the audio.
            channels: The number of audio channels.

        Returns:
            The full path to the saved temporary file, or None on error.
        """
        if not wav_bytes:
            log.error("Cannot save empty WAV bytes to temporary file.")
            return None
        
        try:
            temp_filename = f"{base_filename}.wav"
            temp_file_path = self.temp_path / temp_filename
            
            with wave.open(str(temp_file_path), 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(AudioService.get_pyaudio_sample_size(pyaudio.paInt16))
                wf.setframerate(samplerate)
                wf.writeframes(wav_bytes)
            
            log.info(f"Temporary audio WAV saved to: {temp_file_path}")
            return str(temp_file_path)
        except Exception as e:
            log.exception(f"Error saving temporary audio WAV to {self.temp_path / base_filename}.wav: {e}")
            return None

    def delete_file(self, file_path: str) -> bool:
        """
        Deletes the file at the specified path.

        Args:
            file_path: The absolute path to the file to delete.

        Returns:
            True if deletion was successful or file did not exist, False on error.
        """
        if not file_path:
            log.warning("Attempted to delete file with empty path.")
            return False
        try:
            file_to_delete = Path(file_path)
            if file_to_delete.exists():
                file_to_delete.unlink()
                log.info(f"Successfully deleted file: {file_path}")
            else:
                log.info(f"File not found for deletion (considered successful): {file_path}")
            return True
        except FileNotFoundError: # Should be caught by Path(file_path).exists() but good for robustness
            log.info(f"File not found for deletion (considered successful): {file_path}")
            return True
        except OSError as e:
            log.error(f"Error deleting file {file_path}: {e}")
            return False
        except Exception as e: # Catch any other unexpected error
            log.exception(f"Unexpected error deleting file {file_path}: {e}")
            return False
