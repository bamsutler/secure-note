import whisper
import sqlite3
from datetime import datetime
import io
import wave
import requests
import json
import logging
from rich.logging import RichHandler
from rich.console import Console
import os
import re
import sys # For sys.exit in server startup if needed
import numpy as np

# --- Logger Setup ---
FORMAT = "%(message)s"
# Configure logger for the library/module
# This allows applications using this module to define their own handlers
# if they wish, or rely on this basic configuration.
log = logging.getLogger("core_processing")
if not log.handlers: # Avoid adding multiple handlers if already configured by an app
    log.setLevel(logging.INFO)
    handler = RichHandler(rich_tracebacks=True)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt="[%X]"))
    log.addHandler(handler)

console = Console()

# --- Configuration Constants ---
DB_NAME_DEFAULT = "transcriptions.db"
MARKDOWN_SAVE_PATH_DEFAULT = "transcription_notes"
OLLAMA_API_URL_DEFAULT = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME_DEFAULT = "llama3.2:latest"
DEFAULT_WHISPER_MODEL = "base"

FAILED_ANALYSIS_SUMMARIES = [
    "Ollama analysis failed. See logs for details.",
    "Ollama returned empty response.",
    "LLM output parsing failed or content was not in expected format.",
    "Could not parse summary."
]

# --- Global Whisper Instance ---
whisper_instance_global = None
# ---

def load_global_whisper_model(model_size=DEFAULT_WHISPER_MODEL):
    """Loads the Whisper model globally or returns the existing instance."""
    global whisper_instance_global
    if whisper_instance_global is None:
        try:
            log.info(f"Loading Whisper model '{model_size}' globally... (This may take a moment)")
            whisper_instance_global = whisper.load_model(model_size)
            log.info("Global Whisper model loaded.")
        except Exception as e:
            log.error(f"Error loading global Whisper model: {e}")
            whisper_instance_global = None # Ensure it's None on failure
            raise # Re-raise the exception
    return whisper_instance_global

def transcribe_audio_file(audio_file_path, model_size=DEFAULT_WHISPER_MODEL):
    """
    Transcribes an audio file using the globally loaded Whisper model.
    Returns the transcription text.
    """
    instance = load_global_whisper_model(model_size)
    if instance is None:
        log.error("Whisper model is not loaded. Cannot transcribe.")
        raise RuntimeError("Whisper model failed to load.")

    try:
        log.info(f"Transcribing audio file: {audio_file_path} with model '{model_size}'...")
        result = instance.transcribe(audio_file_path, fp16=False) # fp16 for CPU
        transcribed_text = result["text"].strip()
        log.info("Transcription complete.")
        return transcribed_text
    except Exception as e:
        log.error(f"Error during transcription of {audio_file_path}: {e}")
        raise


def init_db(db_name=DB_NAME_DEFAULT):
    """Initializes the SQLite database and creates/alters the recordings table."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recordings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        audio_wav BLOB NOT NULL,
        samplerate INTEGER NOT NULL,
        model_used TEXT NOT NULL, -- Whisper model used for transcription
        transcription TEXT,
        ollama_model_used TEXT, -- Ollama model used for analysis
        ollama_response_markdown TEXT -- Stores the full markdown from Ollama
    )
    """)
    table_info = cursor.execute("PRAGMA table_info(recordings)").fetchall()
    column_names = [info[1] for info in table_info]
    if 'ollama_response_markdown' not in column_names:
        cursor.execute("ALTER TABLE recordings ADD COLUMN ollama_response_markdown TEXT")
    cols_to_drop = ['summary', 'key_topics', 'todos', 'open_questions']
    current_columns = set(column_names)
    for col in cols_to_drop:
        if col in current_columns:
            try:
                cursor.execute(f"ALTER TABLE recordings DROP COLUMN {col}")
                log.info(f"Column '{col}' dropped from 'recordings' table in '{db_name}'.")
            except sqlite3.OperationalError as e:
                log.warning(f"Could not drop column '{col}' from '{db_name}': {e}. This might be an old SQLite version or column was already removed.")
    conn.commit()
    conn.close()
    log.info(f"Database '{db_name}' initialized/updated.")


def get_last_transcription_for_reanalysis(db_name=DB_NAME_DEFAULT):
    """Fetches the ID and transcription of the most recent recording with non-empty transcription."""
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, transcription FROM recordings WHERE transcription IS NOT NULL AND transcription != '' ORDER BY id DESC LIMIT 1")
        record = cursor.fetchone()
        if record:
            return {"id": record["id"], "transcription": record["transcription"]}
        return None
    except sqlite3.Error as e:
        log.error(f"Database error in '{db_name}' while fetching last transcription: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_db_with_new_analysis(recording_id, ollama_model_used, ollama_response_markdown, db_name=DB_NAME_DEFAULT):
    """Updates an existing recording with new analysis data."""
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE recordings
        SET ollama_model_used = ?, ollama_response_markdown = ?
        WHERE id = ?
        """, (ollama_model_used, ollama_response_markdown, recording_id))
        conn.commit()
        if cursor.rowcount > 0:
            log.info(f"Analysis for recording ID {recording_id} updated in database '{db_name}'.")
        else:
            log.warning(f"No record found with ID {recording_id} to update in '{db_name}'.")
    except sqlite3.Error as e:
        log.error(f"Database error in '{db_name}' while updating analysis for ID {recording_id}: {e}")
    finally:
        if conn:
            conn.close()

def save_to_db(timestamp, audio_wav_bytes, samplerate, whisper_model_used, transcription,
               ollama_model_used_for_analysis, ollama_response_markdown,
               db_name=DB_NAME_DEFAULT):
    """Saves the recording, transcription, and Ollama markdown response to the SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO recordings (timestamp, audio_wav, samplerate, model_used, transcription,
                              ollama_model_used, ollama_response_markdown)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, audio_wav_bytes, samplerate, whisper_model_used, transcription,
              ollama_model_used_for_analysis, ollama_response_markdown))
        conn.commit()
        log.info(f"Recording and analysis markdown from {timestamp} saved to database '{db_name}'.")
    except sqlite3.Error as e:
        log.error(f"Database error in '{db_name}' while saving: {e}")
    finally:
        if conn:
            conn.close()

def generate_ollama_prompt(transcription_text):
    """Generates a prompt for Ollama to analyze the transcription."""
    return f"""Please analyze the following transcription:

---
{transcription_text}
---

Provide your analysis in the following format:

**Title:**
[A concise and descriptive title for this meeting/transcription]

**Summary:**
[Your concise summary here]

**Key Topics:**
- [Topic 1]
- [Topic 2]

**To-Do Items:**
- [Action item 1]
- [Action item 2]

**Open Questions:**
- [Question 1]
- [Question 2]

If a section is empty or no relevant items are found, clearly state 'None found' under that section.
"""

def parse_ollama_response(generated_text):
    """
    Parses the Ollama LLM's output to extract a title and return the full markdown content.
    """
    title = "Meeting Analysis" # Default title
    full_markdown_response = generated_text.strip()
    try:
        lines = generated_text.splitlines()
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            if line_lower.startswith("**title:**") or line_lower.startswith("title:"):
                potential_title = line.split(":", 1)[1].strip()
                if potential_title:
                    title = potential_title
                break
            elif (line_lower == "**title**" or line_lower == "title") and i + 1 < len(lines):
                potential_title_next_line = lines[i+1].strip()
                if potential_title_next_line and not (potential_title_next_line.lower().startswith("**") or potential_title_next_line.lower().startswith("-")):
                    title = potential_title_next_line
                    break
        if not full_markdown_response:
            title = "Empty Ollama Response"
        elif title == "Meeting Analysis" and "summary:" not in full_markdown_response.lower():
             log.warning("Could not parse a specific title from Ollama response; using default or indicating parse issue if response seems malformed.")
    except Exception as e:
        log.error(f"Error parsing title from LLM output: {e}")
    return {"title": title, "full_markdown_response": full_markdown_response}

def analyze_transcription_with_ollama(transcription_text, ollama_api_url=OLLAMA_API_URL_DEFAULT, ollama_model_name=OLLAMA_MODEL_NAME_DEFAULT):
    """Analyzes the transcription text using Ollama and returns structured data."""
    if not ollama_model_name:
        log.warning("[OLLAMA_VISIBILITY] OLLAMA_MODEL_NAME is not set. Skipping analysis.")
        return {"title": "Analysis Skipped", "full_markdown_response": FAILED_ANALYSIS_SUMMARIES[0]}

    log.info(f"----- Analyzing transcription with Ollama (Model: {ollama_model_name}) -----")
    prompt = generate_ollama_prompt(transcription_text)
    payload = {
        "model": ollama_model_name,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.5}
    }
    log.debug(f"""[OLLAMA_VISIBILITY] Request Payload to {ollama_api_url}:
{json.dumps(payload, indent=2)}""")
    accumulated_response_content = []
    full_generated_text = ""
    try:
        # Make a POST request to the Ollama API with streaming
        with requests.post(ollama_api_url, json=payload, stream=True, timeout=300) as response: # 5 min timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            log.info(f"[OLLAMA_VISIBILITY] Ollama API response status: {response.status_code}")
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        json_chunk = json.loads(decoded_line)
                        response_part = json_chunk.get("response", "")
                        accumulated_response_content.append(response_part)
                        full_generated_text += response_part
                        # Optionally print stream to console for real-time feedback
                        # print(response_part, end='', flush=True) 
                        if json_chunk.get("done"):
                            log.info("[OLLAMA_VISIBILITY] Ollama stream finished.")
                            break
                    except json.JSONDecodeError:
                        log.error(f"[OLLAMA_VISIBILITY] Error decoding JSON from Ollama stream: {decoded_line}")
                    except Exception as e:
                         log.error(f"[OLLAMA_VISIBILITY] Error processing Ollama stream chunk: {e}")
    except requests.exceptions.RequestException as e:
        log.exception(f"[OLLAMA_VISIBILITY] Ollama API request failed: {e}")
        return {"title": "Analysis Failed", "full_markdown_response": FAILED_ANALYSIS_SUMMARIES[0]}
    except Exception as e:
        log.exception(f"[OLLAMA_VISIBILITY] An unexpected error during Ollama analysis: {e}")
        return {"title": "Analysis Failed", "full_markdown_response": FAILED_ANALYSIS_SUMMARIES[0]}
    finally:
        log.info("----- Ollama analysis stream finished -----")
    return parse_ollama_response(full_generated_text)


def save_transcription_to_markdown(transcription_text: str, timestamp_obj: datetime, base_path: str = MARKDOWN_SAVE_PATH_DEFAULT) -> str | None:
    """
    Saves the raw transcription text to a Markdown file.

    Args:
        transcription_text: The text of the transcription.
        timestamp_obj: A datetime object for generating the filename.
        base_path: The directory where the file will be saved.

    Returns:
        The path to the saved file, or None if saving failed.
    """
    if not transcription_text or not transcription_text.strip():
        log.info("Transcription text is empty. Skipping Markdown file save for raw transcription.")
        return None

    try:
        os.makedirs(base_path, exist_ok=True)
        filename_timestamp = timestamp_obj.strftime("%Y%m%d_%H%M%S")
        # Sanitize title for filename - keep it simple for raw transcription
        filename = f"transcription_{filename_timestamp}.md"
        filepath = os.path.join(base_path, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Transcription - {timestamp_obj.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(transcription_text)
        
        log.info(f"Raw transcription saved to Markdown file: {filepath}")
        return filepath
    except Exception as e:
        log.error(f"Error saving raw transcription to Markdown file: {e}")
        return None

def save_markdown_file(title, markdown_content, timestamp_obj, base_path=MARKDOWN_SAVE_PATH_DEFAULT):
    """Saves the markdown content to a file with a sanitized title and date."""
    if not markdown_content or not markdown_content.strip():
        log.warning("Markdown content is empty. Skipping file save.")
        return None
    try:
        os.makedirs(base_path, exist_ok=True)
        sanitized_title = re.sub(r'[^\w\s-]', '', title).strip()
        sanitized_title = re.sub(r'[-\s]+', '_', sanitized_title)
        if not sanitized_title: sanitized_title = "Untitled_Meeting"
        date_str = timestamp_obj.strftime("%Y-%m-%d")
        filename = f"{date_str}_{sanitized_title}.md"
        filepath = os.path.join(base_path, filename)
        if len(filepath) > 255:
            sanitized_title = sanitized_title[:50]
            filename = f"{date_str}_{sanitized_title}_TRUNCATED.md"
            filepath = os.path.join(base_path, filename)
            log.warning(f"Original filename was too long, truncated to: {filename}")
        with open(filepath, 'w', encoding='utf-8') as f:
            # The Ollama response should already contain a title, so just write the content
            f.write(markdown_content)
        log.info(f"Markdown analysis saved to: {filepath}")
        return filepath
    except Exception as e:
        log.error(f"Error saving markdown file '{filepath if 'filepath' in locals() else ''}': {e}")
        return None

# --- Helper for server to create WAV bytes if needed ---
def convert_float32_to_wav_bytes(audio_data_float32, samplerate):
    """Converts a NumPy array of float32 audio data to WAV file bytes."""
    if audio_data_float32 is None or audio_data_float32.size == 0:
        log.warning("Audio data is empty, cannot convert to WAV bytes.")
        return b''
    try:
        audio_data_int16 = (audio_data_float32.flatten() * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(samplerate)
            wf.writeframes(audio_data_int16.tobytes())
        return wav_buffer.getvalue()
    except Exception as e:
        log.error(f"Error converting float32 audio to WAV bytes: {e}")
        return b'' 