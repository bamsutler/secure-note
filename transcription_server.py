from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import os
import sys
from datetime import datetime
import tempfile # For creating secure temporary files/directories
import wave # To read samplerate from WAV if needed
import shutil # For saving UploadFile contents

import core_processing

# --- FastAPI App Initialization ---
app = FastAPI(title="Audio Transcription and Analysis API")

# --- Configuration ---\
# Using constants from core_processing
UPLOAD_FOLDER = 'api_uploads' # Temporary storage for uploaded files for processing
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'} # Whisper supported

# --- Application Setup ---\
# This setup runs when the module is imported by Uvicorn
try:
    core_processing.init_db()
    # Load the default Whisper model on startup
    core_processing.load_global_whisper_model(core_processing.DEFAULT_WHISPER_MODEL)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    core_processing.log.info("FastAPI application startup: Database initialized, Whisper model loaded, upload folder ensured.")
except Exception as e:
    core_processing.log.critical(f"Fatal error during FastAPI application startup: {e}")
    core_processing.log.critical("Application may not function correctly. Please check configurations and dependencies.")
    # Unlike Flask's sys.exit, with Uvicorn, the server might still start but endpoints could fail.
    # The error will be logged. Consider a more robust startup check if needed.

def allowed_file(filename: str):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/transcribe_and_analyze")
async def transcribe_and_analyze_endpoint(
    audio_file: UploadFile = File(...),
    whisper_model_size: str = Form(core_processing.DEFAULT_WHISPER_MODEL),
    ollama_model_name: str = Form(core_processing.OLLAMA_MODEL_NAME_DEFAULT),
    ollama_api_url: str = Form(core_processing.OLLAMA_API_URL_DEFAULT)
):
    core_processing.log.info(f"Received request for /transcribe_and_analyze (FastAPI)")
    
    if not audio_file: # Should be caught by FastAPI's File(...) if truly missing
        core_processing.log.warning("No audio_file part in request (FastAPI check)")
        raise HTTPException(status_code=400, detail="No audio_file part in the request")
    
    if audio_file.filename == '':
        core_processing.log.warning("No selected audio file in request (FastAPI check)")
        raise HTTPException(status_code=400, detail="No selected audio file")

    if not allowed_file(audio_file.filename):
        core_processing.log.warning(f"File type not allowed: {audio_file.filename}")
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")

    filename = audio_file.filename
    temp_file_path = ""
    temp_dir = "" # Keep track of temp_dir for cleanup
    audio_wav_bytes_for_db = b''
    samplerate_for_db = 16000 # Default

    try:
        # Save to a temporary file securely using UploadFile's methods
        temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
        temp_file_path = os.path.join(temp_dir, filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        core_processing.log.info(f"Uploaded audio file saved to temporary path: {temp_file_path}")

        # Read WAV bytes for DB and attempt to get samplerate if it's a WAV
        try:
            with open(temp_file_path, 'rb') as f_bytes:
                audio_wav_bytes_for_db = f_bytes.read()
            if filename.lower().endswith('.wav'):
                with wave.open(temp_file_path, 'rb') as wf:
                    samplerate_for_db = wf.getframerate()
                    core_processing.log.info(f"Extracted samplerate {samplerate_for_db} Hz from WAV file.")
        except Exception as e:
            core_processing.log.warning(f"Could not read WAV bytes or samplerate from {temp_file_path}: {e}. Using defaults for DB.")

        # 1. Transcribe
        core_processing.log.info(f"Starting transcription for: {temp_file_path}")
        # Ensure the global model is loaded if it wasn't (e.g., if server restarted or had issues)
        # load_global_whisper_model is idempotent so calling it again is fine.
        core_processing.load_global_whisper_model(whisper_model_size)

        transcribed_text = core_processing.transcribe_audio_file(temp_file_path, model_size=whisper_model_size)
        core_processing.log.info(f"Transcription successful for: {temp_file_path}")
        current_time_obj = datetime.now()

        # Save raw transcription to Markdown
        if transcribed_text:
            raw_transcription_md_path = core_processing.save_transcription_to_markdown(
                transcribed_text,
                current_time_obj,
                core_processing.MARKDOWN_SAVE_PATH_DEFAULT
            )
            if raw_transcription_md_path:
                core_processing.log.info(f"Raw transcription also saved to: {raw_transcription_md_path}")
            else:
                core_processing.log.warning("Failed to save raw transcription to a separate Markdown file.")

        # 2. Analyze with Ollama (if transcription successful)
        analysis_results = None
        ollama_response_md_to_save = ''
        ollama_model_for_db = ''
        markdown_file_path = None

        if transcribed_text:
            core_processing.log.info(f"Starting Ollama analysis with model: {ollama_model_name} via {ollama_api_url}")
            analysis_results = core_processing.analyze_transcription_with_ollama(
                transcribed_text,
                ollama_api_url=ollama_api_url,
                ollama_model_name=ollama_model_name
            )
            core_processing.log.info("Ollama analysis attempt complete.")

            if analysis_results and analysis_results.get("full_markdown_response") not in core_processing.FAILED_ANALYSIS_SUMMARIES and analysis_results.get("full_markdown_response", "").strip():
                core_processing.log.info("Ollama analysis successful.")
                markdown_file_path = core_processing.save_markdown_file(
                    analysis_results.get('title', 'Server Analysis'),
                    analysis_results.get('full_markdown_response', ''),
                    current_time_obj,
                    core_processing.MARKDOWN_SAVE_PATH_DEFAULT
                )
                ollama_response_md_to_save = analysis_results.get('full_markdown_response', '')
                ollama_model_for_db = ollama_model_name
            else:
                core_processing.log.warning("Ollama analysis was not successful or returned an empty/failed response.")
                if analysis_results:
                    core_processing.log.debug(f"Full Ollama response: {analysis_results.get('full_markdown_response')}")
        else:
            core_processing.log.info("Transcription was empty. Skipping Ollama analysis.")

        # 3. Save to Database
        core_processing.log.info("Saving results to database...")
        core_processing.save_to_db(
            timestamp=current_time_obj.isoformat(),
            audio_wav_bytes=audio_wav_bytes_for_db,
            samplerate=samplerate_for_db,
            model_used=whisper_model_size,
            transcription=transcribed_text,
            ollama_model_used_for_analysis=ollama_model_for_db,
            ollama_response_markdown=ollama_response_md_to_save,
            db_name=core_processing.DB_NAME_DEFAULT
        )
        core_processing.log.info("Database save complete.")

        return {
            "message": "Processing successful",
            "transcription": transcribed_text,
            "title": analysis_results.get('title', 'N/A') if analysis_results else 'N/A',
            "ollama_model_used": ollama_model_for_db,
            "markdown_file_path": markdown_file_path if markdown_file_path else "Not generated",
            "timestamp": current_time_obj.isoformat()
        }

    except RuntimeError as r_err: # Catch specific error from transcribe_audio_file if model not loaded
        core_processing.log.error(f"Runtime error during processing: {r_err}")
        raise HTTPException(status_code=500, detail=str(r_err))
    except HTTPException: # Re-raise HTTPExceptions so FastAPI handles them
        raise
    except Exception as e:
        core_processing.log.exception(f"An unexpected error occurred during processing of {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up the temporary file and directory
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                core_processing.log.info(f"Temporary file {temp_file_path} deleted.")
            except OSError as e_clean_file:
                core_processing.log.error(f"Error deleting temporary file {temp_file_path}: {e_clean_file}")
        if temp_dir and os.path.exists(temp_dir): # Ensure temp_dir was created and exists
             try:
                os.rmdir(temp_dir) # Remove the unique temp_dir
                core_processing.log.info(f"Temporary directory {temp_dir} deleted.")
             except OSError as e_clean_dir:
                # This might fail if other files were somehow created there, or due to timing.
                core_processing.log.error(f"Error deleting temporary directory {temp_dir}: {e_clean_dir}")
        if audio_file:
            await audio_file.close()


if __name__ == '__main__':
    # This block is for direct execution (python transcription_server.py)
    # For FastAPI, it's more common to run with Uvicorn:  uvicorn transcription_server:app --reload
    # However, you can add uvicorn.run here if you want `python transcription_server.py` to work directly.
    import uvicorn
    core_processing.log.info(f"Starting FastAPI server with Uvicorn. Listening on http://127.0.0.1:8000")
    core_processing.log.info("Access API docs at http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc")
    uvicorn.run(app, host="127.0.0.1", port=8000) 