import requests
import json
from pathlib import Path
import sys # Added for sys._MEIPASS
import os # Ensure os is imported

from src.config_service import ConfigurationService
from src.logging_service import LoggingService

config_service = ConfigurationService()
log = LoggingService.get_logger(__name__)

# --- Base Provider Class (Optional, but good for structure) ---
class BaseAnalysisProvider:
    def analyze(self, transcription: str, model_config: dict = None) -> dict:
        """
        Analyzes the transcription and returns a structured dictionary.
        Args:
            transcription (str): The text to analyze.
            model_config (dict, optional): Specific configuration for the model if needed.
        Returns:
            dict: Structured analysis (e.g., {title, summary, key_topics, ...}).
                  Should include an 'error' key if analysis fails.
        """
        raise NotImplementedError("Provider must implement the 'analyze' method.")

    def _load_prompt_template(self, prompt_file_name: str) -> str | None:
        """Helper to load a prompt template from the configured directory."""
        prompt_dir_str = config_service.get('paths', 'prompt_templates', default='prompt_templates')
        
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running in a PyInstaller bundle
            base_path = Path(sys._MEIPASS)
        else:
            # Running as a normal script
            # Assuming script is run from project root or prompt_templates is relative to CWD
            # Or, more robustly for script execution if analysis_service.py is in src/ and prompts are at root:
            # base_path = Path(__file__).resolve().parent.parent 
            base_path = Path(".") # Defaulting to CWD for script mode

        prompt_dir = base_path / prompt_dir_str

        prompt_file_path = prompt_dir / prompt_file_name
        if not prompt_file_path.exists():
            log.error(f"Prompt template file not found: {prompt_file_path}")
            return None
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            log.error(f"Error loading prompt template {prompt_file_path}: {e}")
            return None

# --- Ollama Provider ---
class OllamaProvider(BaseAnalysisProvider):
    _core_ollama_instructions = None # Cache for core instructions

    def __init__(self):
        self.default_api_url = config_service.get('models', 'ollama_api_url')
        self.default_model_name = config_service.get('models', 'llm_model_name') # Unified LLM model name
        self._load_core_ollama_instructions_cached()

    def _load_core_ollama_instructions_cached(self):
        if OllamaProvider._core_ollama_instructions is None:
            core_instructions_text = self._load_prompt_template("core_instructions.txt")
            if core_instructions_text is None:
                log.error("CRITICAL: Core Ollama instructions (core_instructions.txt) not loaded. Analysis will likely fail.")
                # Store a placeholder or raise if this is absolutely critical for provider instantiation
                OllamaProvider._core_ollama_instructions = "Error: Core instructions not loaded."
            else:
                OllamaProvider._core_ollama_instructions = core_instructions_text
                log.info("Core Ollama instructions loaded and cached.")
        return OllamaProvider._core_ollama_instructions

    def _call_ollama_for_specific_analysis(
        self,
        transcription_text: str,
        section_prompt_instruction: str,
        ollama_api_url: str,
        ollama_model_name: str
    ) -> str:
        core_instructions = self._load_core_ollama_instructions_cached()
        if "Error:" in core_instructions:
            return f"{core_instructions} Specific section prompt was: {section_prompt_instruction}"

        full_prompt = f"{core_instructions}\n\n{section_prompt_instruction}\n\n---\nTranscription:\n{transcription_text}\n---"
        payload = {
            "model": ollama_model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": config_service.get('models', 'temperature', default=0.1),
                "top_k": config_service.get('models', 'top_k', default=64),
                "top_p": config_service.get('models', 'top_p', default=0.95),
                "num_ctx": config_service.get('models', 'num_ctx', default=100000)
                }
        }
        try:
            response = requests.post(ollama_api_url, json=payload, timeout=120) # 120s timeout
            response.raise_for_status() # Raise an exception for HTTP errors
            
            # Handle streamed response from Ollama
            full_generated_text = []
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        json_chunk = json.loads(decoded_line)
                        response_part = json_chunk.get("response", "")
                        full_generated_text.append(response_part)
                        # Optional: Check for 'done': True if you need to handle the end of stream explicitly
                        # if json_chunk.get('done'):
                        #     break 
                    except json.JSONDecodeError as e_json_chunk:
                        log.warning(f"Skipping non-JSON line from Ollama stream: {decoded_line}. Error: {e_json_chunk}")
                        continue # Skip lines that are not valid JSON

            generated_text = "".join(full_generated_text).strip()
            log.debug(f"Ollama full generated text for section: {generated_text[:100]}...")
            return generated_text
        except requests.exceptions.RequestException as e:
            log.error(f"Ollama API request failed for model {ollama_model_name} at {ollama_api_url}: {e}")
            return f"Error: Ollama API request failed - {e}"
        except json.JSONDecodeError as e:
            log.error(f"Failed to decode JSON response from Ollama: {e}. Response text: {response.text[:200]}...")
            return f"Error: Ollama returned invalid JSON - {e}"

    def analyze(self, transcription: str, model_config: dict = None) -> dict:
        effective_config = model_config or {}
        api_url = effective_config.get('ollama_api_url', self.default_api_url)
        model_name = effective_config.get('ollama_model_name', self.default_model_name)

        if not transcription or not transcription.strip():
            log.warning("OllamaProvider: Transcription is empty. Skipping analysis.")
            return {
                "title": "Transcription Empty", "summary": "", "key_topics": [],
                "action_items": [], "open_questions": [], "model_used": model_name,
                "error": "Transcription was empty."
            }

        analysis_results = {"model_used": model_name}
        sections_and_prompts = {
            "title": "title_prompt.txt",
            "summary": "summary_prompt.txt",
            "key_topics": "key_topics_prompt.txt",
            "action_items": "action_items_prompt.txt",
            "open_questions": "open_questions_prompt.txt"
        }
        any_section_successful = False

        for section_key, prompt_file in sections_and_prompts.items():
            section_prompt_text = self._load_prompt_template(prompt_file)
            if not section_prompt_text:
                analysis_results[section_key] = f"Error: Prompt file '{prompt_file}' not found or empty."
                analysis_results["error"] = analysis_results.get("error", "") + f"Prompt for {section_key} missing. "
                continue
            
            log.info(f"OllamaProvider: Analyzing '{section_key}' for transcription...")
            generated_text = self._call_ollama_for_specific_analysis(
                transcription, section_prompt_text, api_url, model_name
            )
            if "Error:" in generated_text:
                analysis_results[section_key] = generated_text # Store error message for this section
                analysis_results["error"] = analysis_results.get("error", "") + f"{section_key} analysis failed. "
            else:
                analysis_results[section_key] = generated_text
                any_section_successful = True
        
        # Assemble a full markdown response (simplified version from core_processing)
        # In a refined version, this markdown assembly could be more sophisticated.
        full_markdown_parts = []
        if not analysis_results.get("error") or any_section_successful:
            if analysis_results.get('title') and "Error:" not in analysis_results.get('title'):
                full_markdown_parts.append(f"# {analysis_results.get('title')}")
            for key in ["summary", "key_topics", "action_items", "open_questions"]:
                content = analysis_results.get(key)
                if content and "Error:" not in content:
                    # Capitalize section headers for markdown
                    header = key.replace("_", " ").title()
                    full_markdown_parts.append(f"## {header}\n{content}")
            analysis_results["full_markdown_response"] = "\n\n".join(full_markdown_parts) if full_markdown_parts else "No analysis content generated."
        else:
            analysis_results["full_markdown_response"] = f"Ollama analysis failed for all sections. Error(s): {analysis_results.get('error')}"

        if not any_section_successful and "error" not in analysis_results:
             analysis_results["error"] = "Analysis completed but all sections resulted in errors or empty content."

        return analysis_results

class AnalysisService:
    def __init__(self, provider_type: str = "ollama"):
        """
        Initializes the AnalysisService with a specific provider.
        Args:
            provider_type (str): 'ollama' or 'local_llm'. Defaults to 'ollama'.
        """
        self.provider_type = provider_type.lower()
        if self.provider_type == "ollama":
            try:
                self.provider = OllamaProvider()
                log.info("AnalysisService initialized with OllamaProvider.")
            except Exception as e:
                log.error(f"Failed to initialize OllamaProvider for AnalysisService: {e}")
                self.provider = None # Provider failed to init
        else:
            log.error(f"Invalid provider type for AnalysisService: {provider_type}. Supported: 'ollama', 'local_llm'.")
            self.provider = None
            # raise ValueError(f"Unsupported analysis provider: {provider_type}")

    def analyze_transcription(self, transcription: str, model_config: dict = None) -> dict:
        """
        Analyzes the transcription using the configured provider.
        Args:
            transcription (str): The text content to analyze.
            model_config (dict, optional): Provider-specific model configuration.
                                         For Ollama: {'ollama_api_url', 'ollama_model_name'}
                                         For LocalLLM: (Currently uses instance config)
        Returns:
            dict: A dictionary containing the analysis results or an error message.
                  Example: {'title': '...', 'summary': '...', 'full_markdown_response': '...', 'model_used': '...'}
        """
        if self.provider is None:
            log.error(f"AnalysisService cannot analyze: Provider '{self.provider_type}' was not initialized successfully.")
            return {"error": f"Analysis provider '{self.provider_type}' not available.", "full_markdown_response": "Provider not initialized."}

        if not transcription or not transcription.strip():
            log.warning("AnalysisService: Transcription is empty. Cannot analyze.")
            return {"error": "Transcription was empty.", "full_markdown_response": "Transcription was empty."}
        
        log.info(f"AnalysisService: Request to analyze transcription using {self.provider_type} provider.")
        try:
            analysis_result = self.provider.analyze(transcription, model_config=model_config)
            log.info(f"AnalysisService: Analysis by {self.provider_type} completed.")
            # Ensure essential keys are present, even if empty, for consistent structure
            # Providers should ideally return these, but this is a safeguard.
            for key in ["title", "summary", "key_topics", "action_items", "open_questions", "model_used", "full_markdown_response", "error"]:
                if key not in analysis_result:
                    if key in ["key_topics", "action_items", "open_questions"]:
                        analysis_result[key] = [] if key != "error" else None # list for multivalue, None for error if missing
                    else:
                        analysis_result[key] = None # Default to None if key is missing
            
            return analysis_result
        except Exception as e:
            log.exception(f"AnalysisService: Unexpected error during analysis with {self.provider_type}: {e}")
            return {"error": f"Unexpected error in AnalysisService: {e}", "full_markdown_response": f"Analysis failed: {e}"}
