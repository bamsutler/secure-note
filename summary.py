import sqlite3
from datetime import datetime
import re
import torch # Added for PyTorch
from transformers import AutoTokenizer, AutoModelForCausalLM # Added for Hugging Face

DB_NAME = "transcriptions.db"

# --- LLM Configuration ---
# IMPORTANT: Replace this with the actual path to your downloaded Hugging Face model
# For example: "/path/to/your/downloaded/llama-3-8b-instruct"
# Or a Hugging Face model ID if you want to download it on the fly (requires internet and disk space)
# For example: "meta-llama/Llama-3-8B-Instruct"
MODEL_PATH = "meta-llama/Llama-3-8B-Instruct" # <<< CHANGE THIS PATH/ID

# Global variables for model and tokenizer to avoid reloading them for each transcription
llm_model = None
llm_tokenizer = None
llm_device = None

def load_llm_model_and_tokenizer():
    """Loads the LLM model and tokenizer. Tries to use GPU if available."""
    global llm_model, llm_tokenizer, llm_device
    if llm_model is not None and llm_tokenizer is not None:
        return True
    try:
        print(f"Loading tokenizer from: {MODEL_PATH}...")
        llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        print(f"Loading model from: {MODEL_PATH}...")
        # device_map="auto" will try to use GPU if available, otherwise CPU.
        # For more control, you can set device_map to a specific device e.g. {"" : "cuda:0"} or {"" : "cpu"}
        llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16, # or torch.float16, depending on model/GPU. Use torch.float32 for CPU if issues.
            device_map="auto", # Automatically uses GPU if available
            trust_remote_code=True # Required for some models like Llama 3
        )
        # If not using device_map="auto", explicitly move model to device:
        # llm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # llm_model.to(llm_device)
        # print(f"Model loaded on device: {llm_model.device}") 
        # The device is implicitly handled by device_map="auto", check via llm_model.device after loading.
        print(f"Model loaded. Main device: {llm_model.device}")
        llm_model.eval() # Set model to evaluation mode
        return True
    except Exception as e:
        print(f"Error loading LLM model/tokenizer from '{MODEL_PATH}': {e}")
        print("Please ensure you have downloaded the model to the specified path, or that the Hugging Face ID is correct.")
        print("You might also need to install additional dependencies like 'sentencepiece' or 'protobuf'.")
        print("For Llama 3 models, ensure you have accepted the license on Hugging Face and are logged in via `huggingface-cli login` if downloading on the fly.")
        llm_model = None # Ensure it's None if loading failed
        llm_tokenizer = None
        return False

def connect_db(db_name=DB_NAME):
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database '{db_name}': {e}")
        return None

def get_all_transcriptions(conn):
    """Fetches all transcriptions from the database."""
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, transcription, model_used FROM recordings ORDER BY timestamp DESC")
        records = cursor.fetchall()
        return records
    except sqlite3.Error as e:
        print(f"Error fetching transcriptions: {e}")
        return []

def generate_llm_prompt(transcription_text):
    # This is a basic prompt template for Llama 3 Instruct. 
    # You MAY need to adjust this based on the specific LLaMA model you are using.
    # Some models use different special tokens or structures.
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes transcriptions. Extract a concise summary, key topics, to-do items, and open questions. Present each section clearly. If a section is empty, state 'None found'."},
        {
            "role": "user",
            "content": f"""Please analyze the following transcription:

---
{transcription_text}
---

Provide your analysis in the following format:

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
"""
        }
    ]
    # The apply_chat_template function handles the specific formatting for the model.
    prompt_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt_text

def parse_llm_output(generated_text):
    """Parses the LLM's output to extract structured information."""
    analysis = {
        "summary": "Could not parse summary.",
        "key_topics": [],
        "todos": [],
        "open_questions": []
    }
    try:
        # Attempt to find the start of the model's actual response if the prompt is included
        # For Llama 3 instruct, the generation starts after the last assistant prompt template.
        # A simpler approach if the model directly gives the structured output:
        sections = {
            "Summary": "summary",
            "Key Topics": "key_topics",
            "To-Do Items": "todos",
            "Open Questions": "open_questions"
        }
        current_section_key = None
        
        # Find the start of the actual response part if the model echoes parts of the prompt.
        # This might need adjustment. For Llama 3, the good stuff starts after the user's prompt effectively.
        # A common marker can be the first bolded header if your prompt asks for it.
        # Let's assume generated_text contains the model's reply directly after the chat template processing.

        for line in generated_text.splitlines():
            line = line.strip()
            if not line: continue

            matched_section = False
            for header, key in sections.items():
                if line.lower().startswith(f"**{header.lower()}**") or line.lower().startswith(header.lower()):
                    current_section_key = key
                    matched_section = True
                    # Special handling for summary as it's usually a paragraph, not a list
                    if key == "summary":
                        summary_text = line.replace(f"**{header}**", "", 1).replace(header, "", 1).strip(": ").strip()
                        if summary_text: analysis[key] = summary_text 
                        else: analysis[key] = "" # Initialize for multi-line summary
                    break
            if matched_section:
                continue

            if current_section_key:
                if current_section_key == "summary":
                    # Append to summary if it's multi-line and already initialized
                    if analysis["summary"]: analysis["summary"] += " " + line
                    elif line : analysis["summary"] = line # If summary starts on next line
                elif line.startswith("-") or line.startswith("*") or (current_section_key and line): # Treat non-empty lines as items under current section
                    item = line.lstrip("-* ").strip()
                    if item and item.lower() != "none found":
                        analysis[current_section_key].append(item)
        
        # If summary is still the default and nothing was parsed, set to a message.
        if analysis["summary"] == "Could not parse summary." and not any(analysis["key_topics"] + analysis["todos"] + analysis["open_questions"]):
            analysis["summary"] = "LLM output parsing failed or content was not in expected format. Raw output below:"
            analysis["key_topics"].append(f"RAW: {generated_text[:500]}...") # Add raw output snippet
        elif analysis["summary"] == "": # if it was initialized for multiline but got nothing
             analysis["summary"] = "Summary not found or empty in LLM output."

    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        analysis["summary"] = "Error during parsing of LLM output."
        analysis["key_topics"].append(f"RAW (parsing error): {generated_text[:500]}...")
    return analysis

def analyze_text_with_llm(transcription_text):
    if not llm_model or not llm_tokenizer:
        print("LLM model or tokenizer not loaded. Skipping analysis.")
        return {"summary": "LLM not loaded.", "key_topics": [], "todos": [], "open_questions": []}

    print(f"\n--- Analyzing text with local LLM ({MODEL_PATH}) ---")
    
    prompt = generate_llm_prompt(transcription_text)
    
    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4000) # Max length depends on model
        inputs = {k: v.to(llm_model.device) for k, v in inputs.items()} # Ensure inputs are on the same device as model

        print("Generating response...")
        with torch.no_grad(): # Disable gradient calculations for inference
            # Adjust max_new_tokens as needed. This is the length of the generated summary, not input + output.
            # Common Llama 3 context length is 8k tokens. We set input max_length above.
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=1024, # Max tokens for the generated response
                eos_token_id=llm_tokenizer.eos_token_id,
                pad_token_id=llm_tokenizer.pad_token_id,
                do_sample=True, # You can try False for more deterministic output
                temperature=0.6, # Adjust for creativity vs. factuality
                top_p=0.9,
            )
        
        # Decoding the output: outputs[0] is the full sequence (prompt + generation)
        # We need to decode only the generated part.
        # For batch size 1 (inputs.input_ids.shape[0]), this works:
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print("--- LLM Analysis Complete ---")
        print(f"Raw LLM Output:\n{response_text}")
        return parse_llm_output(response_text)

    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return {"summary": f"LLM inference error: {e}", "key_topics": [], "todos": [], "open_questions": []}

def main():
    print("Transcription Summarizer with Local LLM")
    print(f"Attempting to use model: '{MODEL_PATH}'")
    print("IMPORTANT: Ensure MODEL_PATH is correctly set in the script and the model is downloaded.")

    if not load_llm_model_and_tokenizer():
        print("Failed to load LLM model. Exiting.")
        return

    print(f"Reading transcriptions from '{DB_NAME}'")
    conn = connect_db()
    if not conn:
        return

    transcriptions = get_all_transcriptions(conn)
    conn.close()

    if not transcriptions:
        print("No transcriptions found in the database.")
        return

    print(f"\nFound {len(transcriptions)} transcription(s) to process.\n")

    for record in transcriptions:
        print("="*70)
        print(f"Record ID: {record['id']}")
        print(f"Timestamp: {record['timestamp']}")
        print(f"Whisper Model Used (for transcription): {record['model_used']}")
        print(f"Original Transcription:\n'''{record['transcription']}'''")

        if record['transcription'] and record['transcription'].strip():
            analysis = analyze_text_with_llm(record['transcription'])
            
            print("\n--- Parsed Analysis ---")
            print("Summary:")
            print(analysis["summary"])
            
            print("\nKey Topics:")
            if analysis["key_topics"]:
                for topic in analysis["key_topics"]:
                    print(f"- {topic}")
            else: print("None found or parsed.")
                
            print("\nTo-Do Items:")
            if analysis["todos"]:
                for todo in analysis["todos"]:
                    print(f"- {todo}")
            else: print("None found or parsed.")
                
            print("\nOpen Questions:")
            if analysis["open_questions"]:
                for question in analysis["open_questions"]:
                    print(f"- {question}")
            else: print("None found or parsed.")
        else:
            print("\nTranscription is empty, skipping analysis.")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
