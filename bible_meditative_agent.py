import json
import sys
import os
from transformers import pipeline

# Constants
SNIPPET_CHAR_LIMIT = 1000  # Number of characters to send to the model

# Read HF token and model name from environment variables
hf_token = os.getenv("HF_TOKEN")
model_name = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM3-3B")

def load_chapter(file_path, chapter_number):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    chapter_key = str(chapter_number)
    return data[chapter_key]

def main():
    if len(sys.argv) != 3:
        print("Usage: python bible_meditative_agent.py <chapter_number> <bible_file>")
        sys.exit(1)

    chapter_number = sys.argv[1]
    bible_file = sys.argv[2]

    print(f"[INFO] Loading chapter {chapter_number} from {bible_file}...")
    chapter_text = load_chapter(bible_file, chapter_number)
    
    print(f"[INFO] Full chapter length: {len(chapter_text)}")

    snippet = chapter_text[:SNIPPET_CHAR_LIMIT]

    prompt = f"""You are a meditative Bible study assistant. Reflect on this passage:

{snippet}

What spiritual insights can be drawn from this passage?
"""

    # Load the pipeline
    generator = pipeline("text-generation", model=model_name, token=hf_token)

    print("\n[INFO] Generating meditative response...\n")
    output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    print("[Meditative Reflection]\n")
    print(output[0]['generated_text'])

if __name__ == "__main__":
    main()
