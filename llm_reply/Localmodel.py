import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def load_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model


def process_transcribed_text(model):
    if not model:
        return "Model not initialized."

    # Expect file in project root (one level up)
    filename = "../transcribed_subject.txt"
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except FileNotFoundError:
        return f"File {filename} not found at {file_path}"

    if not text:
        return "No text found in file"

    # Gemini can handle larger context, but keeping it reasonable
    prompt = f"You are a helpful assistant. Process this text and provide a helpful response: {text}"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"


if __name__ == "__main__":
    model = load_model()
    if model:
        result = process_transcribed_text(model)
        print(result)
