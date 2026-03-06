import os
import io
import tempfile
import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write as write_wav
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
RECORD_SECONDS = 5  # Default recording duration
ELEVENLABS_VOICE = "Rachel"  # Default voice, can be changed

# Conversation history for context
conversation_history = [
    {"role": "system", "content": "You are Jarvis, a helpful and intelligent AI assistant. Be concise, friendly, and helpful. Keep responses brief but informative."}
]


def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    """Record audio from microphone."""
    print(f"\n🎤 Listening for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete.")
    return audio


def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    # Save audio to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        write_wav(temp_file.name, SAMPLE_RATE, audio)
        temp_path = temp_file.name
    
    try:
        with open(temp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript.strip()
    finally:
        os.unlink(temp_path)  # Clean up temp file


def get_llm_response(user_message: str) -> str:
    """Get response from GPT model."""
    conversation_history.append({"role": "user", "content": user_message})
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # Using GPT-4o as the latest available model
        messages=conversation_history,
        max_tokens=500,
        temperature=0.7
    )
    
    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message


def speak(text: str) -> None:
    """Convert text to speech using ElevenLabs and play it."""
    print(f"\n🔊 Jarvis: {text}\n")
    
    audio = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
        model_id="eleven_turbo_v2_5",  # Updated model for free tier
        output_format="mp3_44100_128"
    )
    
    # Collect all audio chunks
    audio_bytes = b"".join(audio)
    
    # Save to temporary file and play with sounddevice
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name
    
    try:
        # Read audio file and play
        data, samplerate = sf.read(temp_path)
        sd.play(data, samplerate)
        sd.wait()  # Wait until audio finishes playing
    finally:
        os.unlink(temp_path)  # Clean up temp file


def main():
    """Main loop for Jarvis assistant."""
    print("=" * 50)
    print("🤖 JARVIS AI ASSISTANT")
    print("=" * 50)
    print("\nCommands:")
    print("  - Press ENTER to start recording (5 seconds)")
    print("  - Type a number + ENTER to set recording duration")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'clear' to clear conversation history")
    print("=" * 50)
    
    # Greeting
    speak("Hello! I'm Jarvis, your AI assistant. How can I help you today?")
    
    while True:
        try:
            user_input = input("\n⏎ Press ENTER to speak (or type command): ").strip().lower()
            
            if user_input in ['quit', 'exit', 'q']:
                speak("Goodbye! Have a great day.")
                break
            
            if user_input == 'clear':
                conversation_history.clear()
                conversation_history.append({
                    "role": "system", 
                    "content": "You are Jarvis, a helpful and intelligent AI assistant. Be concise, friendly, and helpful."
                })
                print("🗑️ Conversation history cleared.")
                continue
            
            # Set recording duration if number provided
            duration = RECORD_SECONDS
            if user_input.isdigit():
                duration = int(user_input)
                print(f"Recording duration set to {duration} seconds.")
            
            # Record and transcribe
            audio = record_audio(duration)
            user_text = transcribe_audio(audio)
            
            if not user_text:
                print("❌ Could not understand audio. Please try again.")
                continue
            
            print(f"\n📝 You said: {user_text}")
            
            # Get LLM response
            response = get_llm_response(user_text)
            
            # Speak the response
            speak(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()

