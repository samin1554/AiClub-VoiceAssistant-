# Jarvis AI Assistant

A voice-controlled AI assistant using:
- **OpenAI Whisper** for speech-to-text
- **OpenAI GPT** for intelligent responses
- **ElevenLabs** for natural text-to-speech

## Setup

### 1. Install Dependencies

```powershell
cd C:\Users\Alyso\Projects\jarvis
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit the `.env` file and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

Get your keys from:
- OpenAI: https://platform.openai.com/api-keys
- ElevenLabs: https://elevenlabs.io/app/settings/api-keys

### 3. Run Jarvis

```powershell
python jarvis.py
```

## Usage

- Press **ENTER** to start recording (default 5 seconds)
- Type a **number** to change recording duration
- Type **'clear'** to reset conversation history
- Type **'quit'** or **'exit'** to stop

## Troubleshooting

### Microphone not working
Make sure your microphone is set as the default recording device in Windows Sound settings.

### Audio playback issues
Install ffmpeg if you encounter playback problems:
```powershell
winget install ffmpeg
```

