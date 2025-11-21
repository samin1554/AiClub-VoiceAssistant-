import pyaudio
import wave
import numpy as np 
import os 
from faster_whisper import WhisperModel
import time

FORMAT = pyaudio.paInt16 
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 5.0
MAX_RECORDING_DURATION = 15.0
AUDIO_FILE = "temp_audoRecording.wav"


audio = pyaudio.PyAudio()

model_size = "base.en"
model = WhisperModel(model_size, device="cpu", compute_type="int8")


def record_audio():
    print("Listening for audio")
    
    stream = audio.open(format=FORMAT , channels = CHANNELS, rate = RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    start_time = time.time() 
    silent_start_time = None
    
    while True:
        data = stream.read(CHUNK , exception_on_overflow=False)
        frames.append(data)
        
        #Detect silence with rms amplitude
        current_time = time.time() 
        
        audio_data = np.frombuffer(data, dtype= np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        
        
        # checking for silence period
        if rms < SILENCE_THRESHOLD:
            if silent_start_time is None:
                silent_start_time = current_time 
        else:
            silent_start_time = None
            
        # Stop recoding if silence duration matches preset duration
        if silent_start_time and (current_time - silent_start_time) > SILENCE_DURATION and len(frames) > int(RATE / CHUNK * 0.5): #at least 2 sec of rec
            print("Slence detected")
            break 

        if (current_time - start_time) >= MAX_RECORDING_DURATION:
            print("Max recording")
            break
        
    print("Finished audio recording")
    stream.stop_stream()
    stream.close()
    
    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        
    #transcribing  recorded audio
def transcribe_recoding(filename):
    if not os.path.exists(filename):
        print(f'Error: {filename} not foujnd / exists')
        return ''
    
    print("Transcribing Audio")
    try:
        segments, info = model.transcribe(filename, beam_size=5)
        transcribe = "".join(segment.text for segment in segments).strip()
        print(F"Transcribe: {transcribe}")
        return transcribe
    except Exception as e:
        print(f" ERror {e}")
        return ''

#  saving transcribed text
def save_transcripted_text(text):
    with open("transcribed_subject.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Transcript saved to transcribed_subject.txt")
    
    
def main():
    try:
        record_audio()
        text = transcribe_recoding(AUDIO_FILE)
        if text:
            print(f"Transcribed audio : {text}")

            #  saving txt file

            save_transcripted_text(text)
            return text
        else: 
            print(f'transcribe failure')
            return ""
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}\n")
    finally:
        # Clean up 
        if os.path.exists(AUDIO_FILE): 
            os.remove(AUDIO_FILE)
            print(f"Temporary file {AUDIO_FILE} deleted.")

if __name__ == "__main__":
    main()
    
        
        