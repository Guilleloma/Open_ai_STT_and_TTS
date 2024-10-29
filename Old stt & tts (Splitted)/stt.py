# stt.py
import requests
import sounddevice as sd
from scipy.io.wavfile import write
import os

def read_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        return f.read().strip()

def record_audio(duration=8, fs=44100, output_file='audio.wav'):
    print("Grabando...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Esperar a que la grabación termine
    print("Grabación terminada.")

    # Guardar el audio en un archivo WAV
    write(output_file, fs, audio_data)
    return output_file

def speech_to_text(audio_file):
    api_key = read_api_key()
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    files = {
        'file': (audio_file, open(audio_file, 'rb')),
        'model': (None, 'whisper-1'),
    }

    response = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, files=files)

    if response.status_code == 200:
        transcript = response.json()['text']
        return transcript
    else:
        print(f"Error en la transcripción: {response.status_code}, {response.text}")
        return None
