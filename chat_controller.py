# chat_controller.py

import openai  # Usamos openai para obtener la versión, no OpenAI como clase
from openai import OpenAI  # Cambiado según la migración
import sounddevice as sd
from scipy.io.wavfile import write
import requests
from pydub import AudioSegment
from pydub.playback import play
import os

class ChatController:
    def __init__(self, api_key_path='api_key.txt'):
        self.api_key = self.read_api_key(api_key_path)
        openai.api_key = self.api_key

    def read_api_key(self, file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()

    def record_audio(self, duration=8, fs=44100, output_file='audio.wav'):
        print("Recording...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        print("Recording finished.")
        write(output_file, fs, audio_data)
        return output_file

    def transcribe_audio(self, audio_file):
        headers = {'Authorization': f'Bearer {self.api_key}'}
        files = {
            'file': (audio_file, open(audio_file, 'rb')),
            'model': (None, 'whisper-1'),
        }
        response = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, files=files)

        if response.status_code == 200:
            transcript = response.json()['text']
            return transcript
        else:
            print(f"Transcription error: {response.status_code}, {response.text}")
            return None
    
    
    def generate_response(self, text):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text}],
            max_tokens=150
        )
        return response.choices[0].message.content

    def text_to_speech_streaming(self, text):
        print("Iniciando el proceso de Text-to-Speech con streaming...")  # Debug

        try:
           # Realizar la solicitud a la API de Text-to-Speech con streaming
           response = openai.audio.speech.create(
               model="tts-1",
               voice="alloy",  # Cambiar por la voz que prefieras
               input=text,
               stream=True
               )
           with open("output.mp3", "wb") as f:
            for chunk in response:
                f.write(chunk)
           print("Audio guardado exitosamente como 'output.mp3'.")
        except Exception as e:
           print(f"Error en la solicitud de Text-to-Speech: {e}")


    # Función para reproducir el archivo de audio generado
    def reproducir_audio(audio_file):
        try:
            print(f"Reproduciendo el archivo de audio: {audio_file}")
            sound = AudioSegment.from_mp3(audio_file)
            play(sound)
            print("Reproducción de audio finalizada.")
        except Exception as e:
            print(f"Error al reproducir el archivo de audio: {e}")

# Example usage
if __name__ == '__main__':
    chat = ChatController()
    audio_file = chat.record_audio()
    text = chat.transcribe_audio(audio_file)
    print(f"Transcribed text: {text}")
    response = chat.generate_response(text)
    print(f"Generated response: {response}")
    chat.text_to_speech_streaming(response)
    chat.play_audio("output.mp3")
