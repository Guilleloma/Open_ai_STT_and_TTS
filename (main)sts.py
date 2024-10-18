import base64
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import numpy as np
import wave



# Función para leer la clave de API desde el archivo
def read_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        return f.read().strip()
    
# Crear el cliente de OpenAI con la clave de API
client = OpenAI(api_key=read_api_key())

# Función para reproducir un archivo de audio WAV
def play_audio(file_path):
    # Abrir el archivo de audio
    with wave.open(file_path, 'rb') as f:
        # Obtener los parámetros del audio
        fs = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    
    # Reproducir el audio
    print(f"Reproduciendo {file_path}...")
    sd.play(data, fs)
    sd.wait()  # Esperar a que termine la reproducción

# Función para grabar audio
def record_audio(duration=8, fs=44100, output_file='audio.wav'):
    print("Grabando...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Esperar a que la grabación termine
    print("Grabación terminada.")

    # Guardar el audio en un archivo WAV
    write(output_file, fs, audio_data)
    return output_file

# Función para enviar el audio al modelo y obtener una respuesta en audio
def audio_sts(audio_file):
    print(f"Leyendo audio desde: {audio_file}")
    # Leer el archivo de audio y convertirlo en base64
    with open(audio_file, "rb") as audio_data:
        wav_data = audio_data.read()

    encoded_string = base64.b64encode(wav_data).decode('utf-8')
    print("Audio codificado en base64")

    # Enviar la solicitud a OpenAI
    print("Enviando solicitud al modelo GPT-4o para generar una respuesta...")
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav"
                        }
                    }
                ]
            },
        ]
    )
     # Imprimir la respuesta completa del modelo
    print("Respuesta completa del modelo:", completion)

    # Guardar la respuesta de audio
    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open("response_audio.wav", "wb") as f:
        f.write(wav_bytes)
    # Reproducir el audio guardado
    play_audio("response_audio.wav")
    print("Respuesta de audio guardada y reproducida.")



# Función principal que maneja el flujo de speech-to-speech
def main():
    print("Iniciando flujo de trabajo speech-to-speech...")
    recorded_file = record_audio()  # Graba 8 segundos de audio
    audio_sts(recorded_file)  # Procesa el archivo grabado

if __name__ == "__main__":
    main()
