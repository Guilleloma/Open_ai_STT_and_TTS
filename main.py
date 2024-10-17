#main.py
from stt import record_audio, speech_to_text
import openai  # Usamos openai para obtener la versión, no OpenAI como clase
from openai import OpenAI  # Cambiado según la migración
import os

# Función para leer la clave de la API antes de crear el cliente OpenAI
def read_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        return f.read().strip()

def main():
    # Obtenemos la versión correctamente usando openai
    print("Versión de OpenAI:", openai.__version__)  
    print("Ruta del módulo OpenAI:", openai.__file__)

    # Crear el cliente de OpenAI usando la API key
    client = OpenAI(api_key=read_api_key())

    # Grabar audio desde el micrófono
    audio_file = record_audio(duration=8, output_file='audio.wav')
    
    # Transcribir el audio a texto
    texto_transcrito = speech_to_text(audio_file)
    print(f"Texto transcrito: {texto_transcrito}")
    
    # Generar una respuesta utilizando ChatGPT
    respuesta = procesar_texto(client, texto_transcrito)  # Pasa el cliente a la función
    print(f"Respuesta generada: {respuesta}")

def procesar_texto(client, texto):
    # Generar una respuesta con el modelo GPT-3.5-turbo utilizando la API correcta de ChatCompletion
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': texto}],  # 'messages' es el formato correcto
        max_tokens=150  # Limita el número de tokens en la respuesta
    )
    
    return response.choices[0].message.content  # Acceso correcto a la respuesta

if __name__ == '__main__':
    main()
