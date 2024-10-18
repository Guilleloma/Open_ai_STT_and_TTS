from openai import OpenAI
import os

# Crear el cliente OpenAI con la API key
client = OpenAI(api_key='TU_API_KEY')

# Función para convertir el texto a voz utilizando streaming real-time
def texto_a_voz_streaming(client, texto):
    print("Iniciando el proceso de Text-to-Speech con streaming...")  # Debug

    try:
        # Realizar la solicitud a la API de Text-to-Speech con streaming
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Cambiar por la voz que prefieras
            input=texto
        )
        # Transmitir a un archivo mientras se reproduce
        response.stream_to_file("output.mp3")
        print("Audio guardado exitosamente como 'output.mp3'.")
    except Exception as e:
        print(f"Error en la solicitud de Text-to-Speech: {e}")

