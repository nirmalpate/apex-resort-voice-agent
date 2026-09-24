import asyncio
import websockets
import json
import io
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

recognizer = sr.Recognizer()

def record_speech_from_mic(duration=5, sample_rate=16000) -> str:
    """Records audio directly via sounddevice without PyAudio or C++ build tools."""
    print(f"\n[LISTENING...] Speak into your microphone now ({duration}s)...")
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("[PROCESSING...] Converting speech to text...")

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
        wav_buffer.seek(0)

        with sr.AudioFile(wav_buffer) as source:
            sr_audio = recognizer.record(source)

        text = recognizer.recognize_google(sr_audio, language="en-US")
        print(f"[YOU SAID]: {text}")
        return text
    except sr.UnknownValueError:
        print("[MIC] Could not understand audio. Please try speaking again.")
        return ""
    except Exception as err:
        print(f"[ERROR] Mic recording issue: {err}")
        return ""

def play_audio_bytes(audio_bytes: bytes):
    """Plays audio response bytes directly through local speakers."""
    if not audio_bytes:
        return
    try:
        audio_stream = io.BytesIO(audio_bytes)
        data, fs = sf.read(audio_stream)
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"[AUDIO ERROR] Could not play sound: {e}")

async def run_live_voice_session():
    uri = "ws://localhost:8000/ws/voice"
    print(f"Connecting to Apex Voice Server at {uri}...")

    # Disable ping timeouts so connection stays open during local LLM inference
    async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
        # 1. Receive Greeting Payload
        greeting_json_raw = await websocket.recv()
        greeting_data = json.loads(greeting_json_raw)
        
        greeting_audio_bytes = await websocket.recv()
        
        print(f"\n[MAYA]: {greeting_data['content']}")
        play_audio_bytes(greeting_audio_bytes)

        # 2. Interactive Voice Loop
        while True:
            user_text = record_speech_from_mic(duration=5)
            
            if not user_text:
                continue

            if user_text.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("Ending live voice session.")
                break

            await websocket.send(json.dumps({"message": user_text}))

            response_json_raw = await websocket.recv()
            response_data = json.loads(response_json_raw)

            response_audio_bytes = await websocket.recv()

            print(f"\n[MAYA]: {response_data['content']}")
            play_audio_bytes(response_audio_bytes)

if __name__ == "__main__":
    asyncio.run(run_live_voice_session())
