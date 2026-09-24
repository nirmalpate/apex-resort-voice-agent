# apex-resort-voice-agent
AI Voice Receptionist built with LangGraph, FastAPI, and Edge-TTS.
# Apex Resort AI Voice Receptionist

An end-to-end, real-time AI voice receptionist pipeline using LangGraph, Ollama (Llama 3.2), FastAPI WebSockets, Edge-TTS, and Snowflake.

## Features
- **Local Speech Recognition**: Audio capture via `sounddevice` and `speech_recognition`.
- **Stateful Conversational Graph**: Orchestrated via `langgraph` with structured state extraction using `pydantic`.
- **Database Integration**: Queries room availability and rates directly from Snowflake.
- **Low-Latency Audio Streaming**: Synthesizes response audio on-the-fly using `edge-tts`.

## File Overview
- `apex_voice_agent.py`: LangGraph state machine, Snowflake tool definitions, and Llama 3.2 integration.
- `apex_server.py`: FastAPI server managing WebSocket sessions and Edge-TTS synthesis.
- `apex_client.py`: Microphone client capturing speech input and playing synthesized audio.

## Prerequisites
- Python 3.10+
- Ollama running locally with the `llama3.2` model
- Snowflake Account with active credentials

## Getting Started

1. **Clone the repository**:
   ```powershell
   git clone [https://github.com/](https://github.com/)<YOUR-USERNAME>/apex-resort-voice-agent.git
   cd apex-resort-voice-agent
