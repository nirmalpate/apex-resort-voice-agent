import os
import io
import json
import uuid
import asyncio
import edge_tts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from apex_voice_agent import receptionist_app
from langchain_core.messages import HumanMessage

app = FastAPI(title="Apex Resort Voice Receptionist API")

async def text_to_speech_bytes(text: str) -> bytes:
    """Synthesizes text into MP3 bytes directly using Edge-TTS."""
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
    return mp3_buffer.getvalue()

@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"apex_voice_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"\n[SERVER] Client connected! Session ID: {session_id}")
    
    greeting = "Hello! Welcome to Apex Luxury Resort. My name is Maya. How may I assist you with your booking today?"
    audio_bytes = await text_to_speech_bytes(greeting)
    
    await websocket.send_json({"type": "text", "content": greeting})
    await websocket.send_bytes(audio_bytes)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_text = payload.get("message", "")

            if not user_text:
                continue

            print(f"Guest ({session_id}): {user_text}")

            # Run LangGraph pipeline off the main event loop thread
            result = await asyncio.to_thread(
                receptionist_app.invoke,
                {"messages": [HumanMessage(content=user_text)]},
                config
            )
            
            response_text = result["messages"][-1].content
            print(f"Receptionist ({session_id}): {response_text}")

            audio_response = await text_to_speech_bytes(response_text)

            await websocket.send_json({"type": "text", "content": response_text})
            await websocket.send_bytes(audio_response)

    except WebSocketDisconnect:
        print(f"[SERVER] Client disconnected: {session_id}")
    except Exception as e:
        print(f"[SERVER ERROR]: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apex_server:app", host="0.0.0.0", port=8000, reload=True)
