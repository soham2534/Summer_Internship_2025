from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import os
from gtts import gTTS
from dotenv import load_dotenv
from helper.config import load_config
from helper.models import Message, BookingDetails
from helper.hotel_service import HotelService
from helper.ollama_service import OllamaService
from helper.session_manager import SessionManager
from helper.booking_service import BookingService
from helper.audio_service import AudioService

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = load_config()
AUDIO_DIR = config["audio_dir"]

# Initialize services
hotel_service = HotelService(config)
ollama_service = OllamaService(config)
session_manager = SessionManager()
booking_service = BookingService(hotel_service, ollama_service, session_manager)
audio_service = AudioService(AUDIO_DIR)

# Ensure audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Chat endpoint
@app.post("/chat/{session_id}")
async def chat(session_id: str, msg: Message):
    if not msg.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Process the chat message
        response_data = await booking_service.process_chat_message(session_id, msg.message)
        
        # Generate audio for the response
        audio_id = str(uuid.uuid4())
        audio_url = await audio_service.generate_audio(response_data["reply"], audio_id)
        response_data["audio_url"] = audio_url
        
        return response_data
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {e}")

# Confirm booking endpoint
@app.post("/confirm/{session_id}")
async def confirm_booking(session_id: str, details: BookingDetails):
    try:
        # Process booking confirmation
        response_data = await booking_service.confirm_booking(session_id, details)
        
        # Generate audio for the response
        audio_id = str(uuid.uuid4())
        audio_url = await audio_service.generate_audio(response_data["reply"], audio_id)
        response_data["audio_url"] = audio_url
        
        return response_data
        
    except Exception as e:
        print(f"Booking confirmation error: {e}")
        raise HTTPException(status_code=500, detail=f"Booking confirmation error: {e}")

# Reset endpoint
@app.post("/reset/{session_id}")
async def reset_last_message(session_id: str):
    return session_manager.reset_last_message(session_id)

# Audio endpoint
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    return await audio_service.get_audio_file(filename)

# Hotels endpoint
@app.get("/hotels")
async def get_hotels():
    return {"hotels": hotel_service.get_all_hotels()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)