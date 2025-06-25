import os
from gtts import gTTS
from fastapi import HTTPException
from fastapi.responses import FileResponse

class AudioService:
    def __init__(self, audio_dir: str):
        self.audio_dir = audio_dir
        
    async def generate_audio(self, text: str, audio_id: str) -> str:
        """Generate audio file from text using gTTS"""
        audio_path = os.path.join(self.audio_dir, f"{audio_id}.mp3")
        
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_path)
            return f"/audio/{audio_id}.mp3"
        except Exception as e:
            print(f"TTS generation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate audio response")
    
    async def get_audio_file(self, filename: str):
        """Get audio file by filename"""
        audio_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(audio_path)