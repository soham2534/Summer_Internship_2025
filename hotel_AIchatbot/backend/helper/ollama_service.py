import requests
from fastapi import HTTPException

class OllamaService:
    def __init__(self, config):
        self.ollama_host = config["ollama_host"]
        self.ollama_model = config["ollama_model"]
    
    def chat(self, messages, temperature=0.7, max_tokens=700):
        """Interact with Ollama API"""
        try:
            payload = {
                "model": self.ollama_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(f"{self.ollama_host}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "message" not in result or "content" not in result["message"]:
                raise ValueError("Unexpected Ollama response format")
            
            return result["message"]["content"].strip()
            
        except requests.RequestException as e:
            print(f"Ollama chat error: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama chat error: {e}")